"""Train a Random Forest tree-type classifier from satellite embeddings.

The script expects Google Satellite Embedding CSV exports created by
dataCollection/exportSatelliteEmbeddings.py. It maps McCoy inventory species
names into the requested tree-type classes, splits by tree identifier so years
for the same tree do not leak across train/validation/test, and saves model
artifacts and evaluation tables.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_EMBEDDING_DIR = Path("C:/Users/larki/Desktop/PollenSense/training/SatelliteEmbedding")
DEFAULT_OUT_DIR = Path("C:/Users/larki/Desktop/PollenSense/training/TreeTypeRandomForest")
EMBEDDING_BANDS = [f"A{i:02d}" for i in range(64)]
TARGET_CLASSES = [
    "Quercus",
    "Cupressaceae",
    "Morus",
    "Ulmus",
    "Fraxinus",
    "Betula",
    "Acer",
    "Populus",
    "Pinaceae",
    "Other",
]
NAMED_CLASSES = [label for label in TARGET_CLASSES if label != "Other"]
BINARY_CLASSES = ["Other", "NamedTaxa"]
EXPANDED_OTHER_GENERA = [
    "Tilia",
    "Gleditsia",
    "Prunus",
    "Platanus",
    "Pyrus",
    "Sabal",
    "Malus",
    "Lagerstroemia",
    "Zelkova",
    "Cercis",
    "Magnolia",
    "Celtis",
    "Liquidambar",
    "Syringa",
]
INTERNAL_TAXON_CLASSES = NAMED_CLASSES + EXPANDED_OTHER_GENERA
INTERNAL_TO_FINAL_CLASS = {label: label for label in NAMED_CLASSES}
INTERNAL_TO_FINAL_CLASS.update({label: "Other" for label in EXPANDED_OTHER_GENERA})
NON_TREE_PATTERN = re.compile(
    r"vacant|stump|empty planting|planting site|planting space|available site|removed tree|tree removed",
    re.IGNORECASE,
)
PINACEAE_GENERA = {
    "Abies",
    "Cedrus",
    "Keteleeria",
    "Larix",
    "Nothotsuga",
    "Picea",
    "Pinus",
    "Pseudolarix",
    "Pseudotsuga",
    "Tsuga",
}
CUPRESSACEAE_GENERA = {
    "Actinostrobus",
    "Athrotaxis",
    "Austrocedrus",
    "Callitris",
    "Calocedrus",
    "Chamaecyparis",
    "Cryptomeria",
    "Cunninghamia",
    "Cupressus",
    "Diselma",
    "Fitzroya",
    "Glyptostrobus",
    "Hesperocyparis",
    "Juniperus",
    "Libocedrus",
    "Metasequoia",
    "Microbiota",
    "Neocallitropsis",
    "Platycladus",
    "Sequoia",
    "Sequoiadendron",
    "Taiwania",
    "Taxodium",
    "Tetraclinis",
    "Thuja",
    "Thujopsis",
    "Widdringtonia",
    "Xanthocyparis",
}
GENUS_CLASSES = {
    "Quercus": "Quercus",
    "Morus": "Morus",
    "Ulmus": "Ulmus",
    "Fraxinus": "Fraxinus",
    "Betula": "Betula",
    "Acer": "Acer",
    "Populus": "Populus",
}
METADATA_COLS = [
    "tree_uid",
    "mccoy_file",
    "mccoy_city_file",
    "mccoy_batch",
    "mccoy_row",
    "longitude",
    "latitude",
    "tree_ID",
    "city_ID",
    "city",
    "state",
    "common_name",
    "scientific_name",
    "embedding_year",
    "internal_tree_type",
    "tree_type",
    "split",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a Random Forest classifier for selected tree taxa from satellite embeddings."
    )
    parser.add_argument("--embedding-dir", type=Path, default=DEFAULT_EMBEDDING_DIR)
    parser.add_argument("--pattern", default="mccoy_satellite_embedding_*.csv")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        help="Folder for cached preprocessed modeling tables. Defaults to <out-dir>/cache.",
    )
    parser.add_argument("--rebuild-cache", action="store_true", help="Ignore and replace any matching preprocessing cache.")
    parser.add_argument("--no-cache", action="store_true", help="Disable preprocessing cache reads and writes.")
    parser.add_argument("--chunk-size", type=int, default=100_000)
    parser.add_argument(
        "--include-non-tree-records",
        action="store_true",
        help="Keep obvious vacant/stump/non-tree inventory records instead of screening them out.",
    )
    parser.add_argument("--max-rows-per-class-split", type=int, default=50_000)
    parser.add_argument("--train-frac", type=float, default=0.70)
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--test-frac", type=float, default=0.15)
    parser.add_argument("--n-estimators", type=int, default=500)
    parser.add_argument("--max-depth", type=int)
    parser.add_argument("--min-samples-leaf", type=int, default=2)
    parser.add_argument("--min-samples-split", type=int, default=2)
    parser.add_argument(
        "--max-features",
        default="sqrt",
        help="Random Forest max_features value, e.g. sqrt, log2, none, or a numeric value.",
    )
    parser.add_argument(
        "--max-samples",
        type=float,
        help="Bootstrap sample fraction per tree. Requires bootstrap=True; leave unset to use all rows.",
    )
    parser.add_argument("--bootstrap", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--class-weight", default="balanced_subsample")
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--other-threshold",
        default="auto",
        help=(
            "Use 'auto' to tune on validation trees, or provide a numeric "
            "threshold. If the best non-Other probability is below this value, "
            "predict Other."
        ),
    )
    parser.add_argument("--threshold-min", type=float, default=0.0)
    parser.add_argument("--threshold-max", type=float, default=0.95)
    parser.add_argument("--threshold-step", type=float, default=0.025)
    parser.add_argument(
        "--threshold-objective",
        choices=[
            "row_accuracy",
            "row_macro_f1",
            "row_weighted_f1",
            "tree_accuracy",
            "tree_macro_f1",
            "tree_weighted_f1",
        ],
        default="tree_macro_f1",
        help="Validation metric used to choose the auto NamedTaxa threshold.",
    )
    parser.add_argument("--save-sampled-data", action="store_true")
    return parser.parse_args()


def parse_max_features(value: object):
    if value is None:
        return "sqrt"
    text = str(value).strip().lower()
    if text in {"none", "null"}:
        return None
    if text in {"sqrt", "log2"}:
        return text

    number = float(text)
    if number.is_integer() and number >= 1:
        return int(number)
    return number


def scientific_name_to_genus(scientific_name: object) -> str:
    if pd.isna(scientific_name):
        return ""
    text = str(scientific_name).strip()
    if not text:
        return ""

    text = text.replace("×", "x")
    match = re.search(r"[A-Za-z]+", text)
    if not match:
        return ""
    genus = match.group(0)
    return genus[:1].upper() + genus[1:].lower()


def tree_type_from_scientific_name(scientific_name: object) -> str:
    genus = scientific_name_to_genus(scientific_name)
    if genus in GENUS_CLASSES:
        return GENUS_CLASSES[genus]
    if genus in CUPRESSACEAE_GENERA:
        return "Cupressaceae"
    if genus in PINACEAE_GENERA:
        return "Pinaceae"
    return "Other"


def internal_tree_type_from_scientific_name(scientific_name: object) -> str:
    genus = scientific_name_to_genus(scientific_name)
    final_type = tree_type_from_scientific_name(scientific_name)
    if final_type != "Other":
        return final_type
    if genus in EXPANDED_OTHER_GENERA:
        return genus
    return "Other"


def final_class_from_internal_label(label: str) -> str:
    return INTERNAL_TO_FINAL_CLASS.get(label, "Other")


def collapse_internal_probabilities(internal_probabilities: np.ndarray) -> np.ndarray:
    collapsed = np.zeros((len(internal_probabilities), len(TARGET_CLASSES)), dtype=float)
    for internal_idx, internal_label in enumerate(INTERNAL_TAXON_CLASSES):
        final_label = final_class_from_internal_label(internal_label)
        final_idx = TARGET_CLASSES.index(final_label)
        collapsed[:, final_idx] += internal_probabilities[:, internal_idx]
    return collapsed


def is_non_tree_record(df: pd.DataFrame) -> pd.Series:
    mask = pd.Series(False, index=df.index)
    for column in ["common_name", "scientific_name", "condition", "location_type"]:
        if column in df.columns:
            values = df[column].fillna("").astype(str)
            mask = mask | values.str.contains(NON_TREE_PATTERN, na=False)
    return mask


def stable_fraction(value: str) -> float:
    digest = hashlib.md5(value.encode("utf-8")).hexdigest()
    return int(digest[:12], 16) / float(16**12)


def split_from_uid(uid: str, train_frac: float, val_frac: float) -> str:
    frac = stable_fraction(uid)
    if frac < train_frac:
        return "train"
    if frac < train_frac + val_frac:
        return "val"
    return "test"


def make_tree_uid(df: pd.DataFrame) -> pd.Series:
    if "tree_ID" in df.columns and df["tree_ID"].notna().any():
        tree_id = df["tree_ID"].fillna("").astype(str).str.strip()
    else:
        tree_id = pd.Series([""] * len(df), index=df.index)

    fallback_parts = []
    for column in ["mccoy_file", "mccoy_row", "longitude", "latitude"]:
        if column in df.columns:
            fallback_parts.append(df[column].fillna("").astype(str))
        else:
            fallback_parts.append(pd.Series([""] * len(df), index=df.index))
    fallback = fallback_parts[0]
    for part in fallback_parts[1:]:
        fallback = fallback + "|" + part

    return np.where(tree_id != "", tree_id, fallback)


def read_embedding_columns(csv_path: Path) -> list[str]:
    return pd.read_csv(csv_path, nrows=0).columns.tolist()


def downsample_bucket(df: pd.DataFrame, max_rows: int, random_state: int) -> pd.DataFrame:
    if len(df) <= max_rows:
        return df
    return df.sample(n=max_rows, random_state=random_state)


def embedding_source_manifest(args: argparse.Namespace) -> list[dict]:
    csv_files = sorted(args.embedding_dir.glob(args.pattern))
    manifest = []
    for csv_path in csv_files:
        stat = csv_path.stat()
        manifest.append(
            {
                "name": csv_path.name,
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
            }
        )
    return manifest


def preprocessing_cache_payload(args: argparse.Namespace) -> dict:
    return {
        "cache_version": 1,
        "embedding_dir": str(args.embedding_dir.resolve()),
        "pattern": args.pattern,
        "chunk_size": args.chunk_size,
        "max_rows_per_class_split": args.max_rows_per_class_split,
        "train_frac": args.train_frac,
        "val_frac": args.val_frac,
        "test_frac": args.test_frac,
        "random_state": args.random_state,
        "include_non_tree_records": args.include_non_tree_records,
        "non_tree_pattern": NON_TREE_PATTERN.pattern,
        "target_classes": TARGET_CLASSES,
        "named_classes": NAMED_CLASSES,
        "expanded_other_genera": EXPANDED_OTHER_GENERA,
        "internal_taxon_classes": INTERNAL_TAXON_CLASSES,
        "internal_to_final_class": INTERNAL_TO_FINAL_CLASS,
        "pinaceae_genera": sorted(PINACEAE_GENERA),
        "cupressaceae_genera": sorted(CUPRESSACEAE_GENERA),
        "genus_classes": GENUS_CLASSES,
        "embedding_bands": EMBEDDING_BANDS,
        "sources": embedding_source_manifest(args),
    }


def preprocessing_cache_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    cache_dir = args.cache_dir if args.cache_dir is not None else args.out_dir / "cache"
    payload = preprocessing_cache_payload(args)
    cache_key = hashlib.md5(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:16]
    return cache_dir / f"tree_type_training_rows_{cache_key}.pkl", cache_dir / f"tree_type_training_rows_{cache_key}.json"


def load_or_build_training_rows(args: argparse.Namespace) -> pd.DataFrame:
    if args.no_cache:
        return load_training_rows(args)

    cache_path, meta_path = preprocessing_cache_paths(args)
    if cache_path.exists() and not args.rebuild_cache:
        print(f"Loading preprocessed training rows from cache: {cache_path}")
        data = pd.read_pickle(cache_path)
        data["tree_type"] = pd.Categorical(data["tree_type"], categories=TARGET_CLASSES, ordered=True)
        data["internal_tree_type"] = pd.Categorical(
            data["internal_tree_type"],
            categories=INTERNAL_TAXON_CLASSES + ["Other"],
            ordered=True,
        )
        return data

    data = load_training_rows(args)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    data.to_pickle(cache_path)
    with meta_path.open("w", encoding="utf-8") as handle:
        json.dump(preprocessing_cache_payload(args), handle, indent=2)
    print(f"Saved preprocessed training rows cache: {cache_path}")
    return data


def load_training_rows(args: argparse.Namespace) -> pd.DataFrame:
    csv_files = sorted(args.embedding_dir.glob(args.pattern))
    if not csv_files:
        raise FileNotFoundError(f"No embedding CSV files found in {args.embedding_dir} with pattern {args.pattern}")

    split_total = args.train_frac + args.val_frac + args.test_frac
    if not np.isclose(split_total, 1.0):
        raise ValueError("--train-frac + --val-frac + --test-frac must equal 1.0")

    buckets: dict[tuple[str, str], list[pd.DataFrame]] = defaultdict(list)
    read_cols = None
    label_counts = defaultdict(int)
    screened_non_tree_count = 0
    kept_counts = defaultdict(int)

    for csv_path in csv_files:
        columns = read_embedding_columns(csv_path)
        missing_bands = [band for band in EMBEDDING_BANDS if band not in columns]
        if missing_bands:
            print(f"Skipping {csv_path.name}; missing embedding bands: {missing_bands[:5]}")
            continue
        if "scientific_name" not in columns:
            print(f"Skipping {csv_path.name}; missing scientific_name")
            continue

        if read_cols is None:
            read_cols = [c for c in METADATA_COLS if c in columns and c not in {"tree_uid", "tree_type", "split"}]
            read_cols = list(dict.fromkeys(read_cols + EMBEDDING_BANDS))

        print(f"Reading {csv_path.name}")
        for chunk in pd.read_csv(csv_path, usecols=read_cols, chunksize=args.chunk_size, low_memory=False):
            chunk = chunk.dropna(subset=EMBEDDING_BANDS, how="any")
            if chunk.empty:
                continue

            if not args.include_non_tree_records:
                non_tree_mask = is_non_tree_record(chunk)
                screened_non_tree_count += int(non_tree_mask.sum())
                chunk = chunk.loc[~non_tree_mask].copy()
                if chunk.empty:
                    continue

            chunk["tree_uid"] = make_tree_uid(chunk)
            chunk["tree_type"] = chunk["scientific_name"].map(tree_type_from_scientific_name)
            chunk["internal_tree_type"] = chunk["scientific_name"].map(internal_tree_type_from_scientific_name)
            chunk["split"] = chunk["tree_uid"].map(
                lambda uid: split_from_uid(str(uid), args.train_frac, args.val_frac)
            )

            for tree_type, count in chunk["tree_type"].value_counts().items():
                label_counts[tree_type] += int(count)

            keep_cols = [c for c in METADATA_COLS if c in chunk.columns] + EMBEDDING_BANDS
            for (split, internal_tree_type), group in chunk[keep_cols].groupby(["split", "internal_tree_type"], sort=False):
                bucket_key = (split, internal_tree_type)
                current = pd.concat(buckets[bucket_key] + [group], ignore_index=True)
                current = downsample_bucket(
                    current,
                    max_rows=args.max_rows_per_class_split,
                    random_state=args.random_state,
                )
                buckets[bucket_key] = [current]
                kept_counts[bucket_key] = len(current)

    rows = []
    for frames in buckets.values():
        rows.extend(frames)
    if not rows:
        raise ValueError("No usable training rows were loaded.")

    data = pd.concat(rows, ignore_index=True)
    data["tree_type"] = pd.Categorical(data["tree_type"], categories=TARGET_CLASSES, ordered=True)
    data["internal_tree_type"] = pd.Categorical(
        data["internal_tree_type"],
        categories=INTERNAL_TAXON_CLASSES + ["Other"],
        ordered=True,
    )
    data = data.dropna(subset=["tree_type"]).sample(frac=1.0, random_state=args.random_state).reset_index(drop=True)

    print("\nRaw label counts before per-split caps:")
    for label in TARGET_CLASSES:
        print(f"{label}: {label_counts[label]}")
    if not args.include_non_tree_records:
        print(f"\nScreened obvious non-tree/vacant/stump records: {screened_non_tree_count}")

    print("\nKept rows by split and class:")
    print(data.groupby(["split", "tree_type"], observed=False).size().unstack(fill_value=0))
    print("\nKept rows by split and internal class:")
    print(data.groupby(["split", "internal_tree_type"], observed=False).size().unstack(fill_value=0))
    return data


def apply_other_threshold(probabilities: np.ndarray, class_names: list[str], threshold: float) -> np.ndarray:
    predictions = probabilities.argmax(axis=1)
    if threshold <= 0:
        return predictions

    other_idx = class_names.index("Other")
    non_other_indices = [i for i, name in enumerate(class_names) if name != "Other"]
    non_other_probs = probabilities[:, non_other_indices]
    best_non_other_pos = non_other_probs.argmax(axis=1)
    best_non_other_prob = non_other_probs[np.arange(len(probabilities)), best_non_other_pos]
    best_non_other_idx = np.array(non_other_indices)[best_non_other_pos]

    return np.where(best_non_other_prob >= threshold, best_non_other_idx, other_idx)


def predict_full_proba(model, x: np.ndarray, n_classes: int) -> np.ndarray:
    probabilities = model.predict_proba(x)
    full = np.zeros((len(x), n_classes), dtype=float)
    for source_col, class_idx in enumerate(model.classes_):
        full[:, int(class_idx)] = probabilities[:, source_col]
    return full


def predict_binary_named_probability(model, x: np.ndarray) -> np.ndarray:
    probabilities = model.predict_proba(x)
    named_prob = np.zeros(len(x), dtype=float)
    for source_col, class_idx in enumerate(model.classes_):
        if int(class_idx) == 1:
            named_prob = probabilities[:, source_col]
    return named_prob


def predict_named_taxon_probabilities(model, x: np.ndarray) -> np.ndarray:
    probabilities = model.predict_proba(x)
    full = np.zeros((len(x), len(INTERNAL_TAXON_CLASSES)), dtype=float)
    for source_col, class_idx in enumerate(model.classes_):
        full[:, int(class_idx)] = probabilities[:, source_col]
    return full


def combine_two_stage_probabilities(named_prob: np.ndarray, internal_taxon_probabilities: np.ndarray) -> np.ndarray:
    combined = np.zeros((len(named_prob), len(TARGET_CLASSES)), dtype=float)
    collapsed_taxon_probabilities = collapse_internal_probabilities(internal_taxon_probabilities)
    combined += collapsed_taxon_probabilities * named_prob[:, None]
    combined[:, TARGET_CLASSES.index("Other")] += 1.0 - named_prob
    return combined


def two_stage_probabilities(binary_model, taxon_model, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = df[EMBEDDING_BANDS].to_numpy(dtype=np.float32)
    named_prob = predict_binary_named_probability(binary_model, x)
    taxon_probabilities = predict_named_taxon_probabilities(taxon_model, x)
    combined = combine_two_stage_probabilities(named_prob, taxon_probabilities)
    return combined, named_prob, taxon_probabilities


def two_stage_predictions(named_prob: np.ndarray, taxon_probabilities: np.ndarray, threshold: float) -> np.ndarray:
    other_idx = TARGET_CLASSES.index("Other")
    best_internal = taxon_probabilities.argmax(axis=1)
    best_final = np.array(
        [TARGET_CLASSES.index(final_class_from_internal_label(INTERNAL_TAXON_CLASSES[i])) for i in best_internal],
        dtype=int,
    )
    return np.where(named_prob >= threshold, best_final, other_idx)


def aggregate_tree_probabilities(
    df: pd.DataFrame,
    probabilities: np.ndarray,
    class_names: list[str],
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    rows = []
    prob_rows = []
    y_true = []
    other_idx = class_names.index("Other")
    work = df.reset_index(drop=True)

    for tree_uid, group in work.groupby("tree_uid", sort=False):
        idx = group.index.to_numpy()
        mean_prob = probabilities[idx].mean(axis=0)
        first = group.iloc[0]
        row = {c: first[c] for c in METADATA_COLS if c in group.columns}
        row["tree_uid"] = tree_uid
        row["n_embedding_rows"] = int(len(group))
        row["n_embedding_years"] = int(group["embedding_year"].nunique()) if "embedding_year" in group.columns else np.nan
        rows.append(row)
        prob_rows.append(mean_prob)

        true_codes = group["tree_type"].cat.codes.to_numpy()
        non_other_codes = true_codes[true_codes != other_idx]
        if len(non_other_codes) > 0:
            y_true.append(int(non_other_codes[0]))
        else:
            y_true.append(other_idx)

    return pd.DataFrame(rows), np.vstack(prob_rows), np.array(y_true, dtype=int)


def aggregate_two_stage_probabilities(
    df: pd.DataFrame,
    combined_probabilities: np.ndarray,
    named_probabilities: np.ndarray,
    taxon_probabilities: np.ndarray,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    tree_df, combined_tree_probabilities, tree_y_true = aggregate_tree_probabilities(
        df,
        combined_probabilities,
        TARGET_CLASSES,
    )
    work = df.reset_index(drop=True)
    named_rows = []
    taxon_rows = []
    for _, group in work.groupby("tree_uid", sort=False):
        idx = group.index.to_numpy()
        named_rows.append(float(named_probabilities[idx].mean()))
        taxon_rows.append(taxon_probabilities[idx].mean(axis=0))

    return (
        tree_df,
        combined_tree_probabilities,
        np.array(named_rows, dtype=float),
        np.vstack(taxon_rows),
        tree_y_true,
    )


def score_predictions(y_true: np.ndarray, y_pred: np.ndarray, class_names: list[str]) -> dict:
    from sklearn.metrics import classification_report

    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(class_names))),
        target_names=class_names,
        digits=4,
        zero_division=0,
        output_dict=True,
    )
    return {
        "accuracy": float(report["accuracy"]),
        "macro_f1": float(report["macro avg"]["f1-score"]),
        "weighted_f1": float(report["weighted avg"]["f1-score"]),
    }


def write_evaluation_outputs(
    level_df: pd.DataFrame,
    probabilities: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    split: str,
    level: str,
    class_names: list[str],
    out_dir: Path,
) -> dict:
    from sklearn.metrics import classification_report, confusion_matrix

    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(class_names))),
        target_names=class_names,
        digits=4,
        zero_division=0,
        output_dict=True,
    )
    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(out_dir / f"{split}_{level}_classification_report.csv")

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    pd.DataFrame(cm, index=class_names, columns=class_names).to_csv(out_dir / f"{split}_{level}_confusion_matrix.csv")

    pred_df = level_df[[c for c in METADATA_COLS if c in level_df.columns]].copy()
    if "n_embedding_rows" in level_df.columns:
        pred_df["n_embedding_rows"] = level_df["n_embedding_rows"]
    if "n_embedding_years" in level_df.columns:
        pred_df["n_embedding_years"] = level_df["n_embedding_years"]
    pred_df["predicted_tree_type"] = [class_names[i] for i in y_pred]
    pred_df["predicted_probability"] = probabilities[np.arange(len(y_pred)), y_pred]
    for i, class_name in enumerate(class_names):
        pred_df[f"prob_{class_name}"] = probabilities[:, i]
    pred_df.to_csv(out_dir / f"{split}_{level}_predictions.csv", index=False)

    return {
        "split": split,
        "level": level,
        "rows": int(len(level_df)),
        "accuracy": float(report["accuracy"]),
        "macro_f1": float(report["macro avg"]["f1-score"]),
        "weighted_f1": float(report["weighted avg"]["f1-score"]),
    }


def split_probabilities(model, df: pd.DataFrame, split: str, class_names: list[str]) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    split_df = df[df["split"] == split].copy().reset_index(drop=True)
    if split_df.empty:
        return split_df, np.empty((0, len(class_names))), np.array([], dtype=int)

    y_true = split_df["tree_type"].cat.codes.to_numpy()
    x = split_df[EMBEDDING_BANDS].to_numpy(dtype=np.float32)
    probabilities = predict_full_proba(model, x, len(class_names))
    return split_df, probabilities, y_true


def threshold_grid(args: argparse.Namespace) -> np.ndarray:
    if args.threshold_step <= 0:
        raise ValueError("--threshold-step must be positive.")
    if args.threshold_max < args.threshold_min:
        raise ValueError("--threshold-max must be >= --threshold-min.")
    count = int(np.floor((args.threshold_max - args.threshold_min) / args.threshold_step)) + 1
    grid = args.threshold_min + np.arange(count + 1) * args.threshold_step
    return np.round(grid[grid <= args.threshold_max + 1e-12], 6)


def tune_named_taxa_threshold(
    val_df: pd.DataFrame,
    val_combined_probabilities: np.ndarray,
    val_named_probabilities: np.ndarray,
    val_taxon_probabilities: np.ndarray,
    args: argparse.Namespace,
) -> tuple[float, pd.DataFrame]:
    tree_df, _, tree_named_probabilities, tree_taxon_probabilities, tree_y_true = aggregate_two_stage_probabilities(
        val_df,
        val_combined_probabilities,
        val_named_probabilities,
        val_taxon_probabilities,
    )
    rows = []
    for threshold in threshold_grid(args):
        row_pred = two_stage_predictions(val_named_probabilities, val_taxon_probabilities, float(threshold))
        row_scores = score_predictions(val_df["tree_type"].cat.codes.to_numpy(), row_pred, TARGET_CLASSES)

        tree_pred = two_stage_predictions(tree_named_probabilities, tree_taxon_probabilities, float(threshold))
        tree_scores = score_predictions(tree_y_true, tree_pred, TARGET_CLASSES)
        rows.append(
            {
                "threshold": float(threshold),
                "val_row_accuracy": row_scores["accuracy"],
                "val_row_macro_f1": row_scores["macro_f1"],
                "val_row_weighted_f1": row_scores["weighted_f1"],
                "val_tree_accuracy": tree_scores["accuracy"],
                "val_tree_macro_f1": tree_scores["macro_f1"],
                "val_tree_weighted_f1": tree_scores["weighted_f1"],
                "val_tree_rows": int(len(tree_df)),
            }
        )

    tuning_df = pd.DataFrame(rows)
    objective_col = f"val_{args.threshold_objective}"
    tie_breakers = [
        objective_col,
        "val_tree_macro_f1",
        "val_row_macro_f1",
        "val_tree_weighted_f1",
        "val_row_weighted_f1",
        "val_tree_accuracy",
        "val_row_accuracy",
    ]
    tie_breakers = list(dict.fromkeys(tie_breakers))
    best = tuning_df.sort_values(tie_breakers, ascending=False).iloc[0]
    return float(best["threshold"]), tuning_df


def resolve_other_threshold(
    binary_model,
    taxon_model,
    data: pd.DataFrame,
    args: argparse.Namespace,
) -> float:
    if str(args.other_threshold).lower() != "auto":
        return float(args.other_threshold)

    val_df = data[data["split"] == "val"].copy().reset_index(drop=True)
    if val_df.empty:
        print("Validation split is empty; using other threshold 0.0.")
        return 0.0

    val_combined_probabilities, val_named_probabilities, val_taxon_probabilities = two_stage_probabilities(
        binary_model,
        taxon_model,
        val_df,
    )
    best_threshold, tuning_df = tune_named_taxa_threshold(
        val_df,
        val_combined_probabilities,
        val_named_probabilities,
        val_taxon_probabilities,
        args,
    )
    tuning_df.to_csv(args.out_dir / "named_taxa_threshold_tuning.csv", index=False)
    print(f"\nSelected NamedTaxa threshold from validation {args.threshold_objective}: {best_threshold:.3f}")
    return best_threshold


def evaluate_split(
    binary_model,
    taxon_model,
    df: pd.DataFrame,
    split: str,
    other_threshold: float,
    out_dir: Path,
) -> list[dict]:
    class_names = TARGET_CLASSES
    split_df = df[df["split"] == split].copy().reset_index(drop=True)
    if split_df.empty:
        return [{"split": split, "level": "row", "rows": 0}, {"split": split, "level": "tree", "rows": 0}]

    probabilities, named_probabilities, taxon_probabilities = two_stage_probabilities(
        binary_model,
        taxon_model,
        split_df,
    )
    y_true = split_df["tree_type"].cat.codes.to_numpy()
    row_pred = two_stage_predictions(named_probabilities, taxon_probabilities, other_threshold)
    row_summary = write_evaluation_outputs(
        level_df=split_df,
        probabilities=probabilities,
        y_true=y_true,
        y_pred=row_pred,
        split=split,
        level="row",
        class_names=class_names,
        out_dir=out_dir,
    )

    tree_df, tree_probabilities, tree_named_probabilities, tree_taxon_probabilities, tree_y_true = aggregate_two_stage_probabilities(
        split_df,
        probabilities,
        named_probabilities,
        taxon_probabilities,
    )
    tree_pred = two_stage_predictions(tree_named_probabilities, tree_taxon_probabilities, other_threshold)
    tree_summary = write_evaluation_outputs(
        level_df=tree_df,
        probabilities=tree_probabilities,
        y_true=tree_y_true,
        y_pred=tree_pred,
        split=split,
        level="tree",
        class_names=class_names,
        out_dir=out_dir,
    )
    return [row_summary, tree_summary]


def main() -> None:
    args = parse_args()
    if args.max_samples is not None and not args.bootstrap:
        raise ValueError("--max-samples requires --bootstrap. Remove --no-bootstrap or omit --max-samples.")
    if args.max_samples is not None and not 0 < args.max_samples <= 1:
        raise ValueError("--max-samples must be a fraction in (0, 1].")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    data = load_or_build_training_rows(args)
    train_df = data[data["split"] == "train"].copy()
    if train_df.empty:
        raise ValueError("Training split is empty.")

    x_train = train_df[EMBEDDING_BANDS].to_numpy(dtype=np.float32)
    y_train_binary = (train_df["tree_type"].astype(str) != "Other").astype(int).to_numpy()
    if len(np.unique(y_train_binary)) < 2:
        raise ValueError("Training split must contain both NamedTaxa and Other rows.")

    train_taxon_df = train_df[train_df["internal_tree_type"].astype(str) != "Other"].copy()
    if train_taxon_df.empty:
        raise ValueError("Training split has no target or expanded taxon classes.")

    internal_type = pd.Categorical(
        train_taxon_df["internal_tree_type"].astype(str),
        categories=INTERNAL_TAXON_CLASSES,
        ordered=True,
    )
    y_train_taxon = internal_type.codes
    x_train_taxon = train_taxon_df[EMBEDDING_BANDS].to_numpy(dtype=np.float32)
    if len(np.unique(y_train_taxon)) < 2:
        raise ValueError("Training split must contain at least two internal taxon classes.")

    print("\nTraining binary Random Forest: NamedTaxa vs Other...")
    from sklearn.ensemble import RandomForestClassifier

    class_weight = args.class_weight if args.class_weight.lower() != "none" else None
    max_features = parse_max_features(args.max_features)
    binary_model = RandomForestClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        min_samples_leaf=args.min_samples_leaf,
        min_samples_split=args.min_samples_split,
        max_features=max_features,
        max_samples=args.max_samples,
        bootstrap=args.bootstrap,
        class_weight=class_weight,
        n_jobs=args.n_jobs,
        random_state=args.random_state,
        verbose=1,
    )
    binary_model.fit(x_train, y_train_binary)

    print("\nTraining internal taxon Random Forest...")
    taxon_model = RandomForestClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        min_samples_leaf=args.min_samples_leaf,
        min_samples_split=args.min_samples_split,
        max_features=max_features,
        max_samples=args.max_samples,
        bootstrap=args.bootstrap,
        class_weight=class_weight,
        n_jobs=args.n_jobs,
        random_state=args.random_state,
        verbose=1,
    )
    taxon_model.fit(x_train_taxon, y_train_taxon)

    selected_other_threshold = resolve_other_threshold(
        binary_model=binary_model,
        taxon_model=taxon_model,
        data=data,
        args=args,
    )

    summaries = []
    for split in ["train", "val", "test"]:
        summaries.extend(
            evaluate_split(
                binary_model=binary_model,
                taxon_model=taxon_model,
                df=data,
                split=split,
                other_threshold=selected_other_threshold,
                out_dir=args.out_dir,
            )
        )

    pd.DataFrame(summaries).to_csv(args.out_dir / "split_metrics.csv", index=False)
    pd.DataFrame(
        {
            "feature": EMBEDDING_BANDS,
            "importance": binary_model.feature_importances_,
        }
    ).sort_values("importance", ascending=False).to_csv(args.out_dir / "binary_feature_importance.csv", index=False)
    pd.DataFrame(
        {
            "feature": EMBEDDING_BANDS,
            "importance": taxon_model.feature_importances_,
        }
    ).sort_values("importance", ascending=False).to_csv(args.out_dir / "taxon_feature_importance.csv", index=False)

    with (args.out_dir / "random_forest_two_stage_tree_type.pkl").open("wb") as handle:
        pickle.dump(
            {
                "binary_model": binary_model,
                "taxon_model": taxon_model,
                "target_classes": TARGET_CLASSES,
                "named_classes": NAMED_CLASSES,
                "expanded_other_genera": EXPANDED_OTHER_GENERA,
                "internal_taxon_classes": INTERNAL_TAXON_CLASSES,
                "internal_to_final_class": INTERNAL_TO_FINAL_CLASS,
                "selected_other_threshold": selected_other_threshold,
            },
            handle,
        )
    with (args.out_dir / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "model_strategy": "two_stage_target_gate_then_expanded_taxon_collapsed_reporting",
                "target_classes": TARGET_CLASSES,
                "named_classes": NAMED_CLASSES,
                "expanded_other_genera": EXPANDED_OTHER_GENERA,
                "internal_taxon_classes": INTERNAL_TAXON_CLASSES,
                "internal_to_final_class": INTERNAL_TO_FINAL_CLASS,
                "binary_classes": BINARY_CLASSES,
                "embedding_bands": EMBEDDING_BANDS,
                "embedding_dir": str(args.embedding_dir),
                "pattern": args.pattern,
                "cache_dir": str(args.cache_dir if args.cache_dir is not None else args.out_dir / "cache"),
                "cache_enabled": not args.no_cache,
                "rebuild_cache": args.rebuild_cache,
                "include_non_tree_records": args.include_non_tree_records,
                "non_tree_pattern": NON_TREE_PATTERN.pattern,
                "chunk_size": args.chunk_size,
                "max_rows_per_class_split": args.max_rows_per_class_split,
                "train_frac": args.train_frac,
                "val_frac": args.val_frac,
                "test_frac": args.test_frac,
                "n_estimators": args.n_estimators,
                "max_depth": args.max_depth,
                "min_samples_leaf": args.min_samples_leaf,
                "min_samples_split": args.min_samples_split,
                "max_features": args.max_features,
                "parsed_max_features": max_features,
                "max_samples": args.max_samples,
                "bootstrap": args.bootstrap,
                "class_weight": args.class_weight,
                "other_threshold_arg": args.other_threshold,
                "selected_other_threshold": selected_other_threshold,
                "threshold_objective": args.threshold_objective,
                "threshold_min": args.threshold_min,
                "threshold_max": args.threshold_max,
                "threshold_step": args.threshold_step,
                "random_state": args.random_state,
                "split_method": "deterministic hash of tree_uid",
                "tree_aggregation": "mean class probability across embedding rows for each tree_uid",
                "binary_training_rows": int(len(train_df)),
                "taxon_training_rows": int(len(train_taxon_df)),
            },
            handle,
            indent=2,
        )

    if args.save_sampled_data:
        data.to_parquet(args.out_dir / "sampled_training_rows.parquet", index=False)

    print("\nSaved outputs to:", args.out_dir)
    print(pd.DataFrame(summaries))


if __name__ == "__main__":
    main()
