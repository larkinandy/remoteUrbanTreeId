#!/usr/bin/env python3
"""Preprocess McCoy inputs and train the recommended multimodal taxon model.

The model combines a ConvNeXt-Tiny NAIP encoder, a Transformer encoder for
irregular Sentinel-2 + ERA5 sequences, an MLP/attention encoder for annual
Google Satellite Embeddings, and learned gated late fusion.  Two heads are
trained jointly: NamedTaxa vs Other and the seven-way conditional taxon head.

Large sequential split shards, compact indexes, and model artifacts are written
under E:/TreeID by default. NAIP pixels are not duplicated: shards contain
stable references into the city .npy files.

Example full-workflow smoke test (Albuquerque only, two epochs by default):

    python -m classifyModel.scripts.train_multimodal_taxa --dry-run

NAIP + Google Satellite Embedding pretraining while Sentinel-2/ERA5 are still
being collected:

    python -m classifyModel.scripts.train_multimodal_taxa --workflow naip-embedding-pretrain

Full run:

    python -m classifyModel.scripts.train_multimodal_taxa
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, f1_score
from torch.utils.data import DataLoader, Dataset, IterableDataset, get_worker_info


S2_RAW_COLS = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]
S2_INDEX_COLS = [
    "NDVI", "GNDVI", "CIg", "CIre", "MTCI", "MCARI", "NDVIre1",
    "NDVIre2", "REPI", "NDII", "MSAVI", "LAI_re", "LAI_ndvi",
]
S2_COLS = S2_RAW_COLS + S2_INDEX_COLS
ERA5_COLS = [
    "temp_mean_7d_c", "temp_mean_14d_c", "temp_mean_30d_c",
    "precip_sum_7d_mm", "precip_sum_14d_mm", "precip_sum_30d_mm",
    "srad_sum_7d_j_m2", "srad_sum_14d_j_m2", "srad_sum_30d_j_m2",
    "gdd_cum_ytd_base10_c",
]
EMBEDDING_COLS = [f"A{i:02d}" for i in range(64)]
NAMED_CLASSES = ["Quercus", "Acer", "Betula", "Ulmus", "Fraxinus", "Populus", "Pinaceae"]
FINAL_CLASSES = NAMED_CLASSES + ["Other"]
PINACEAE_GENERA = {
    "Abies", "Cathaya", "Cedrus", "Keteleeria", "Larix", "Nothotsuga",
    "Picea", "Pinus", "Pseudolarix", "Pseudotsuga", "Tsuga",
}
NON_TREE_PATTERN = re.compile(
    r"vacant|stump|empty planting|planting site|planting space|available site|removed tree|tree removed",
    re.IGNORECASE,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--workflow", choices=["full", "naip-embedding-pretrain"], default="full",
                   help="Training workflow. Pretraining mode uses only NAIP chips and GEE Satellite Embeddings.")
    p.add_argument("--data-root", type=Path, default=Path(r"E:/TreeID"),
                   help="Root containing Sentinel2, ERA5, SatelliteEmbedding, and NAIP_Chips folders.")
    p.add_argument("--inventory-dir", type=Path, default=Path(r"C:/Users/larki/Desktop/PollenSense/training/McCoy"))
    p.add_argument("--sentinel-dir", type=Path,
                   help="Sentinel-2 table directory; defaults to <data-root>/Sentinel2.")
    p.add_argument("--sentinel-cell-map-dir", type=Path,
                   default=Path("dataCollection/mccoy_sentinel_10m_cells_utm"),
                   help="Folder of per-city tree_to_sentinel10m_cell.csv crosswalks for reduced Sentinel-2 cells.")
    p.add_argument("--era5-dir", type=Path,
                   help="ERA5 table directory; defaults to <data-root>/ERA5.")
    p.add_argument("--embedding-dir", type=Path,
                   help="Google Satellite Embedding directory; defaults to <data-root>/SatelliteEmbedding.")
    p.add_argument("--naip-dir", type=Path,
                   help="NAIP chip directory; defaults to <data-root>/NAIP_Chips.")
    p.add_argument("--bulk-work-dir", type=Path,
                   help="Workspace for large sequential shards; defaults to <data-root>.")
    p.add_argument("--fast-work-dir", type=Path,
                   help="Workspace for indexes, metadata, and outputs; defaults to <data-root>.")
    p.add_argument("--model-input-dir", type=Path,
                   help="Shard directory; defaults to <bulk-work-dir>/ModelInputs.")
    p.add_argument("--output-dir", type=Path,
                   help="Model artifact directory; defaults to <fast-work-dir>/ModelOutputs.")
    p.add_argument("--inventory-pattern", default="*_Final_*.csv")
    p.add_argument("--sentinel-pattern", default="**/*.csv")
    p.add_argument("--era5-pattern", default="**/*.csv")
    p.add_argument("--embedding-pattern", default="mccoy_satellite_embedding_*.csv")
    p.add_argument("--dry-run", action="store_true", help="Use one city only and short training settings.")
    p.add_argument("--dry-run-city", default="Albuquerque",
                   help="City token to use when --dry-run is set. Common typo 'Alberquerque' is accepted.")
    p.add_argument("--rebuild-cache", action="store_true")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--dry-run-epochs", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--prefetch-factor", type=int, default=3)
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--patience", type=int, default=7)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--train-frac", type=float, default=0.70)
    p.add_argument("--val-frac", type=float, default=0.15)
    p.add_argument("--max-sequence-length", type=int, default=144)
    p.add_argument("--max-dry-run-trees", type=int, default=20_000)
    p.add_argument("--chunk-size", type=int, default=100_000)
    p.add_argument("--shard-size", type=int, default=25_000, help="Trees per sequential training shard.")
    p.add_argument("--allow-global-modality-scan", action="store_true",
                   help="Allow rescanning modality files whose path has no city token (slow for full runs).")
    p.add_argument("--gate-loss-weight", type=float, default=1.0)
    p.add_argument("--taxon-loss-weight", type=float, default=1.0)
    p.add_argument("--gate-threshold", default="auto", help="auto or a probability in [0,1].")
    p.add_argument("--taxon-threshold", default="auto", help="auto or a conditional probability in [0,1].")
    p.add_argument("--modality-dropout", type=float, default=0.15)
    p.add_argument("--keep-missing-active-modalities", action="store_true",
                   help="Keep rows that lack every modality used by the selected workflow.")
    p.add_argument("--no-pretrained-naip", action="store_true", help="Do not request ImageNet ConvNeXt weights.")
    p.add_argument("--no-amp", action="store_true", help="Disable CUDA BF16 automatic mixed precision.")
    p.add_argument("--compile", action="store_true", help="Use torch.compile after validating the uncompiled workflow.")
    args = p.parse_args()
    if args.sentinel_dir is None:
        args.sentinel_dir = args.data_root / "Sentinel2"
    if args.era5_dir is None:
        args.era5_dir = args.data_root / "ERA5"
    if args.embedding_dir is None:
        args.embedding_dir = args.data_root / "SatelliteEmbedding"
    if args.naip_dir is None:
        args.naip_dir = args.data_root / "NAIP_Chips"
    if args.bulk_work_dir is None:
        args.bulk_work_dir = args.data_root
    if args.fast_work_dir is None:
        args.fast_work_dir = args.data_root
    if args.model_input_dir is None:
        args.model_input_dir = args.bulk_work_dir / "ModelInputs"
    if args.output_dir is None:
        args.output_dir = args.fast_work_dir / "ModelOutputs"
    return args


def seed_everything(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def stable_fraction(text: str, salt: str = "") -> float:
    digest = hashlib.sha256((salt + text).encode("utf-8")).hexdigest()
    return int(digest[:15], 16) / float(16**15)


def genus_from_name(value: Any) -> str:
    if pd.isna(value):
        return ""
    match = re.search(r"[A-Za-z]+", str(value).replace("Ã—", "x").strip())
    return "" if not match else match.group(0).capitalize()


def taxon_from_name(value: Any) -> str:
    genus = genus_from_name(value)
    if genus in NAMED_CLASSES[:-1]:
        return genus
    if genus in PINACEAE_GENERA:
        return "Pinaceae"
    return "Other"


def normalize_token(value: Any) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


def canonical_city_token(value: str) -> str:
    token = normalize_token(value)
    if token in {"alberquerque", "albuqerque", "abq"}:
        return "albuquerque"
    return token


def workflow_tag(args: argparse.Namespace) -> str:
    base = args.workflow.replace("-", "_")
    if args.dry_run:
        return f"dry_run_{canonical_city_token(args.dry_run_city)}_{base}"
    return base


def active_modalities(args: argparse.Namespace) -> set[str]:
    if args.workflow == "naip-embedding-pretrain":
        return {"naip", "satellite_embedding"}
    return {"naip", "satellite_embedding", "sentinel_era5"}


def add_s2_indices(df: pd.DataFrame) -> pd.DataFrame:
    """Match the preliminary model's spectral indices."""
    if all(c in df.columns for c in S2_INDEX_COLS):
        return df
    missing = [c for c in S2_RAW_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Sentinel-2 table lacks raw bands needed for indices: {missing}")
    x = df.copy(); eps = 1e-10
    b = {c: pd.to_numeric(x[c], errors="coerce") for c in S2_RAW_COLS}
    x["NDVI"] = (b["B8"] - b["B4"]) / (b["B8"] + b["B4"] + eps)
    x["GNDVI"] = (b["B8"] - b["B3"]) / (b["B8"] + b["B3"] + eps)
    x["CIg"] = b["B8"] / (b["B3"] + eps) - 1
    x["CIre"] = b["B8A"] / (b["B5"] + eps) - 1
    x["MTCI"] = (b["B6"] - b["B5"]) / (b["B5"] - b["B4"] + eps)
    x["MCARI"] = ((b["B5"] - b["B4"]) - .2 * (b["B5"] - b["B3"])) * (b["B5"] / (b["B4"] + eps))
    x["NDVIre1"] = (b["B8"] - b["B5"]) / (b["B8"] + b["B5"] + eps)
    x["NDVIre2"] = (b["B8"] - b["B6"]) / (b["B8"] + b["B6"] + eps)
    x["REPI"] = 700 + 40 * (((b["B4"] + b["B7"]) / 2 - b["B5"]) / (b["B6"] - b["B5"] + eps))
    x["NDII"] = (b["B8"] - b["B11"]) / (b["B8"] + b["B11"] + eps)
    radicand = ((2 * b["B8"] + 1) ** 2 - 8 * (b["B8"] - b["B4"])).clip(lower=0)
    x["MSAVI"] = .5 * (2 * b["B8"] + 1 - np.sqrt(radicand))
    x["LAI_ndvi"] = np.clip(-np.log(((.69 - x["NDVI"]) / .59).clip(lower=eps)), 0, 6) / 6
    x["LAI_re"] = np.clip(3.618 * x["CIre"] - .118, 0, 6) / 6
    return x.replace([np.inf, -np.inf], np.nan)


def canonical_uid(df: pd.DataFrame, source_name: str = "") -> pd.Series:
    """Resolve identifiers emitted by the repository's collection scripts."""
    if "tree_uid" in df.columns:
        return df["tree_uid"].astype(str)
    if "mccoy_file" in df.columns and "mccoy_row" in df.columns:
        return df["mccoy_file"].astype(str) + "|" + df["mccoy_row"].astype("Int64").astype(str)
    if "uniqueID" in df.columns:
        return df["uniqueID"].astype(str)
    if "source_csv_row" in df.columns:
        return source_name + "|" + df["source_csv_row"].astype("Int64").astype(str)
    raise ValueError("Input needs tree_uid, uniqueID, mccoy_file+mccoy_row, or source_csv_row.")


def load_inventory(args: argparse.Namespace) -> pd.DataFrame:
    rows = []
    files = sorted(args.inventory_dir.glob(args.inventory_pattern))
    if args.dry_run:
        dry_token = canonical_city_token(args.dry_run_city)
        files = [p for p in files if dry_token in normalize_token(p.stem)]
    if not files:
        raise FileNotFoundError(f"No inventory files found under {args.inventory_dir}")
    for path in files:
        header = pd.read_csv(path, nrows=0).columns
        wanted = [c for c in ["tree_ID", "city_ID", "city", "state", "common_name", "scientific_name", "condition", "location_type"] if c in header]
        frame = pd.read_csv(path, usecols=wanted, low_memory=False)
        frame["mccoy_file"] = path.name
        frame["mccoy_row"] = np.arange(2, len(frame) + 2)
        frame["tree_uid"] = canonical_uid(frame)
        frame["taxon"] = frame["scientific_name"].map(taxon_from_name)
        mask = pd.Series(False, index=frame.index)
        for c in ["common_name", "scientific_name", "condition", "location_type"]:
            if c in frame:
                mask |= frame[c].fillna("").astype(str).str.contains(NON_TREE_PATTERN)
        rows.append(frame.loc[~mask])
    result = pd.concat(rows, ignore_index=True).drop_duplicates("tree_uid")
    if args.dry_run and len(result) > args.max_dry_run_trees:
        # Stratified cap keeps every target represented if present.
        result = result.groupby("taxon", group_keys=False).apply(
            lambda g: g.sample(min(len(g), max(100, args.max_dry_run_trees // len(FINAL_CLASSES))), random_state=args.seed)
        ).reset_index(drop=True)
    return result


def assign_splits(inventory: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    out = inventory.copy()
    cities = sorted(out["city"].fillna("").astype(str).unique()) if "city" in out else []
    if not args.dry_run and len(cities) >= 3:
        city_split = {}
        for city in cities:
            f = stable_fraction(city, "city-split-")
            city_split[city] = "train" if f < args.train_frac else ("val" if f < args.train_frac + args.val_frac else "test")
        out["split"] = out["city"].fillna("").astype(str).map(city_split)
        # Deterministic fallback in the unlikely event hashing leaves a split empty.
        if set(out["split"]) != {"train", "val", "test"}:
            ordered = sorted(cities, key=lambda c: stable_fraction(c, "fallback-"))
            city_split[ordered[-1]] = "test"; city_split[ordered[-2]] = "val"
            out["split"] = out["city"].fillna("").astype(str).map(city_split)
    else:
        # One-city dry run: tree-level hashing is only a workflow smoke test.
        values = out["tree_uid"].map(lambda x: stable_fraction(str(x), "tree-split-"))
        out["split"] = np.where(values < args.train_frac, "train", np.where(values < args.train_frac + args.val_frac, "val", "test"))
    return out


def city_files(root: Path, pattern: str, city: str | None, allow_global: bool) -> list[Path]:
    files = sorted(root.glob(pattern))
    if not files:
        return []
    if not city:
        return files
    token = canonical_city_token(city)
    matched = [p for p in files if token in normalize_token(p.relative_to(root))]
    if matched:
        return matched
    if allow_global:
        print(f"WARNING: no city-tagged files for {city} under {root}; rescanning {len(files)} global file(s).")
        return files
    raise FileNotFoundError(
        f"No modality files under {root} contain the city token '{city}'. For scalable preprocessing, "
        "place files in a city-named folder or include the city in each filename. Use "
        "--allow-global-modality-scan only for legacy/global inputs."
    )


def read_csv_collection(root: Path, pattern: str, needed_uids: set[str], columns: list[str], chunk_size: int,
                        city: str | None = None, allow_global: bool = False) -> pd.DataFrame:
    pieces = []
    files = city_files(root, pattern, city, allow_global)
    if not files:
        return pd.DataFrame(columns=["tree_uid"] + columns)
    for path in files:
        header = pd.read_csv(path, nrows=0).columns.tolist()
        id_cols = [c for c in ["tree_uid", "uniqueID", "mccoy_file", "mccoy_row"] if c in header]
        usecols = list(dict.fromkeys(id_cols + [c for c in columns if c in header]))
        if not id_cols:
            continue
        for chunk in pd.read_csv(path, usecols=usecols, chunksize=chunk_size, low_memory=False):
            chunk["tree_uid"] = canonical_uid(chunk)
            chunk = chunk[chunk["tree_uid"].isin(needed_uids)]
            if len(chunk):
                pieces.append(chunk)
    return pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame(columns=["tree_uid"] + columns)


def sentinel_cell_map_paths(root: Path, city: str | None) -> list[Path]:
    if not city:
        return sorted(root.glob("**/tree_to_sentinel10m_cell.csv"))
    token = canonical_city_token(city)
    direct = root / token / "tree_to_sentinel10m_cell.csv"
    paths = [direct] if direct.exists() else []
    legacy_albuquerque = Path("dataCollection/albuquerque_sentinel_10m_cells/tree_to_sentinel10m_cell.csv")
    if token == "albuquerque" and legacy_albuquerque.exists():
        paths.append(legacy_albuquerque)
    if not paths:
        paths = [
            p for p in sorted(root.glob("**/tree_to_sentinel10m_cell.csv"))
            if token in normalize_token(p.parent.name)
        ]
    return list(dict.fromkeys(paths))


def load_sentinel_cell_map(args: argparse.Namespace, needed_uids: set[str], city: str | None) -> pd.DataFrame:
    pieces = []
    for path in sentinel_cell_map_paths(args.sentinel_cell_map_dir, city):
        header = pd.read_csv(path, nrows=0).columns.tolist()
        if not {"source_file", "source_row", "reduced_id"}.issubset(header):
            continue
        frame = pd.read_csv(path, usecols=["source_file", "source_row", "reduced_id"], low_memory=False)
        frame["tree_uid"] = frame["source_file"].astype(str) + "|" + frame["source_row"].astype("Int64").astype(str)
        frame = frame[frame["tree_uid"].isin(needed_uids)]
        if len(frame):
            frame["row_index"] = pd.to_numeric(frame["reduced_id"], errors="coerce").astype("Int64")
            pieces.append(frame[["row_index", "tree_uid"]].dropna())
    if not pieces:
        return pd.DataFrame(columns=["row_index", "tree_uid"])
    return pd.concat(pieces, ignore_index=True).drop_duplicates()


def read_sentinel_collection(args: argparse.Namespace, needed_uids: set[str], city: str | None = None) -> pd.DataFrame:
    pieces = []
    files = city_files(args.sentinel_dir, args.sentinel_pattern, city, args.allow_global_modality_scan)
    if not files:
        return pd.DataFrame(columns=["tree_uid", "date"] + S2_COLS)
    cell_map = pd.DataFrame(columns=["row_index", "tree_uid"])
    for path in files:
        header = pd.read_csv(path, nrows=0).columns.tolist()
        id_cols = [c for c in ["tree_uid", "uniqueID", "mccoy_file", "mccoy_row", "source_csv_row"] if c in header]
        has_reduced_cell_id = "row_index" in header
        usecols = list(dict.fromkeys(id_cols + (["row_index"] if has_reduced_cell_id else []) + [c for c in ["date"] + S2_COLS if c in header]))
        if not id_cols and not has_reduced_cell_id:
            continue
        for chunk in pd.read_csv(path, usecols=usecols, chunksize=args.chunk_size, low_memory=False):
            if id_cols:
                chunk["tree_uid"] = canonical_uid(chunk)
                chunk = chunk[chunk["tree_uid"].isin(needed_uids)]
            elif has_reduced_cell_id:
                if cell_map.empty:
                    cell_map = load_sentinel_cell_map(args, needed_uids, city)
                if cell_map.empty:
                    continue
                chunk["row_index"] = pd.to_numeric(chunk["row_index"], errors="coerce").astype("Int64")
                chunk = chunk.merge(cell_map, on="row_index", how="inner")
            if len(chunk):
                pieces.append(chunk)
    return pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame(columns=["tree_uid", "date"] + S2_COLS)


def build_sequences(args: argparse.Namespace, uids: set[str], city: str | None = None) -> dict[str, np.ndarray]:
    s2 = read_sentinel_collection(args, uids, city)
    if s2.empty:
        print("WARNING: no matching Sentinel-2 rows; sequence modality will be marked missing.")
        return {}
    s2 = add_s2_indices(s2)
    s2["date"] = pd.to_datetime(s2["date"], errors="coerce").dt.normalize()
    era = read_csv_collection(args.era5_dir, args.era5_pattern, uids, ["date"] + ERA5_COLS,
                              args.chunk_size, city, args.allow_global_modality_scan)
    if not era.empty:
        era["date"] = pd.to_datetime(era["date"], errors="coerce").dt.normalize()
        era = era.drop_duplicates(["tree_uid", "date"])
        s2 = s2.merge(era[["tree_uid", "date"] + ERA5_COLS], on=["tree_uid", "date"], how="left")
    else:
        print("WARNING: no matching ERA5 rows; weather columns will be missing.")
        for c in ERA5_COLS: s2[c] = np.nan
    s2 = s2.dropna(subset=["date"]).sort_values(["tree_uid", "date"])
    result = {}
    feature_cols = S2_COLS + ERA5_COLS
    for uid, g in s2.groupby("tree_uid", sort=False):
        if len(g) > args.max_sequence_length:
            take = np.linspace(0, len(g) - 1, args.max_sequence_length).round().astype(int)
            g = g.iloc[take]
        values = g[feature_cols].apply(pd.to_numeric, errors="coerce").to_numpy(np.float32)
        mask = np.isfinite(values).astype(np.float32)
        values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
        doy = g["date"].dt.dayofyear.to_numpy(np.float32)
        delta = g["date"].diff().dt.days.fillna(0).to_numpy(np.float32) / 30.0
        calendar = np.column_stack([delta, np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25)]).astype(np.float32)
        result[str(uid)] = np.concatenate([values, mask, calendar], axis=1)
    return result


def build_embeddings(args: argparse.Namespace, uids: set[str], city: str | None = None) -> dict[str, np.ndarray]:
    emb = read_csv_collection(args.embedding_dir, args.embedding_pattern, uids, ["embedding_year"] + EMBEDDING_COLS,
                              args.chunk_size, city, args.allow_global_modality_scan)
    if emb.empty:
        print("WARNING: no matching Satellite Embedding rows; embedding modality will be marked missing.")
        return {}
    emb[EMBEDDING_COLS] = emb[EMBEDDING_COLS].apply(pd.to_numeric, errors="coerce")
    year_col = "embedding_year" if "embedding_year" in emb else None
    result = {}
    for uid, g in emb.groupby("tree_uid", sort=False):
        if year_col: g = g.sort_values(year_col)
        values = g[EMBEDDING_COLS].to_numpy(np.float32)
        result[str(uid)] = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    return result


def build_naip_lookup(args: argparse.Namespace, inventory: pd.DataFrame) -> dict[str, tuple[str, int]]:
    result = {}
    by_filename = {p.name: p for p in args.naip_dir.glob("*_chips.npy")}
    for index_path in sorted(args.naip_dir.glob("*_index.csv")):
        chip_name = index_path.name.replace("_index.csv", "_chips.npy")
        chip_path = by_filename.get(chip_name)
        if not chip_path:
            continue
        # Match the index's city token to its inventory file.
        token = index_path.name.replace("_index.csv", "").split("_", 1)[-1].lower()
        matches = inventory[inventory["mccoy_file"].str.lower().str.startswith(token + "_final_")]
        if matches.empty:
            continue
        index = pd.read_csv(index_path)
        if "covered" in index: index = index[index["covered"] == 1]
        row_to_uid = dict(zip(matches["mccoy_row"].astype(int), matches["tree_uid"].astype(str)))
        for row in index.itertuples(index=False):
            uid = row_to_uid.get(int(row.source_csv_row))
            if uid: result[uid] = (str(chip_path), int(row.chip_index))
    return result


def training_stats(records: list[dict[str, Any]], sequence_dim: int) -> dict[str, Any]:
    seq_sum = np.zeros(sequence_dim, np.float64); seq_sq = np.zeros(sequence_dim, np.float64); seq_n = 0
    emb_sum = np.zeros(64, np.float64); emb_sq = np.zeros(64, np.float64); emb_n = 0
    for r in records:
        seq = r["sequence"]
        if len(seq):
            seq_sum += seq.sum(0); seq_sq += (seq.astype(np.float64) ** 2).sum(0); seq_n += len(seq)
        emb = r["embeddings"]
        if len(emb):
            emb_sum += emb.sum(0); emb_sq += (emb.astype(np.float64) ** 2).sum(0); emb_n += len(emb)
    def finish(total, squares, n):
        mean = total / max(n, 1); std = np.sqrt(np.maximum(squares / max(n, 1) - mean**2, 1e-8))
        return mean.astype(np.float32), std.astype(np.float32)
    sm, ss = finish(seq_sum, seq_sq, seq_n); em, es = finish(emb_sum, emb_sq, emb_n)
    return {"sequence_mean": sm, "sequence_std": ss, "embedding_mean": em, "embedding_std": es}


def preprocess(args: argparse.Namespace) -> dict[str, Any]:
    inventory = assign_splits(load_inventory(args), args)
    print(f"Inventory trees: {len(inventory):,}; classes:\n{inventory['taxon'].value_counts().to_string()}")
    sequence_dim = 2 * (len(S2_COLS) + len(ERA5_COLS)) + 3
    tag = workflow_tag(args)
    active = active_modalities(args)
    shard_dir = args.model_input_dir / tag
    shard_dir.mkdir(parents=True, exist_ok=True)
    for old in shard_dir.glob("*_shard_*.pt"):
        old.unlink()

    buffers: dict[str, list[dict[str, Any]]] = {s: [] for s in ["train", "val", "test"]}
    shard_manifest: dict[str, list[dict[str, Any]]] = {s: [] for s in buffers}
    split_counts = {s: 0 for s in buffers}
    class_counts = {s: np.zeros(len(FINAL_CLASSES), np.int64) for s in buffers}
    modality_counts = {"sentinel_era5": 0, "satellite_embedding": 0, "naip": 0}
    dropped_missing_active = 0
    seq_sum = np.zeros(sequence_dim, np.float64); seq_sq = np.zeros(sequence_dim, np.float64); seq_n = 0
    emb_sum = np.zeros(64, np.float64); emb_sq = np.zeros(64, np.float64); emb_n = 0

    def flush(split: str) -> None:
        if not buffers[split]:
            return
        number = len(shard_manifest[split])
        path = shard_dir / f"{split}_shard_{number:05d}.pt"
        torch.save(buffers[split], path)
        shard_manifest[split].append({"path": str(path), "count": len(buffers[split])})
        buffers[split] = []

    # One source city at a time bounds peak preprocessing RAM. Modality exports
    # should be partitioned into city-named files/folders to avoid global rescans.
    for source_file, city_inventory in inventory.groupby("mccoy_file", sort=True):
        city_token = str(source_file).split("_Final_", 1)[0]
        uids = set(city_inventory["tree_uid"].astype(str))
        print(f"Preprocessing {city_token}: {len(city_inventory):,} trees")
        sequences = build_sequences(args, uids, city_token) if "sentinel_era5" in active else {}
        if "satellite_embedding" in active:
            try:
                embeddings = build_embeddings(args, uids, city_token)
            except FileNotFoundError as exc:
                if args.workflow != "naip-embedding-pretrain":
                    raise
                print(f"WARNING: {exc}; continuing without embeddings for {city_token}.")
                embeddings = {}
        else:
            embeddings = {}
        naip = build_naip_lookup(args, city_inventory) if "naip" in active else {}
        for row in city_inventory.itertuples(index=False):
            uid = str(row.tree_uid); label = str(row.taxon); split = str(row.split)
            record = {
                "tree_uid": uid, "city": str(getattr(row, "city", "")), "label": FINAL_CLASSES.index(label),
                "gate_label": int(label != "Other"), "taxon_label": NAMED_CLASSES.index(label) if label != "Other" else -1,
                "sequence": sequences.get(uid, np.empty((0, sequence_dim), np.float32)),
                "embeddings": embeddings.get(uid, np.empty((0, 64), np.float32)), "naip": naip.get(uid),
            }
            has_active = (
                ("sentinel_era5" in active and len(record["sequence"]) > 0)
                or ("satellite_embedding" in active and len(record["embeddings"]) > 0)
                or ("naip" in active and record["naip"] is not None)
            )
            if not has_active and not args.keep_missing_active_modalities:
                dropped_missing_active += 1
                continue
            buffers[split].append(record); split_counts[split] += 1; class_counts[split][record["label"]] += 1
            modality_counts["sentinel_era5"] += int(len(record["sequence"]) > 0)
            modality_counts["satellite_embedding"] += int(len(record["embeddings"]) > 0)
            modality_counts["naip"] += int(record["naip"] is not None)
            if split == "train":
                seq = record["sequence"]; emb = record["embeddings"]
                if len(seq):
                    seq_sum += seq.sum(0); seq_sq += (seq.astype(np.float64) ** 2).sum(0); seq_n += len(seq)
                if len(emb):
                    emb_sum += emb.sum(0); emb_sq += (emb.astype(np.float64) ** 2).sum(0); emb_n += len(emb)
            if len(buffers[split]) >= args.shard_size:
                flush(split)
        del sequences, embeddings, naip
    for split in buffers:
        flush(split)

    def finish(total, squares, n):
        mean = total / max(n, 1); std = np.sqrt(np.maximum(squares / max(n, 1) - mean**2, 1e-8))
        return mean.astype(np.float32), std.astype(np.float32)
    sm, ss = finish(seq_sum, seq_sq, seq_n); em, es = finish(emb_sum, emb_sq, emb_n)
    stats = {"sequence_mean": sm, "sequence_std": ss, "embedding_mean": em, "embedding_std": es}
    metadata = {
        "classes": FINAL_CLASSES, "named_classes": NAMED_CLASSES, "sequence_dim": sequence_dim,
        "s2_columns": S2_COLS, "era5_columns": ERA5_COLS, "embedding_columns": EMBEDDING_COLS,
        "stats": stats, "dry_run": args.dry_run,
        "split_counts": split_counts, "class_counts": {k: v.tolist() for k, v in class_counts.items()},
        "available_modalities": modality_counts, "shards": shard_manifest,
        "workflow": args.workflow, "active_modalities": sorted(active),
        "dropped_missing_active_modalities": dropped_missing_active,
        "storage_layout": {"bulk_shards": str(shard_dir), "fast_metadata": str(args.fast_work_dir / "Indexes")},
    }
    if sum(metadata["available_modalities"][m] for m in active) == 0:
        raise ValueError(
            f"No active modality records matched the McCoy inventory for workflow {args.workflow!r}. "
            "Ensure active modalities carry either tree_uid, uniqueID, or the mccoy_file+mccoy_row pair. "
            "For NAIP, ensure *_index.csv source_csv_row values correspond to McCoy CSV rows."
        )
    index_dir = args.fast_work_dir / "Indexes"; index_dir.mkdir(parents=True, exist_ok=True)
    serializable = dict(metadata); serializable["stats"] = {k: v.tolist() for k, v in stats.items()}
    meta_path = index_dir / f"{tag}_preprocessing.json"
    meta_path.write_text(json.dumps(serializable, indent=2), encoding="utf-8")
    return metadata


def load_cache(args: argparse.Namespace) -> dict[str, Any]:
    tag = workflow_tag(args)
    meta_path = args.fast_work_dir / "Indexes" / f"{tag}_preprocessing.json"
    if args.rebuild_cache or not meta_path.exists():
        return preprocess(args)
    raw = json.loads(meta_path.read_text(encoding="utf-8"))
    raw["stats"] = {k: np.asarray(v, np.float32) for k, v in raw["stats"].items()}
    missing = [entry["path"] for values in raw["shards"].values() for entry in values if not Path(entry["path"]).exists()]
    if missing:
        raise FileNotFoundError(f"Shard manifest references missing files; rerun with --rebuild-cache. First: {missing[0]}")
    return raw


class TreeDataset(Dataset):
    def __init__(self, records, stats, train=False, modality_dropout=0.0):
        self.records = records; self.stats = stats; self.train = train; self.modality_dropout = modality_dropout
        self.memmap: dict[str, np.ndarray] = {}

    def __len__(self): return len(self.records)

    def __getitem__(self, i):
        r = self.records[i]
        seq = r["sequence"].copy()
        if len(seq): seq = (seq - self.stats["sequence_mean"]) / self.stats["sequence_std"]
        emb = r["embeddings"].copy()
        if len(emb): emb = (emb - self.stats["embedding_mean"]) / self.stats["embedding_std"]
        image = np.zeros((3, 34, 34), np.float32); present = np.array([0, len(seq) > 0, len(emb) > 0], np.float32)
        if r["naip"] is not None:
            path, idx = r["naip"]
            if path not in self.memmap: self.memmap[path] = np.load(path, mmap_mode="r")
            chip = np.asarray(self.memmap[path][idx], dtype=np.float32)
            if chip.ndim == 3 and chip.shape[-1] >= 3:
                image = np.moveaxis(chip[..., :3], -1, 0) / 255.0; present[0] = 1
        if self.train and self.modality_dropout > 0:
            drop = (np.random.random(3) < self.modality_dropout) & (present > 0)
            if drop.all() and present.any(): drop[np.flatnonzero(present)[0]] = False
            if drop[0]: image.fill(0); present[0] = 0
            if drop[1]: seq = np.empty((0, self.stats["sequence_mean"].shape[0]), np.float32); present[1] = 0
            if drop[2]: emb = np.empty((0, 64), np.float32); present[2] = 0
        return {"image": image, "sequence": seq, "embeddings": emb, "present": present,
                "gate": r["gate_label"], "taxon": r["taxon_label"], "label": r["label"], "uid": r["tree_uid"]}


class ShardedTreeDataset(IterableDataset):
    """Load a bounded shard at a time and distribute shards across workers."""
    def __init__(self, manifest, stats, train=False, modality_dropout=0.0, seed=42):
        super().__init__(); self.manifest = manifest; self.stats = stats; self.train = train
        self.modality_dropout = modality_dropout; self.seed = seed; self.epoch = 0

    def __len__(self):
        return sum(int(item["count"]) for item in self.manifest)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self):
        worker = get_worker_info(); worker_id = 0 if worker is None else worker.id
        worker_count = 1 if worker is None else worker.num_workers
        rng = np.random.default_rng(self.seed + 1009 * self.epoch + worker_id)
        shard_indices = np.arange(len(self.manifest))
        if self.train: rng.shuffle(shard_indices)
        shard_indices = shard_indices[worker_id::worker_count]
        for shard_index in shard_indices:
            records = torch.load(self.manifest[int(shard_index)]["path"], map_location="cpu", weights_only=False)
            order = np.arange(len(records))
            if self.train: rng.shuffle(order)
            dataset = TreeDataset(records, self.stats, self.train, self.modality_dropout)
            for record_index in order:
                yield dataset[int(record_index)]
            del dataset, records


def collate(batch):
    b = len(batch); seq_dim = batch[0]["sequence"].shape[1]; max_s = max(1, max(len(x["sequence"]) for x in batch)); max_e = max(1, max(len(x["embeddings"]) for x in batch))
    seq = torch.zeros(b, max_s, seq_dim); seq_mask = torch.ones(b, max_s, dtype=torch.bool)
    emb = torch.zeros(b, max_e, 64); emb_mask = torch.ones(b, max_e, dtype=torch.bool)
    for i, x in enumerate(batch):
        if len(x["sequence"]): seq[i, :len(x["sequence"])] = torch.from_numpy(x["sequence"]); seq_mask[i, :len(x["sequence"])] = False
        else: seq_mask[i, 0] = False  # avoid an all-masked Transformer row; modality gate removes it later
        if len(x["embeddings"]): emb[i, :len(x["embeddings"])] = torch.from_numpy(x["embeddings"]); emb_mask[i, :len(x["embeddings"])] = False
    return {"image": torch.tensor(np.stack([x["image"] for x in batch])), "sequence": seq, "seq_mask": seq_mask,
            "embeddings": emb, "emb_mask": emb_mask, "present": torch.tensor(np.stack([x["present"] for x in batch])),
            "gate": torch.tensor([x["gate"] for x in batch]), "taxon": torch.tensor([x["taxon"] for x in batch]),
            "label": torch.tensor([x["label"] for x in batch]), "uid": [x["uid"] for x in batch]}


class AttentionPool(nn.Module):
    def __init__(self, dim): super().__init__(); self.score = nn.Linear(dim, 1)
    def forward(self, x, padding_mask):
        score = self.score(x).squeeze(-1).masked_fill(padding_mask, -1e4)
        return (x * torch.softmax(score, dim=1).unsqueeze(-1)).sum(1)


class MultimodalTaxonModel(nn.Module):
    def __init__(self, sequence_dim: int, pretrained_naip: bool = True, dropout: float = .25):
        super().__init__()
        try:
            from torchvision.models import ConvNeXt_Tiny_Weights, convnext_tiny
            try:
                weights = ConvNeXt_Tiny_Weights.DEFAULT if pretrained_naip else None
                backbone = convnext_tiny(weights=weights)
            except Exception as exc:
                print(f"WARNING: pretrained ConvNeXt weights unavailable ({exc}); using random initialization.")
                backbone = convnext_tiny(weights=None)
            self.naip = nn.Sequential(backbone.features, backbone.avgpool, nn.Flatten(), nn.Linear(768, 256), nn.GELU())
        except Exception as exc:
            print(f"WARNING: ConvNeXt unavailable ({exc}); using compact CNN fallback.")
            self.naip = nn.Sequential(nn.Conv2d(3, 32, 3, padding=1), nn.GELU(), nn.MaxPool2d(2),
                nn.Conv2d(32, 64, 3, padding=1), nn.GELU(), nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(64, 256), nn.GELU())
        self.seq_in = nn.Linear(sequence_dim, 192)
        layer = nn.TransformerEncoderLayer(192, 6, 384, dropout, batch_first=True, norm_first=True)
        self.seq_encoder = nn.TransformerEncoder(layer, 4); self.seq_pool = AttentionPool(192); self.seq_out = nn.Linear(192, 256)
        self.emb_in = nn.Sequential(nn.Linear(64, 128), nn.GELU())
        self.emb_pool = AttentionPool(128); self.emb_out = nn.Linear(128, 256)
        self.gates = nn.Sequential(nn.Linear(256 * 3 + 3, 128), nn.GELU(), nn.Linear(128, 3))
        self.fusion = nn.Sequential(nn.Linear(256, 256), nn.GELU(), nn.Dropout(dropout), nn.LayerNorm(256))
        self.gate_head = nn.Linear(256, 2); self.taxon_head = nn.Linear(256, len(NAMED_CLASSES))

    def forward(self, image, sequence, seq_mask, embeddings, emb_mask, present):
        # ConvNeXt accepts small chips but 64x64 stabilizes its downsampling stages.
        image = nn.functional.interpolate(image, size=(64, 64), mode="bilinear", align_corners=False)
        z0 = self.naip(image)
        if present[:, 1].sum() > 0:
            z1 = self.seq_out(self.seq_pool(self.seq_encoder(self.seq_in(sequence), src_key_padding_mask=seq_mask), seq_mask))
        else:
            z1 = torch.zeros_like(z0)
        if present[:, 2].sum() > 0:
            z2 = self.emb_out(self.emb_pool(self.emb_in(embeddings), emb_mask))
        else:
            z2 = torch.zeros_like(z0)
        stacked = torch.stack([z0, z1, z2], dim=1) * present.unsqueeze(-1)
        logits = self.gates(torch.cat([z0, z1, z2, present], dim=1)).masked_fill(present == 0, -1e4)
        # No-modality rows are retained for auditability; use equal zero embeddings.
        none = present.sum(1) == 0
        logits[none] = 0
        fused = self.fusion((stacked * torch.softmax(logits, 1).unsqueeze(-1)).sum(1))
        return self.gate_head(fused), self.taxon_head(fused)


def class_weights(labels: list[int], n: int, device):
    counts = np.bincount(np.asarray(labels), minlength=n).astype(np.float32)
    return class_weights_from_counts(counts, device)


def class_weights_from_counts(counts, device):
    counts = np.asarray(counts, dtype=np.float32)
    n = len(counts)
    weights = np.zeros(n, np.float32); nonzero = counts > 0; weights[nonzero] = 1 / np.sqrt(counts[nonzero])
    if nonzero.any(): weights[nonzero] *= nonzero.sum() / weights[nonzero].sum()
    return torch.tensor(weights, device=device)


def run_epoch(model, loader, device, optimizer, gate_loss, taxon_loss, args):
    training = optimizer is not None; model.train(training); total = 0.0; n = 0; started = time.perf_counter()
    outputs = defaultdict(list)
    for batch in loader:
        tensor_keys = ["image", "sequence", "seq_mask", "embeddings", "emb_mask", "present", "gate", "taxon", "label"]
        for key in tensor_keys: batch[key] = batch[key].to(device, non_blocking=True)
        if device.type == "cuda": batch["image"] = batch["image"].contiguous(memory_format=torch.channels_last)
        with torch.set_grad_enabled(training):
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                enabled=device.type == "cuda" and not args.no_amp):
                gate_logits, taxon_logits = model(batch["image"], batch["sequence"], batch["seq_mask"], batch["embeddings"], batch["emb_mask"], batch["present"])
                loss = args.gate_loss_weight * gate_loss(gate_logits, batch["gate"])
                named = batch["taxon"] >= 0
                if named.any(): loss = loss + args.taxon_loss_weight * taxon_loss(taxon_logits[named], batch["taxon"][named])
            if training:
                optimizer.zero_grad(set_to_none=True); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 2.0); optimizer.step()
        total += loss.item() * len(batch["label"]); n += len(batch["label"])
        outputs["gate_prob"].append(torch.softmax(gate_logits, 1)[:, 1].detach().float().cpu().numpy())
        outputs["taxon_prob"].append(torch.softmax(taxon_logits, 1).detach().float().cpu().numpy())
        outputs["label"].append(batch["label"].cpu().numpy()); outputs["uid"].extend(batch["uid"])
    elapsed = time.perf_counter() - started
    return {"loss": total / max(n, 1), "samples_per_second": n / max(elapsed, 1e-9),
            "gate_prob": np.concatenate(outputs["gate_prob"]),
            "taxon_prob": np.concatenate(outputs["taxon_prob"]), "label": np.concatenate(outputs["label"]), "uid": outputs["uid"]}


def predictions(result, gate_threshold, taxon_threshold):
    best = result["taxon_prob"].argmax(1); confidence = result["taxon_prob"].max(1)
    return np.where((result["gate_prob"] >= gate_threshold) & (confidence >= taxon_threshold), best, len(NAMED_CLASSES))


def tune_thresholds(result, gate_arg, taxon_arg):
    if gate_arg != "auto" and taxon_arg != "auto": return float(gate_arg), float(taxon_arg)
    gates = [float(gate_arg)] if gate_arg != "auto" else np.arange(.20, .86, .05)
    taxa = [float(taxon_arg)] if taxon_arg != "auto" else np.arange(.20, .86, .05)
    best = (-1.0, .5, .5)
    for g in gates:
        for t in taxa:
            score = f1_score(result["label"], predictions(result, g, t), labels=range(len(FINAL_CLASSES)), average="macro", zero_division=0)
            if score > best[0]: best = (score, float(g), float(t))
    return best[1], best[2]


def main() -> int:
    args = parse_args(); seed_everything(args.seed)
    if not (0 < args.train_frac < 1 and 0 < args.val_frac < 1 and args.train_frac + args.val_frac < 1):
        raise ValueError("--train-frac and --val-frac must be positive and sum to less than one")
    metadata = load_cache(args)
    if any(metadata["split_counts"][s] == 0 for s in ["train", "val", "test"]):
        raise ValueError(f"Empty split: {metadata['split_counts']}")
    print(
        f"Workflow: {metadata.get('workflow', args.workflow)}; "
        f"active_modalities: {metadata.get('active_modalities', sorted(active_modalities(args)))}"
    )
    print(
        f"Split counts: {metadata['split_counts']}; modalities: {metadata['available_modalities']}; "
        f"dropped_missing_active={metadata.get('dropped_missing_active_modalities', 0)}"
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu"); print(f"Device: {device}")
    loaders = {}; datasets = {}
    for split in ["train", "val", "test"]:
        ds = ShardedTreeDataset(metadata["shards"][split], metadata["stats"], split == "train",
                                args.modality_dropout if split == "train" else 0, args.seed)
        datasets[split] = ds
        loader_kwargs = dict(batch_size=args.batch_size, num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(), collate_fn=collate,
            persistent_workers=args.num_workers > 0 and split != "train")
        if args.num_workers > 0: loader_kwargs["prefetch_factor"] = args.prefetch_factor
        loaders[split] = DataLoader(ds, **loader_kwargs)
    model = MultimodalTaxonModel(metadata["sequence_dim"], not args.no_pretrained_naip).to(device)
    if device.type == "cuda": model = model.to(memory_format=torch.channels_last)
    if args.compile: model = torch.compile(model)
    train_counts = np.asarray(metadata["class_counts"]["train"], dtype=int)
    gate_weights = class_weights_from_counts([train_counts[-1], train_counts[:-1].sum()], device)
    taxon_weights = class_weights_from_counts(train_counts[:-1], device)
    gate_loss = nn.CrossEntropyLoss(weight=gate_weights); taxon_loss = nn.CrossEntropyLoss(weight=taxon_weights)
    optimizer_kwargs = {"lr": args.learning_rate, "weight_decay": args.weight_decay}
    if device.type == "cuda": optimizer_kwargs["fused"] = True
    optimizer = torch.optim.AdamW(model.parameters(), **optimizer_kwargs)
    epochs = args.dry_run_epochs if args.dry_run else args.epochs
    args.output_dir.mkdir(parents=True, exist_ok=True); tag = workflow_tag(args)
    best_path = args.output_dir / f"{tag}_best_model.pt"; history = []; best = -1.0; stale = 0
    for epoch in range(1, epochs + 1):
        datasets["train"].set_epoch(epoch)
        train = run_epoch(model, loaders["train"], device, optimizer, gate_loss, taxon_loss, args)
        val = run_epoch(model, loaders["val"], device, None, gate_loss, taxon_loss, args)
        gt, tt = tune_thresholds(val, args.gate_threshold, args.taxon_threshold)
        score = f1_score(val["label"], predictions(val, gt, tt), average="macro", zero_division=0)
        history.append({"epoch": epoch, "train_loss": train["loss"], "val_loss": val["loss"],
            "train_samples_per_second": train["samples_per_second"], "val_macro_f1": score,
            "gate_threshold": gt, "taxon_threshold": tt})
        print(f"epoch {epoch:03d} train_loss={train['loss']:.4f} val_loss={val['loss']:.4f} "
              f"val_macro_f1={score:.4f} train_rate={train['samples_per_second']:.1f}/s thresholds=({gt:.2f},{tt:.2f})")
        if score > best:
            best = score; stale = 0
            torch.save({"state_dict": model.state_dict(), "metadata": metadata, "args": vars(args), "gate_threshold": gt, "taxon_threshold": tt}, best_path)
        else:
            stale += 1
            if stale >= args.patience: break
    checkpoint = torch.load(best_path, map_location=device, weights_only=False); model.load_state_dict(checkpoint["state_dict"])
    test = run_epoch(model, loaders["test"], device, None, gate_loss, taxon_loss, args)
    pred = predictions(test, checkpoint["gate_threshold"], checkpoint["taxon_threshold"])
    report = classification_report(test["label"], pred, labels=range(len(FINAL_CLASSES)), target_names=FINAL_CLASSES, zero_division=0, output_dict=True)
    print(classification_report(test["label"], pred, labels=range(len(FINAL_CLASSES)), target_names=FINAL_CLASSES, zero_division=0))
    pd.DataFrame(history).to_csv(args.output_dir / f"{tag}_history.csv", index=False)
    pd.DataFrame({"tree_uid": test["uid"], "true_label": [FINAL_CLASSES[i] for i in test["label"]],
        "predicted_label": [FINAL_CLASSES[i] for i in pred], "named_probability": test["gate_prob"],
        "conditional_taxon_probability": test["taxon_prob"].max(1)}).to_csv(args.output_dir / f"{tag}_test_predictions.csv", index=False)
    (args.output_dir / f"{tag}_test_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved model: {best_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
