#!/usr/bin/env python3
"""Evaluate no-retrain pruning ablations for clean k=6 classifiers.

This script loads a trained clean k=6 classifier checkpoint, recreates its
held-out split, and re-runs inference while zeroing selected modalities or
feature groups after standardization. Zero after standardization means
"replace with the training mean" for vector/sequence features.

Use this as a cheap pruning screen before short fine-tunes or full retrains.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import train_clean_tree_id_centered_k6_classifier as k6
import train_clean_tree_id_centered_taxon_discriminator as clean_disc


CLEAN_ROOT = Path(r"H:\TreeCenteredModelInputs")
DEFAULT_RUN_DIR = CLEAN_ROOT / "collapsed_group_models_clean" / "clean_species_binomial_k6_classifier_interaction_center28"
DEFAULT_OUTPUT_DIR = CLEAN_ROOT / "pruning_diagnostics_clean"


DEFAULT_ABLATIONS = [
    "none",
    "tree_centered_structure",
    "sentinel_phenology",
    "raw_sentinel",
    "satellite_embedding",
    "prism_normals",
    "tree_image",
    "tree_chm",
    "phenology:NDVI",
    "phenology:EVI2",
    "phenology:NDII",
    "phenology:NDRE",
    "phenology:brightness",
    "raw:B2|B3|B4",
    "raw:B5|B6|B7|B8A",
    "raw:B8",
    "raw:B11|B12",
    "raw:NDVI|GNDVI|CIg|CIre|MTCI|MCARI|NDVIre|REPI",
    "raw:NDII|MSAVI|LAI",
    "raw:sentinel_observed|sentinel_interpolated|data_quality|source_image|interpolation",
    "prism:ppt",
    "prism:tmax|tmean|tmin|tdmean",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--model-run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--complete-shard-root", type=Path, default=CLEAN_ROOT / "tree_centered_complete_sharded100k_clean")
    parser.add_argument("--complete-shard-pattern", default="*_part*_tree_centered_complete_inputs.npz")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--run-name", default="center28_no_retrain_ablation_probe")
    parser.add_argument("--grouping-scheme", choices=("species_k6", "genus_k6"), default="species_k6")
    parser.add_argument("--species-k6-partition", type=Path, default=k6.DEFAULT_SPECIES_K6)
    parser.add_argument("--genus-k6-partition", type=Path, default=k6.DEFAULT_GENUS_K6)
    parser.add_argument("--city-token", action="append", default=None)
    parser.add_argument("--exclude-city-token", action="append", default=[])
    parser.add_argument("--split", choices=("val", "test", "both"), default="test")
    parser.add_argument("--max-eval-samples", type=int, default=0, help="Optional smoke-test sample cap after split recreation.")
    parser.add_argument("--ablation", action="append", default=None, help="Ablation spec. Omit for default pruning screen.")
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--max-shard-cache", type=int, default=10)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def as_numpy(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy().astype(np.float32)
    return np.asarray(value, dtype=np.float32)


def runtime_args(args: argparse.Namespace, checkpoint: dict[str, Any]) -> argparse.Namespace:
    saved = dict(checkpoint.get("args") or load_json(args.model_run_dir / "run_config.json").get("args", {}))
    saved["complete_shard_root"] = args.complete_shard_root
    saved["complete_shard_pattern"] = args.complete_shard_pattern
    saved["city_token"] = args.city_token
    saved["exclude_city_token"] = args.exclude_city_token
    saved["max_records_per_city"] = 0
    saved["max_shard_cache"] = args.max_shard_cache
    saved["batch_size"] = args.batch_size
    saved["num_workers"] = args.num_workers
    saved["eval_num_workers"] = args.num_workers
    saved["image_augmentation"] = False
    saved.setdefault("tree_image_channel_dropout", 0.0)
    saved.setdefault("tree_image_branch_dropout", 0.0)
    saved.setdefault("satellite_embedding_dropout", 0.0)
    saved.setdefault("satellite_embedding_value_dropout", 0.0)
    saved.setdefault("prism_normals_dropout", 0.0)
    saved.setdefault("prism_normals_value_dropout", 0.0)
    return argparse.Namespace(**saved)


def load_partition_for_args(args: argparse.Namespace) -> tuple[dict[str, int], list[str]]:
    path = args.genus_k6_partition if args.grouping_scheme == "genus_k6" else args.species_k6_partition
    return k6.load_partition(path)


def build_manifest(args: argparse.Namespace, run_args: argparse.Namespace) -> tuple[pd.DataFrame, dict[str, Path], list[str]]:
    k6.ensure_shard_pattern(run_args)
    label_to_group, group_names = load_partition_for_args(args)
    manifest, paths = clean_disc.discover_shards(run_args)
    manifest["fine_label"] = clean_disc.scientific_label_from_name(manifest["scientific_name"], "scientific_binomial_or_genus")
    before = len(manifest)
    manifest = manifest.loc[manifest["fine_label"].isin(label_to_group)].copy()
    print(f"Dropped records not represented in {args.grouping_scheme} partition: {before:,}->{len(manifest):,}", flush=True)
    manifest["label_index"] = manifest["fine_label"].map(label_to_group).astype(np.int64)
    manifest["sample_weight"] = clean_disc.compute_sample_weights(manifest, run_args)
    manifest = clean_disc.split_manifest(manifest, run_args)
    manifest = clean_disc.cap_eval_splits(manifest, run_args)
    return manifest.reset_index(drop=True), paths, group_names


def first_shard_columns(paths: dict[str, Path]) -> dict[str, list[str]]:
    first = next(iter(paths.values()))
    data = np.load(first, allow_pickle=True)
    return {
        "phenology": [str(x) for x in data["sentinel_phenology_columns"]] if "sentinel_phenology_columns" in data.files else [],
        "raw": [str(x) for x in data["sentinel_feature_columns"]] if "sentinel_feature_columns" in data.files else [],
        "prism": [str(x) for x in data["prism_normals_feature_names"]] if "prism_normals_feature_names" in data.files else [],
        "structure": [str(x) for x in data["tree_centered_naip_chm_structure_columns"]] if "tree_centered_naip_chm_structure_columns" in data.files else [],
    }


def regex_indices(columns: list[str], pattern: str) -> list[int]:
    rx = re.compile(pattern, flags=re.IGNORECASE)
    return [i for i, name in enumerate(columns) if rx.search(name)]


def build_index_specs(ablations: list[str], columns: dict[str, list[str]]) -> dict[str, dict[str, Any]]:
    specs: dict[str, dict[str, Any]] = {}
    for spec in ablations:
        spec = spec.strip()
        if not spec:
            continue
        item: dict[str, Any] = {"spec": spec}
        if ":" in spec:
            kind, pattern = spec.split(":", 1)
            kind = kind.strip().lower()
            pattern = pattern.strip()
            if kind in {"phenology", "raw", "prism", "structure"}:
                idx = regex_indices(columns[kind], pattern)
                if not idx:
                    print(f"WARNING: ablation {spec!r} matched no {kind} columns.", flush=True)
                item["kind"] = kind
                item["indices"] = idx
                item["matched_columns"] = [columns[kind][i] for i in idx]
            else:
                raise SystemExit(f"Unknown ablation kind in {spec!r}")
        else:
            item["kind"] = spec
            item["indices"] = None
            item["matched_columns"] = []
        specs[spec] = item
    return specs


def clone_batch(batch: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in batch.items():
        if hasattr(value, "clone"):
            out[key] = value.clone()
        else:
            out[key] = value
    return out


def apply_ablation(batch: dict[str, Any], spec: dict[str, Any]) -> dict[str, Any]:
    kind = spec["kind"]
    if kind == "none":
        return batch
    batch = clone_batch(batch)
    if kind == "tree_image":
        batch["tree_image"].zero_()
    elif kind == "tree_chm":
        batch["tree_chm_image"].zero_()
    elif kind == "tree_centered_structure":
        batch["tree_centered_structure"].zero_()
    elif kind == "sentinel_phenology" and "sentinel_phenology" in batch:
        batch["sentinel_phenology"].zero_()
    elif kind == "raw_sentinel" and "sentinel_sequence" in batch:
        batch["sentinel_sequence"].zero_()
    elif kind == "satellite_embedding" and "satellite_embedding" in batch:
        batch["satellite_embedding"].zero_()
    elif kind == "prism_normals" and "prism_normals" in batch:
        batch["prism_normals"].zero_()
    elif kind == "phenology" and "sentinel_phenology" in batch:
        idx = spec["indices"]
        if idx:
            batch["sentinel_phenology"][:, idx] = 0
    elif kind == "raw" and "sentinel_sequence" in batch:
        idx = spec["indices"]
        if idx:
            batch["sentinel_sequence"][:, :, idx] = 0
    elif kind == "prism" and "prism_normals" in batch:
        idx = spec["indices"]
        if idx:
            batch["prism_normals"][:, idx] = 0
    elif kind == "structure" and "tree_centered_structure" in batch:
        idx = spec["indices"]
        if idx:
            batch["tree_centered_structure"][:, idx] = 0
    else:
        raise SystemExit(f"Unknown or unavailable ablation kind: {kind}")
    return batch


def make_loader(ds: Any, args: argparse.Namespace) -> Any:
    torch = clean_disc.train_base.torch
    kwargs: dict[str, Any] = {
        "batch_size": int(args.batch_size),
        "shuffle": False,
        "num_workers": int(args.num_workers),
        "pin_memory": True,
    }
    if int(args.num_workers) > 0:
        kwargs["prefetch_factor"] = 3
        kwargs["persistent_workers"] = True
    return torch.utils.data.DataLoader(ds, **kwargs)


def evaluate_ablation(model: Any, loader: Any, device: Any, class_count: int, group_names: list[str], spec: dict[str, Any]) -> tuple[dict[str, Any], pd.DataFrame]:
    torch = clean_disc.train_base.torch
    model.eval()
    true_rows: list[np.ndarray] = []
    pred_rows: list[np.ndarray] = []
    with torch.no_grad():
        for batch_index, batch in enumerate(loader, start=1):
            batch.pop("city_token", None)
            batch = {key: value.to(device, non_blocking=True) if hasattr(value, "to") else value for key, value in batch.items()}
            batch = apply_ablation(batch, spec)
            logits, _ = model(batch)
            pred = logits.argmax(dim=1)
            true_rows.append(batch["label"].detach().cpu().numpy())
            pred_rows.append(pred.detach().cpu().numpy())
            if batch_index % 100 == 0:
                print(f"  {spec['spec']}: batches={batch_index:,}", flush=True)
    y = np.concatenate(true_rows) if true_rows else np.empty(0, dtype=np.int64)
    yp = np.concatenate(pred_rows) if pred_rows else np.empty(0, dtype=np.int64)
    metrics = clean_disc.f1_metrics(y, yp, class_count)
    per_class = []
    for cls, group in enumerate(group_names):
        true_mask = y == cls
        pred_mask = yp == cls
        tp = int(np.sum(true_mask & pred_mask))
        fp = int(np.sum(~true_mask & pred_mask))
        fn = int(np.sum(true_mask & ~pred_mask))
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1.0e-12)
        per_class.append(
            {
                "ablation": spec["spec"],
                "group": group,
                "support": int(true_mask.sum()),
                "predicted": int(pred_mask.sum()),
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )
    metrics["support"] = int(len(y))
    return metrics, pd.DataFrame(per_class)


def main() -> int:
    args = parse_args()
    if clean_disc.train_base.TORCH_IMPORT_ERROR is not None:
        raise SystemExit(f"PyTorch is required. Original error: {clean_disc.train_base.TORCH_IMPORT_ERROR}")
    torch = clean_disc.train_base.torch
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    checkpoint_path = args.checkpoint or (args.model_run_dir / "best_model.pt")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    run_args = runtime_args(args, checkpoint)

    out_dir = args.output_dir / args.run_name
    if out_dir.exists() and any(out_dir.iterdir()) and not args.force:
        raise SystemExit(f"Output directory exists and is not empty: {out_dir}; pass --force.")
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest, paths, group_names = build_manifest(args, run_args)
    splits = ["val", "test"] if args.split == "both" else [args.split]
    eval_manifest = manifest.loc[manifest["split"].isin(splits)].reset_index(drop=True)
    if args.max_eval_samples > 0 and len(eval_manifest) > args.max_eval_samples:
        eval_manifest = eval_manifest.sample(n=args.max_eval_samples, random_state=int(run_args.seed)).sort_index().reset_index(drop=True)
        print(f"Sampled eval rows for smoke test: {len(eval_manifest):,}", flush=True)
    print(f"Evaluation rows: {len(eval_manifest):,}; split={args.split}", flush=True)

    columns = first_shard_columns(paths)
    ablations = args.ablation if args.ablation else DEFAULT_ABLATIONS
    specs = build_index_specs(ablations, columns)

    store = clean_disc.CleanShardStore(paths, max_cached_shards=args.max_shard_cache)
    ds = clean_disc.CleanTreeDataset(
        eval_manifest,
        store,
        as_numpy(checkpoint["structure_mean"]),
        as_numpy(checkpoint["structure_std"]),
        as_numpy(checkpoint.get("phenology_mean")),
        as_numpy(checkpoint.get("phenology_std")),
        as_numpy(checkpoint.get("raw_sentinel_mean")),
        as_numpy(checkpoint.get("raw_sentinel_std")),
        as_numpy(checkpoint.get("satellite_embedding_mean")),
        as_numpy(checkpoint.get("satellite_embedding_std")),
        as_numpy(checkpoint.get("prism_mean")),
        as_numpy(checkpoint.get("prism_std")),
        run_args,
    )
    loader = make_loader(ds, args)
    model = k6.CleanK6Classifier(
        class_count=len(group_names),
        structure_dim=int(as_numpy(checkpoint["structure_mean"]).shape[0]),
        hidden_dim=int(getattr(run_args, "hidden_dim", 128)),
        dropout=float(getattr(run_args, "dropout", 0.2)),
        phenology_dim=0 if checkpoint.get("phenology_mean") is None else int(as_numpy(checkpoint["phenology_mean"]).shape[0]),
        raw_sentinel_dim=0 if checkpoint.get("raw_sentinel_mean") is None else int(as_numpy(checkpoint["raw_sentinel_mean"]).shape[0]),
        satellite_embedding_dim=0 if checkpoint.get("satellite_embedding_mean") is None else int(as_numpy(checkpoint["satellite_embedding_mean"]).shape[0]),
        prism_normals_dim=0 if checkpoint.get("prism_mean") is None else int(as_numpy(checkpoint["prism_mean"]).shape[0]),
        tree_image_branch_dropout=0.0,
        use_naip_chm_interaction_branch=bool(getattr(run_args, "use_naip_chm_interaction_branch", False)),
        independent_naip_crop_pixels=int(getattr(run_args, "image_center_crop_pixels", 0)),
        interaction_naip_crop_pixels=int(getattr(run_args, "interaction_naip_crop_pixels", 0)),
        interaction_fusion_mode=str(getattr(run_args, "interaction_fusion_mode", "bilinear_upsample")),
        interaction_naip_patch_pixels=int(getattr(run_args, "interaction_naip_patch_pixels", 5)),
        interaction_chm_patch_pixels=int(getattr(run_args, "interaction_chm_patch_pixels", 3)),
        phenology_zero_indices=list(getattr(run_args, "phenology_zero_indices", [])),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    rows: list[dict[str, Any]] = []
    per_class_frames: list[pd.DataFrame] = []
    for name, spec in specs.items():
        print(f"Evaluating ablation: {name}", flush=True)
        metrics, per_class = evaluate_ablation(model, loader, device, len(group_names), group_names, spec)
        row = {
            "ablation": name,
            "kind": spec["kind"],
            "matched_feature_count": 0 if spec["matched_columns"] is None else len(spec["matched_columns"]),
            **metrics,
        }
        rows.append(row)
        per_class_frames.append(per_class)
        print(
            f"  {name}: acc={metrics['accuracy']:.4f}; macro_f1={metrics['macro_f1']:.4f}; "
            f"weighted_f1={metrics['weighted_f1']:.4f}",
            flush=True,
        )
    summary = pd.DataFrame(rows)
    if "none" in summary["ablation"].values:
        base = summary.loc[summary["ablation"].eq("none")].iloc[0]
        for metric in ("accuracy", "macro_f1", "weighted_f1"):
            summary[f"{metric}_delta_vs_none"] = summary[metric] - float(base[metric])
    summary = summary.sort_values("accuracy_delta_vs_none" if "accuracy_delta_vs_none" in summary else "accuracy", ascending=False)
    summary.to_csv(out_dir / "ablation_summary.csv", index=False)
    if per_class_frames:
        pd.concat(per_class_frames, ignore_index=True).to_csv(out_dir / "ablation_per_group_metrics.csv", index=False)
    match_rows = []
    for spec in specs.values():
        for col in spec.get("matched_columns") or []:
            match_rows.append({"ablation": spec["spec"], "feature": col})
    pd.DataFrame(match_rows).to_csv(out_dir / "ablation_matched_features.csv", index=False)
    print("\nAblation summary:", flush=True)
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"), flush=True)
    print(f"\nWrote outputs: {out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
