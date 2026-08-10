#!/usr/bin/env python3
"""Evaluate a probability ensemble of clean k=6 classifier checkpoints.

The intended workflow is:

1. Resume the semi-final classifier with --save-top-k-checkpoints.
2. Run this script on that fine-tune run directory.

The script recreates the clean shard split from the checkpoint/run args,
loads the selected checkpoints at once, averages their softmax probabilities,
and writes ensemble metrics for val/test smoke checks or final comparison.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import evaluate_clean_k6_pruning_ablations as prune
import train_clean_tree_id_centered_k6_classifier as k6
import train_clean_tree_id_centered_taxon_discriminator as clean_disc


CLEAN_ROOT = Path(r"H:\TreeCenteredModelInputs")
DEFAULT_OUTPUT_DIR = CLEAN_ROOT / "ensemble_diagnostics_clean"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--model-run-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", action="append", type=Path, default=None)
    parser.add_argument("--top-k", type=int, default=5, help="Use this many checkpoints from top_checkpoints.csv when --checkpoint is omitted.")
    parser.add_argument("--complete-shard-root", type=Path, default=CLEAN_ROOT / "tree_centered_complete_sharded100k_clean")
    parser.add_argument("--complete-shard-pattern", default="*_part*_tree_centered_complete_inputs.npz")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--grouping-scheme", choices=("species_k6", "genus_k6"), default="species_k6")
    parser.add_argument("--species-k6-partition", type=Path, default=k6.DEFAULT_SPECIES_K6)
    parser.add_argument("--genus-k6-partition", type=Path, default=k6.DEFAULT_GENUS_K6)
    parser.add_argument("--city-token", action="append", default=None)
    parser.add_argument("--exclude-city-token", action="append", default=[])
    parser.add_argument("--split", choices=("val", "test", "both"), default="test")
    parser.add_argument("--max-eval-samples", type=int, default=0, help="Optional smoke-test sample cap after split recreation.")
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--max-shard-cache", type=int, default=10)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--export-predictions", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def checkpoint_paths(args: argparse.Namespace) -> list[Path]:
    if args.checkpoint:
        return [Path(path) for path in args.checkpoint]
    manifest_path = args.model_run_dir / "top_checkpoints" / "top_checkpoints.csv"
    if manifest_path.exists():
        frame = pd.read_csv(manifest_path)
        if "rank" in frame.columns:
            frame = frame.sort_values("rank")
        paths = [Path(value) for value in frame["checkpoint"].head(max(int(args.top_k), 1)).tolist()]
        if paths:
            return paths
    best = args.model_run_dir / "best_model.pt"
    if best.exists():
        print(f"No top-k checkpoint manifest found; using {best}", flush=True)
        return [best]
    raise FileNotFoundError(f"No checkpoints found under {args.model_run_dir}")


def tensor_or_none(value: Any) -> np.ndarray | None:
    return prune.as_numpy(value)


def model_from_checkpoint(checkpoint: dict[str, Any], run_args: argparse.Namespace, class_count: int, device: Any):
    model = k6.CleanK6Classifier(
        class_count=class_count,
        structure_dim=int(tensor_or_none(checkpoint["structure_mean"]).shape[0]),
        hidden_dim=int(getattr(run_args, "hidden_dim", 128)),
        dropout=float(getattr(run_args, "dropout", 0.2)),
        phenology_dim=0 if checkpoint.get("phenology_mean") is None else int(tensor_or_none(checkpoint["phenology_mean"]).shape[0]),
        raw_sentinel_dim=0 if checkpoint.get("raw_sentinel_mean") is None else int(tensor_or_none(checkpoint["raw_sentinel_mean"]).shape[0]),
        satellite_embedding_dim=0
        if checkpoint.get("satellite_embedding_mean") is None
        else int(tensor_or_none(checkpoint["satellite_embedding_mean"]).shape[0]),
        prism_normals_dim=0 if checkpoint.get("prism_mean") is None else int(tensor_or_none(checkpoint["prism_mean"]).shape[0]),
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
    model.eval()
    return model


def per_group_metrics(y_true: np.ndarray, y_pred: np.ndarray, group_names: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for cls, group in enumerate(group_names):
        true_mask = y_true == cls
        pred_mask = y_pred == cls
        tp = int(np.sum(true_mask & pred_mask))
        fp = int(np.sum(~true_mask & pred_mask))
        fn = int(np.sum(true_mask & ~pred_mask))
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1.0e-12)
        rows.append(
            {
                "group": group,
                "support": int(true_mask.sum()),
                "predicted": int(pred_mask.sum()),
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
            }
        )
    return pd.DataFrame(rows)


@clean_disc.train_base.torch.no_grad()
def evaluate_ensemble(models: list[Any], loader: Any, device: Any, class_count: int) -> dict[str, Any]:
    torch = clean_disc.train_base.torch
    y_true: list[np.ndarray] = []
    y_pred: list[np.ndarray] = []
    prob_rows: list[np.ndarray] = []
    gate_rows: list[np.ndarray] = []
    cities: list[str] = []
    tree_ids: list[np.ndarray] = []
    crop_indices: list[np.ndarray] = []

    for batch_index, batch in enumerate(loader, start=1):
        city_token = batch.pop("city_token", None)
        batch = {key: value.to(device, non_blocking=True) if hasattr(value, "to") else value for key, value in batch.items()}
        prob_sum = None
        gate_sum = None
        for model in models:
            logits, gates = model(batch)
            probs = torch.softmax(logits, dim=1)
            prob_sum = probs if prob_sum is None else prob_sum + probs
            gate_sum = gates if gate_sum is None else gate_sum + gates
        avg_prob = prob_sum / max(len(models), 1)
        avg_gate = gate_sum / max(len(models), 1)
        pred = avg_prob.argmax(dim=1)
        y_true.append(batch["label"].detach().cpu().numpy())
        y_pred.append(pred.detach().cpu().numpy())
        prob_rows.append(avg_prob.detach().cpu().numpy())
        gate_rows.append(avg_gate.detach().cpu().numpy())
        if city_token is not None:
            cities.extend(list(city_token))
        tree_ids.append(batch["tree_centered_index"].detach().cpu().numpy())
        crop_indices.append(batch["crop_index"].detach().cpu().numpy())
        if batch_index % 100 == 0:
            print(f"  ensemble inference: batches={batch_index:,}; rows={batch_index * int(batch['label'].shape[0]):,}", flush=True)

    true = np.concatenate(y_true) if y_true else np.empty(0, dtype=np.int64)
    pred = np.concatenate(y_pred) if y_pred else np.empty(0, dtype=np.int64)
    metrics = clean_disc.f1_metrics(true, pred, class_count)
    metrics["probabilities"] = np.concatenate(prob_rows) if prob_rows else np.empty((0, class_count), dtype=np.float32)
    metrics["gate_weights"] = np.concatenate(gate_rows) if gate_rows else np.empty((0, 0), dtype=np.float32)
    metrics["labels"] = true
    metrics["predictions"] = pred
    metrics["city_token"] = np.asarray(cities, dtype="U64")
    metrics["tree_id"] = np.concatenate(tree_ids) if tree_ids else np.empty(0, dtype=np.int64)
    metrics["crop_index"] = np.concatenate(crop_indices) if crop_indices else np.empty(0, dtype=np.int64)
    return metrics


def main() -> int:
    args = parse_args()
    if clean_disc.train_base.TORCH_IMPORT_ERROR is not None:
        raise SystemExit(f"PyTorch is required. Original error: {clean_disc.train_base.TORCH_IMPORT_ERROR}")
    torch = clean_disc.train_base.torch
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")

    paths = checkpoint_paths(args)
    checkpoints = [torch.load(path, map_location=device) for path in paths]
    run_args = prune.runtime_args(args, checkpoints[0])
    run_args.image_augmentation = False
    run_args.tree_image_channel_dropout = 0.0
    run_args.tree_image_branch_dropout = 0.0
    run_args.satellite_embedding_dropout = 0.0
    run_args.satellite_embedding_value_dropout = 0.0
    run_args.prism_normals_dropout = 0.0
    run_args.prism_normals_value_dropout = 0.0

    run_name = args.run_name or f"{args.model_run_dir.name}_{args.split}_top{len(paths)}_ensemble"
    out_dir = args.output_dir / run_name
    if out_dir.exists() and any(out_dir.iterdir()) and not args.force:
        raise SystemExit(f"Output directory exists and is not empty: {out_dir}; pass --force.")
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest, shard_paths, group_names = prune.build_manifest(args, run_args)
    splits = ["val", "test"] if args.split == "both" else [args.split]
    eval_manifest = manifest.loc[manifest["split"].isin(splits)].reset_index(drop=True)
    if args.max_eval_samples > 0 and len(eval_manifest) > args.max_eval_samples:
        eval_manifest = eval_manifest.sample(n=args.max_eval_samples, random_state=int(run_args.seed)).sort_index().reset_index(drop=True)
        print(f"Sampled eval rows for smoke test: {len(eval_manifest):,}", flush=True)
    print(
        f"Evaluating checkpoint ensemble: checkpoints={len(paths)}; split={args.split}; rows={len(eval_manifest):,}",
        flush=True,
    )

    first = checkpoints[0]
    store = clean_disc.CleanShardStore(shard_paths, max_cached_shards=args.max_shard_cache)
    ds = clean_disc.CleanTreeDataset(
        eval_manifest,
        store,
        tensor_or_none(first["structure_mean"]),
        tensor_or_none(first["structure_std"]),
        tensor_or_none(first.get("phenology_mean")),
        tensor_or_none(first.get("phenology_std")),
        tensor_or_none(first.get("raw_sentinel_mean")),
        tensor_or_none(first.get("raw_sentinel_std")),
        tensor_or_none(first.get("satellite_embedding_mean")),
        tensor_or_none(first.get("satellite_embedding_std")),
        tensor_or_none(first.get("prism_mean")),
        tensor_or_none(first.get("prism_std")),
        run_args,
    )
    loader = prune.make_loader(ds, args)
    models = [model_from_checkpoint(checkpoint, run_args, len(group_names), device) for checkpoint in checkpoints]
    metrics = evaluate_ensemble(models, loader, device, len(group_names))
    per_group = per_group_metrics(metrics["labels"], metrics["predictions"], group_names)
    per_group.to_csv(out_dir / "ensemble_per_group_metrics.csv", index=False)
    modality_names = list(models[0].modality_names)
    gate_weights = metrics["gate_weights"]
    gate_summary = pd.DataFrame(
        {
            "modality": modality_names,
            "mean_gate_weight": gate_weights.mean(axis=0),
            "standard_deviation": gate_weights.std(axis=0, ddof=1),
            "median_gate_weight": np.median(gate_weights, axis=0),
            "p05_gate_weight": np.quantile(gate_weights, 0.05, axis=0),
            "p95_gate_weight": np.quantile(gate_weights, 0.95, axis=0),
        }
    ).sort_values("mean_gate_weight", ascending=False)
    gate_summary.to_csv(out_dir / "ensemble_gate_weight_summary.csv", index=False)

    summary = {
        "model_run_dir": str(args.model_run_dir),
        "checkpoint_count": len(paths),
        "checkpoints": [str(path) for path in paths],
        "split": args.split,
        "rows": int(len(metrics["labels"])),
        "accuracy": float(metrics["accuracy"]),
        "macro_f1": float(metrics["macro_f1"]),
        "weighted_f1": float(metrics["weighted_f1"]),
        "group_names": group_names,
        "args": json.loads(json.dumps(clean_disc.safe_json_args(args), default=str)),
    }
    (out_dir / "ensemble_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    pd.DataFrame([summary | {"group_names": ",".join(group_names), "checkpoints": "|".join(str(p) for p in paths)}]).to_csv(
        out_dir / "ensemble_summary.csv",
        index=False,
    )
    if args.export_predictions:
        np.savez_compressed(
            out_dir / "ensemble_predictions.npz",
            probabilities=metrics["probabilities"],
            labels=metrics["labels"],
            predictions=metrics["predictions"],
            city_token=metrics["city_token"],
            tree_id=metrics["tree_id"],
            crop_index=metrics["crop_index"],
            group_names=np.asarray(group_names, dtype="<U64"),
            gate_weights=metrics["gate_weights"],
            modality_names=np.asarray(modality_names, dtype="<U64"),
        )
    print(
        f"Finished ensemble: acc={metrics['accuracy']:.4f}; macro_f1={metrics['macro_f1']:.4f}; "
        f"weighted_f1={metrics['weighted_f1']:.4f}; output={out_dir}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
