#!/usr/bin/env python3
"""Evaluate frozen single-modality models on the independent test split.

This is an inference-only final reporting script: it neither trains models nor
selects checkpoints. It also exports the canonical ``collapsed_k6`` results.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from workflow import canonical_collapse_map, inference_complete, validate_split_manifest

ROOT = Path(__file__).resolve().parents[2]


MODELS = {
    "naip_chm_interaction": "k11_single_naip_chm_interaction_v2",
    "raw_sentinel": "k11_single_raw_sentinel_v2",
    "satellite_embedding": "k11_single_satellite_embedding_v2",
    "sentinel_phenology": "k11_single_sentinel_phenology_v2",
    "tree_centered_chm": "k11_single_tree_centered_chm_v2",
    "tree_centered_naip": "k11_single_tree_centered_naip_v1",
    "tree_centered_structure": "k11_single_tree_centered_structure_v2",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument(
        "--complete-shard-root",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--prepared-mmap-root",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--split-manifest",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/manuscript_analysis/single_modality_test_inference"),
    )
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--num-workers", type=int, default=3)
    parser.add_argument("--overwrite", action="store_true", help="Replace existing inference products.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    validate_split_manifest(args.split_manifest)
    evaluator = ROOT / "model" / "stage2" / "evaluate_checkpoint_ensemble.py"
    args.output_root.mkdir(parents=True, exist_ok=True)
    for index, (modality, directory_name) in enumerate(MODELS.items(), start=1):
        model_dir = args.model_root / directory_name
        checkpoint = model_dir / "best_model.pt"
        run_dir = args.output_root / modality
        if inference_complete(run_dir) and not args.overwrite:
            print(f"[{index}/{len(MODELS)}] {modality}: complete; skipping", flush=True)
            continue
        command = [
            sys.executable,
            str(evaluator),
            "--model-run-dir", str(model_dir),
            "--checkpoint", str(checkpoint),
            "--complete-shard-root", str(args.complete_shard_root),
            "--prepared-mmap-root", str(args.prepared_mmap_root),
            "--output-dir", str(args.output_root),
            "--run-name", modality,
            "--split-manifest", str(args.split_manifest),
            "--split", "test",
            "--batch-size", str(args.batch_size),
            "--num-workers", str(args.num_workers),
            "--device", "cuda",
            "--export-predictions",
        ]
        if args.overwrite:
            command.append("--overwrite")
        print(f"[{index}/{len(MODELS)}] {modality}: starting", flush=True)
        subprocess.run(command, check=True)
        print(f"[{index}/{len(MODELS)}] {modality}: complete", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
