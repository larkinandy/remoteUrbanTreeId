#!/usr/bin/env python3
"""Run frozen independent-test inference for all completed k=11 single-modality models."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


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
    parser.add_argument("--model-root", type=Path, default=Path(r"H:\k11_single_modality_results"))
    parser.add_argument(
        "--complete-shard-root",
        type=Path,
        default=Path(r"H:\TreeCenteredModelInputs\tree_centered_complete_sharded100k_prism_daily3_clean"),
    )
    parser.add_argument(
        "--prepared-mmap-root",
        type=Path,
        default=Path(r"H:\TreeCenteredModelInputs\tree_centered_prepared_float32_v1"),
    )
    parser.add_argument(
        "--partition",
        type=Path,
        default=Path("config/final_k11_partition.npz"),
    )
    parser.add_argument(
        "--split-manifest",
        type=Path,
        default=Path("outputs/reviewer_revision/k11_single_modality_worker_bundle/artifacts/reviewer_stage2_loader_manifest.csv.gz"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/reviewer_revision/k11_single_modality_test_inference"),
    )
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--num-workers", type=int, default=3)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    evaluator = Path(__file__).resolve().parents[2] / "model" / "stage2" / "evaluate_checkpoint_ensemble.py"
    args.output_root.mkdir(parents=True, exist_ok=True)
    for index, (modality, directory_name) in enumerate(MODELS.items(), start=1):
        model_dir = args.model_root / directory_name
        checkpoint = model_dir / "best_model.pt"
        run_dir = args.output_root / modality
        summary = run_dir / "ensemble_summary.json"
        if summary.exists() and not args.force:
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
            "--grouping-scheme", "species_partition",
            "--species-partition", str(args.partition),
            "--split-manifest", str(args.split_manifest),
            "--split", "test",
            "--collapse-map", "1,1,2,2,3,3,4,5,3,6,2",
            "--batch-size", str(args.batch_size),
            "--num-workers", str(args.num_workers),
            "--device", "cuda",
        ]
        if args.force:
            command.append("--force")
        print(f"[{index}/{len(MODELS)}] {modality}: starting", flush=True)
        subprocess.run(command, check=True)
        print(f"[{index}/{len(MODELS)}] {modality}: complete", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
