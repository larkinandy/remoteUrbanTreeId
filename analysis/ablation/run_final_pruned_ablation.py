#!/usr/bin/env python3
"""Retrain final-pruned k11 leave-one-modality-out models and evaluate test ensembles."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from workflow import canonical_collapse_map, validate_split_manifest

ROOT = Path(__file__).resolve().parents[2]
TRAINER = ROOT / "model" / "stage2" / "train_fixed_partition_classifier.py"
EVALUATOR = ROOT / "model" / "stage2" / "evaluate_checkpoint_ensemble.py"
FINAL_MODALITIES = (
    "tree_centered_naip",
    "tree_centered_chm",
    "naip_chm_interaction",
    "sentinel_phenology",
    "raw_sentinel",
    "satellite_embedding",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--modality", action="append", choices=FINAL_MODALITIES, help="Run only selected branches; repeat as needed.")
    parser.add_argument("--complete-shard-root", type=Path, required=True)
    parser.add_argument("--prepared-mmap-root", type=Path, required=True)
    parser.add_argument("--species-partition", type=Path, default=ROOT / "config" / "final_k11_partition.npz")
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--model-output-dir", type=Path, required=True)
    parser.add_argument("--result-output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--early-stopping-patience", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--max-shard-cache", type=int, default=3)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def stream(command: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(subprocess.list2cmdline(command), flush=True)
    with log_path.open("a", encoding="utf-8", buffering=1) as log:
        process = subprocess.Popen(
            command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace", bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
        code = process.wait()
    if code:
        raise subprocess.CalledProcessError(code, command)


def train_command(args: argparse.Namespace, modality: str, run_name: str) -> list[str]:
    command = [
        sys.executable, str(TRAINER),
        "--complete-shard-root", str(args.complete_shard_root),
        "--complete-shard-pattern", "*_part*_tree_centered_complete_inputs.npz",
        "--prepared-mmap-root", str(args.prepared_mmap_root),
        "--output-dir", str(args.model_output_dir), "--run-name", run_name,
        "--grouping-scheme", "species_partition", "--species-partition", str(args.species_partition),
        "--split-manifest", str(args.split_manifest), "--seed", "13", "--reassign-capped-eval-to-train",
        "--exclude-qa-from-model", "--no-exclude-qa-warning-from-model", "--use-qa-sample-weights",
        "--low-center-vegetation-weight", "0.25", "--low-vegetated-height-with-lidar-weight", "0.50",
        "--insufficient-lidar-coverage-weight", "0.75", "--use-distance-sample-weights",
        "--distance-weight-sigma-m", "4.0", "--min-distance-sample-weight", "0.25",
        "--epochs", str(args.epochs), "--early-stopping-patience", str(args.early_stopping_patience),
        "--early-stopping-min-delta", "0.0001", "--checkpoint-selection-score", "blended",
        "--blended-score-macro-weight", "0.50", "--blended-score-accuracy-weight", "0.30",
        "--blended-score-weighted-weight", "0.20", "--save-top-k-checkpoints", "5",
        "--batch-size", str(args.batch_size), "--num-workers", str(args.num_workers), "--eval-num-workers", "0",
        "--prefetch-factor", str(args.prefetch_factor), "--persistent-workers", "--shard-local-train-batches",
        "--shard-window-size", "8", "--max-shard-cache", str(args.max_shard_cache),
        "--no-cache-val-in-memory", "--no-cache-test-in-memory", "--image-center-crop-pixels", "28",
        "--image-encoder-normalization", "group", "--naip-encoder-normalization", "group",
        "--chm-encoder-normalization", "group", "--interaction-encoder-normalization", "group",
        "--independent-chm-full-extent", "--no-interaction-naip-full-extent",
        "--interaction-naip-crop-pixels", "30", "--interaction-fusion-mode", "learned_common_grid",
        "--interaction-naip-patch-pixels", "5", "--interaction-chm-patch-pixels", "3",
        "--image-augmentation", "--use-sentinel-phenology", "--use-raw-sentinel",
        "--use-satellite-embedding", "--no-use-prism-normals", "--use-naip-chm-interaction-branch",
        "--exclude-modality", "tree_centered_structure", "--exclude-modality", modality,
        "--phenology-stat-samples", "20000", "--raw-sentinel-stat-samples", "20000",
        "--satellite-embedding-stat-samples", "20000", "--hidden-dim", "128", "--dropout", "0.20",
        "--learning-rate", "0.0005", "--weight-decay", "0.0001", "--grad-clip", "2.0",
        "--class-weighting", "sqrt_inverse", "--class-weight-max", "3.0", "--device", args.device,
        "--export-validation-predictions", "--validation-only-pilot",
    ]
    if args.force:
        command.append("--force")
    return command


def evaluate_command(args: argparse.Namespace, model_run: Path, modality: str) -> list[str]:
    command = [
        sys.executable, str(EVALUATOR), "--model-run-dir", str(model_run), "--top-k", "5",
        "--complete-shard-root", str(args.complete_shard_root),
        "--complete-shard-pattern", "*_part*_tree_centered_complete_inputs.npz",
        "--prepared-mmap-root", str(args.prepared_mmap_root), "--output-dir", str(args.result_output_dir),
        "--run-name", f"without_{modality}_test_top5", "--grouping-scheme", "species_partition",
        "--species-partition", str(args.species_partition), "--split-manifest", str(args.split_manifest),
        "--split", "test", "--batch-size", str(args.batch_size), "--num-workers", str(args.num_workers),
        "--max-shard-cache", str(args.max_shard_cache), "--collapse-map", canonical_collapse_map(),
        "--device", args.device, "--export-predictions",
    ]
    if args.force:
        command.append("--force")
    return command


def main() -> int:
    args = parse_args()
    for path in (TRAINER, EVALUATOR, args.complete_shard_root, args.prepared_mmap_root, args.species_partition, args.split_manifest):
        if not path.exists():
            raise FileNotFoundError(path)
    validate_split_manifest(args.split_manifest)
    targets = args.modality or list(FINAL_MODALITIES)
    if not args.dry_run:
        args.model_output_dir.mkdir(parents=True, exist_ok=True)
        args.result_output_dir.mkdir(parents=True, exist_ok=True)
    for index, modality in enumerate(targets, start=1):
        run_name = f"final_pruned_without_{modality}_v1"
        model_run = args.model_output_dir / run_name
        result_run = args.result_output_dir / f"without_{modality}_test_top5"
        print(f"[{index}/{len(targets)}] leave out {modality}", flush=True)
        train = train_command(args, modality, run_name)
        evaluate = evaluate_command(args, model_run, modality)
        if args.dry_run:
            print(subprocess.list2cmdline(train), flush=True)
            print(subprocess.list2cmdline(evaluate), flush=True)
            continue
        if not (model_run / "run_config.json").exists():
            stream(train, args.result_output_dir / "logs" / f"without_{modality}_training.log")
        else:
            print(f"Completed training exists; skipping fit: {model_run}", flush=True)
        if not (result_run / "ensemble_summary.json").exists():
            stream(evaluate, args.result_output_dir / "logs" / f"without_{modality}_test_inference.log")
        else:
            print(f"Completed test ensemble exists; skipping inference: {result_run}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
