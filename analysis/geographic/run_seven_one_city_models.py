#!/usr/bin/env python3
"""Train seven complete-coverage one-city k11 models and evaluate transfer."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
TRAINER = ROOT / "model" / "stage2" / "train_fixed_partition_classifier.py"
EVALUATOR = ROOT / "model" / "stage2" / "evaluate_checkpoint_ensemble.py"
COLLAPSE_MAP = "1,1,2,2,3,3,4,5,3,6,2"
SOURCE_CITIES = (
    "denver",
    "portland",
    "ranchocucamonga",
    "sanfrancisco",
    "sanjose",
    "seattle",
    "stlouis",
)
TARGET_CITIES = (
    "atlanta", "auroraco", "baltimore", "buffalo", "capecoral", "columbus",
    "denver", "gardengrove", "huntingtonbeach", "lasvegas", "louisville",
    "madison", "newyork", "pittsburgh", "portland", "ranchocucamonga",
    "sandiego", "sanfrancisco", "sanjose", "seattle", "siouxfalls",
    "stlouis", "washingtondc",
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--city", action="append", choices=SOURCE_CITIES, help="Run selected source city; repeat as needed.")
    p.add_argument("--complete-shard-root", type=Path, default=Path(r"H:\TreeCenteredModelInputs\tree_centered_complete_sharded100k_prism_daily3_clean"))
    p.add_argument("--species-partition", type=Path, default=ROOT / "config" / "final_k11_partition.npz")
    p.add_argument("--split-manifest", type=Path, default=ROOT / "artifacts" / "reviewer_stage2_loader_manifest.csv.gz")
    p.add_argument("--output-root", type=Path, default=Path(r"H:\k11_one_city_seven_source"))
    p.add_argument("--epochs", type=int, default=120)
    p.add_argument("--early-stopping-patience", type=int, default=12)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--num-workers", type=int, default=3)
    p.add_argument("--prefetch-factor", type=int, default=3)
    p.add_argument("--max-shard-cache", type=int, default=10)
    p.add_argument("--device", default="cuda")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--force", action="store_true")
    return p.parse_args()


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


def training_command(a: argparse.Namespace, city: str, run_name: str, model_root: Path) -> list[str]:
    command = [
        sys.executable, str(TRAINER),
        "--complete-shard-root", str(a.complete_shard_root),
        "--complete-shard-pattern", "*_part*_tree_centered_complete_inputs.npz",
        "--output-dir", str(model_root), "--run-name", run_name,
        "--grouping-scheme", "species_partition", "--species-partition", str(a.species_partition),
        "--split-manifest", str(a.split_manifest), "--city-token", city,
        "--seed", "13", "--reassign-capped-eval-to-train", "--exclude-qa-from-model",
        "--no-exclude-qa-warning-from-model", "--use-qa-sample-weights",
        "--low-center-vegetation-weight", "0.25", "--low-vegetated-height-with-lidar-weight", "0.50",
        "--insufficient-lidar-coverage-weight", "0.75", "--use-distance-sample-weights",
        "--distance-weight-sigma-m", "4.0", "--min-distance-sample-weight", "0.25",
        "--epochs", str(a.epochs), "--early-stopping-patience", str(a.early_stopping_patience),
        "--early-stopping-min-delta", "0.0001", "--checkpoint-selection-score", "blended",
        "--blended-score-macro-weight", "0.50", "--blended-score-accuracy-weight", "0.30",
        "--blended-score-weighted-weight", "0.20", "--save-top-k-checkpoints", "5",
        "--batch-size", str(a.batch_size), "--num-workers", str(a.num_workers), "--eval-num-workers", "0",
        "--prefetch-factor", str(a.prefetch_factor), "--persistent-workers", "--shard-local-train-batches",
        "--shard-window-size", "8", "--max-shard-cache", str(a.max_shard_cache),
        "--cache-val-in-memory", "--no-cache-test-in-memory", "--image-center-crop-pixels", "28",
        "--image-encoder-normalization", "group", "--naip-encoder-normalization", "group",
        "--chm-encoder-normalization", "group", "--interaction-encoder-normalization", "group",
        "--independent-chm-full-extent", "--no-interaction-naip-full-extent",
        "--interaction-naip-crop-pixels", "30", "--interaction-fusion-mode", "learned_common_grid",
        "--interaction-naip-patch-pixels", "5", "--interaction-chm-patch-pixels", "3",
        "--image-augmentation", "--use-sentinel-phenology", "--use-raw-sentinel",
        "--use-satellite-embedding", "--no-use-prism-normals", "--use-naip-chm-interaction-branch",
        "--exclude-modality", "tree_centered_structure", "--phenology-stat-samples", "20000",
        "--raw-sentinel-stat-samples", "20000", "--satellite-embedding-stat-samples", "20000",
        "--hidden-dim", "128", "--dropout", "0.20", "--learning-rate", "0.0005",
        "--weight-decay", "0.0001", "--grad-clip", "2.0", "--class-weighting", "sqrt_inverse",
        "--class-weight-max", "3.0", "--device", a.device, "--export-validation-predictions",
        "--validation-only-pilot",
    ]
    if a.force:
        command.append("--force")
    return command


def evaluation_command(
    a: argparse.Namespace, city: str, model_run: Path, result_root: Path, within_city: bool,
) -> list[str]:
    suffix = "within_city" if within_city else "cross_city_eligible"
    command = [
        sys.executable, str(EVALUATOR), "--model-run-dir", str(model_run), "--top-k", "5",
        "--complete-shard-root", str(a.complete_shard_root),
        "--complete-shard-pattern", "*_part*_tree_centered_complete_inputs.npz",
        "--output-dir", str(result_root), "--run-name", f"{city}_{suffix}_test_top5",
        "--grouping-scheme", "species_partition", "--species-partition", str(a.species_partition),
        "--split-manifest", str(a.split_manifest), "--split", "test", "--batch-size", str(a.batch_size),
        "--num-workers", str(a.num_workers), "--max-shard-cache", str(a.max_shard_cache),
        "--collapse-map", COLLAPSE_MAP, "--device", a.device, "--export-predictions",
    ]
    targets = (city,) if within_city else tuple(value for value in TARGET_CITIES if value != city)
    for target in targets:
        command.extend(["--city-token", target])
    if a.force:
        command.append("--force")
    return command


def main() -> int:
    a = parse_args()
    required = (TRAINER, EVALUATOR, a.complete_shard_root, a.species_partition, a.split_manifest)
    for path in required:
        if not path.exists():
            raise FileNotFoundError(path)
    cities = a.city or list(SOURCE_CITIES)
    model_root = a.output_root / "models"
    result_root = a.output_root / "results"
    for index, city in enumerate(cities, start=1):
        print(f"[{index}/{len(cities)}] one-city source: {city}", flush=True)
        run_name = f"{city}_only_pruned_k11_v1"
        model_run = model_root / run_name
        jobs = (
            (training_command(a, city, run_name, model_root), model_run / "run_config.json", result_root / "logs" / f"{city}_training.log"),
            (evaluation_command(a, city, model_run, result_root, True), result_root / f"{city}_within_city_test_top5" / "ensemble_summary.json", result_root / "logs" / f"{city}_within_city_inference.log"),
            (evaluation_command(a, city, model_run, result_root, False), result_root / f"{city}_cross_city_eligible_test_top5" / "ensemble_summary.json", result_root / "logs" / f"{city}_cross_city_inference.log"),
        )
        for command, completion_marker, log in jobs:
            if a.dry_run:
                print(subprocess.list2cmdline(command), flush=True)
            elif completion_marker.exists() and not a.force:
                print(f"Completed output exists; skipping: {completion_marker}", flush=True)
            else:
                stream(command, log)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
