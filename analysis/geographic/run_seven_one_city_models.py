#!/usr/bin/env python3
"""Train the configured one-city k11 models and evaluate national transfer."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from geographic_utils import (
    collapse_map_string,
    eligible_city_support,
    load_geographic_config,
    scientific_training_arguments,
    update_status_manifest,
    validate_split_manifest,
)


ROOT = Path(__file__).resolve().parents[2]
TRAINER = ROOT / "model" / "stage2" / "train_fixed_partition_classifier.py"
EVALUATOR = ROOT / "model" / "stage2" / "evaluate_checkpoint_ensemble.py"
DEFAULT_CONFIG = ROOT / "analysis" / "geographic" / "geographic_config.json"
DEFAULT_GROUPING = ROOT / "config" / "final_grouping.json"


def parse_args() -> argparse.Namespace:
    """Parse portable inputs and runtime-only throughput settings."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--city", action="append", help="Run selected configured source city; repeat as needed.")
    p.add_argument("--complete-shard-root", type=Path, required=True)
    p.add_argument("--grouping-config", type=Path, default=DEFAULT_GROUPING)
    p.add_argument("--geographic-config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--split-manifest", type=Path, required=True)
    p.add_argument("--reference-test-predictions", type=Path, required=True)
    p.add_argument("--qa-metadata", type=Path, required=True)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--minimum-test-records", type=int, default=1000)
    p.add_argument("--epochs", type=int, default=120)
    p.add_argument("--early-stopping-patience", type=int, default=12)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--num-workers", type=int, default=3)
    p.add_argument("--prefetch-factor", type=int, default=3)
    p.add_argument("--max-shard-cache", type=int, default=10)
    p.add_argument("--device", default="cuda")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def stream(command: list[str], log_path: Path) -> None:
    """Run a subprocess while mirroring output to a persistent log."""
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
    """Construct one source-city training command from frozen scientific settings."""
    configuration = load_geographic_config(a.geographic_config)
    command = [
        sys.executable, str(TRAINER),
        "--complete-shard-root", str(a.complete_shard_root),
        "--complete-shard-pattern", "*_part*_tree_centered_complete_inputs.npz",
        "--output-dir", str(model_root), "--run-name", run_name,
        "--split-manifest", str(a.split_manifest), "--city-token", city,
        "--epochs", str(a.epochs), "--early-stopping-patience", str(a.early_stopping_patience),
        "--batch-size", str(a.batch_size), "--num-workers", str(a.num_workers), "--eval-num-workers", "0",
        "--prefetch-factor", str(a.prefetch_factor), "--persistent-workers", "--shard-local-train-batches",
        "--shard-window-size", "8", "--max-shard-cache", str(a.max_shard_cache),
        "--cache-val-in-memory", "--no-cache-test-in-memory",
        *scientific_training_arguments(configuration),
        "--device", a.device, "--export-validation-predictions",
        "--validation-only-pilot",
    ]
    if a.overwrite:
        command.append("--overwrite")
    return command


def evaluation_command(
    a: argparse.Namespace,
    city: str,
    model_run: Path,
    result_root: Path,
    target_cities: tuple[str, ...],
    within_city: bool,
) -> list[str]:
    """Construct within- or cross-city ensemble inference for one source model."""
    suffix = "within_city" if within_city else "cross_city_eligible"
    command = [
        sys.executable, str(EVALUATOR), "--model-run-dir", str(model_run), "--top-k", "5",
        "--complete-shard-root", str(a.complete_shard_root),
        "--complete-shard-pattern", "*_part*_tree_centered_complete_inputs.npz",
        "--output-dir", str(result_root), "--run-name", f"{city}_{suffix}_test_top5",
        "--split-manifest", str(a.split_manifest), "--split", "test", "--batch-size", str(a.batch_size),
        "--num-workers", str(a.num_workers), "--max-shard-cache", str(a.max_shard_cache),
        "--device", a.device, "--export-predictions",
    ]
    targets = (city,) if within_city else tuple(value for value in target_cities if value != city)
    for target in targets:
        command.extend(["--city-token", target])
    if a.overwrite:
        command.append("--overwrite")
    return command


def main() -> int:
    """Run or resume the configured one-city model campaign."""
    a = parse_args()
    required = (
        TRAINER, EVALUATOR, a.complete_shard_root,
        a.grouping_config, a.geographic_config, a.split_manifest,
        a.reference_test_predictions, a.qa_metadata,
    )
    for path in required:
        if not path.exists():
            raise FileNotFoundError(path)
    validate_split_manifest(a.split_manifest, a.geographic_config)
    configuration = json.loads(a.geographic_config.read_text(encoding="utf-8"))
    configured_sources = tuple(str(value) for value in configuration["source_cities"])
    cities = a.city or list(configured_sources)
    unknown = sorted(set(cities).difference(configured_sources))
    if unknown:
        raise SystemExit(f"Cities are not configured one-city sources: {unknown}")
    target_cities = tuple(
        eligible_city_support(
            a.reference_test_predictions, a.qa_metadata, a.minimum_test_records
        )["city"].tolist()
    )
    model_root = a.output_root / "models"
    result_root = a.output_root / "results"
    for index, city in enumerate(cities, start=1):
        print(f"[{index}/{len(cities)}] one-city source: {city}", flush=True)
        run_name = f"{city}_only_pruned_k11_v1"
        model_run = model_root / run_name
        jobs = (
            ("training", training_command(a, city, run_name, model_root), model_run / "run_config.json", result_root / "logs" / f"{city}_training.log"),
            ("within_city_inference", evaluation_command(a, city, model_run, result_root, target_cities, True), result_root / f"{city}_within_city_test_top5" / "ensemble_summary.json", result_root / "logs" / f"{city}_within_city_inference.log"),
            ("cross_city_inference", evaluation_command(a, city, model_run, result_root, target_cities, False), result_root / f"{city}_cross_city_eligible_test_top5" / "ensemble_summary.json", result_root / "logs" / f"{city}_cross_city_inference.log"),
        )
        for stage, command, completion_marker, log in jobs:
            if a.dry_run:
                print(subprocess.list2cmdline(command), flush=True)
            elif completion_marker.exists() and not a.overwrite:
                print(f"Completed output exists; skipping: {completion_marker}", flush=True)
                update_status_manifest(result_root / "run_status.csv", city, stage, "complete")
            else:
                update_status_manifest(result_root / "run_status.csv", city, stage, "running")
                try:
                    stream(command, log)
                except Exception:
                    update_status_manifest(result_root / "run_status.csv", city, stage, "failed")
                    raise
                update_status_manifest(result_root / "run_status.csv", city, stage, "complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
