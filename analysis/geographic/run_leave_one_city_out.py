#!/usr/bin/env python3
"""Run leave-one-city-out training for well-supported focal cities.

For each focal city, the inherited stage-two split is used as follows:

* train: inherited training records from every other city;
* validation: inherited validation records from every other city;
* test: inherited test records from the focal city only.

The finalized pruned k11 architecture is trained from scratch. Its five best
validation checkpoints are ensembled, then probabilities are summed into the
fixed collapsed_k6 reporting groups. Runs are sequential to preserve RAM and GPU
throughput. Final city and pooled summaries are written after all requested
cities finish.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from geographic_utils import (
    collapse_map_string,
    eligible_city_support,
    load_predictions,
    load_geographic_config,
    load_qa_flags,
    metric_set,
    scientific_training_arguments,
    screen_qa,
    update_status_manifest,
    validate_split_manifest,
)


ROOT = Path(__file__).resolve().parents[2]
TRAINER = ROOT / "model" / "stage2" / "train_fixed_partition_classifier.py"
EVALUATOR = ROOT / "model" / "stage2" / "evaluate_checkpoint_ensemble.py"
DEFAULT_RESULT_OUTPUT = ROOT / "outputs" / "reviewer_revision" / "k11_loco_pruned_v1"
DEFAULT_PARTITION = ROOT / "config" / "final_k11_partition.npz"
DEFAULT_GROUPING = ROOT / "config" / "final_grouping.json"
DEFAULT_GEOGRAPHIC_CONFIG = ROOT / "analysis" / "geographic" / "geographic_config.json"


def parse_args() -> argparse.Namespace:
    """Parse portable data paths and runtime-only performance settings."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--complete-shard-root", type=Path, required=True)
    p.add_argument("--prepared-mmap-root", type=Path, required=True)
    p.add_argument("--grouping-config", type=Path, default=DEFAULT_GROUPING)
    p.add_argument("--geographic-config", type=Path, default=DEFAULT_GEOGRAPHIC_CONFIG)
    p.add_argument("--split-manifest", type=Path, required=True)
    p.add_argument("--reference-test-predictions", type=Path, required=True)
    p.add_argument("--qa-metadata", type=Path, required=True)
    p.add_argument("--model-output-dir", type=Path, required=True)
    p.add_argument("--result-output-dir", type=Path, default=DEFAULT_RESULT_OUTPUT)
    p.add_argument("--minimum-test-records", type=int, default=1000)
    p.add_argument("--minimum-group-records", type=int, default=10)
    p.add_argument("--city", action="append", default=None, help="Run only this eligible city; repeat as needed.")
    p.add_argument("--start-at", default=None, help="Skip eligible cities alphabetically before this city.")
    p.add_argument("--epochs", type=int, default=120)
    p.add_argument("--early-stopping-patience", type=int, default=12)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--num-workers", type=int, default=3)
    p.add_argument("--prefetch-factor", type=int, default=3)
    p.add_argument("--shard-window-size", type=int, default=8)
    p.add_argument("--max-shard-cache", type=int, default=10)
    p.add_argument("--device", default="cuda")
    p.add_argument("--list-cities", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--overwrite", action="store_true", help="Restart or replace an existing city run.")
    return p.parse_args()


def run_streamed(command: list[str], log_path: Path, cwd: Path) -> None:
    """Run a subprocess while mirroring its output to the console and a log."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(subprocess.list2cmdline(command), flush=True)
    with log_path.open("a", encoding="utf-8", buffering=1) as log:
        process = subprocess.Popen(
            command, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace", bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
        code = process.wait()
    if code:
        raise subprocess.CalledProcessError(code, command)


def aggregate(
    result_root: Path,
    cities: list[str],
    qa_path: Path,
    minimum_group_records: int,
    collapse_map: str,
) -> None:
    """Aggregate every completed eligible-city prediction with final reporting rules."""
    city_rows, group_rows = [], []
    k11_labels, k11_predictions, collapsed_labels, collapsed_predictions = [], [], [], []
    completed = []
    for city in cities:
        run_dir = result_root / f"loco_{city}_test_top5_ensemble"
        pred_path = run_dir / "ensemble_predictions.npz"
        if not pred_path.exists():
            continue
        predictions = load_predictions(pred_path, "loco")
        screened = screen_qa(
            predictions.merge(load_qa_flags(qa_path), on="record_key", validate="one_to_one")
        )
        y11 = screened["k11_label_loco"].to_numpy()
        p11 = screened["k11_prediction_loco"].to_numpy()
        collapsed_label = screened["collapsed_k6_label_loco"].to_numpy()
        collapsed_prediction = screened["collapsed_k6_prediction_loco"].to_numpy()
        m11, _, _ = metric_set(y11, p11, 11, minimum_group_records)
        collapsed_metrics, collapsed_per_group, _ = metric_set(
            collapsed_label, collapsed_prediction, 6, minimum_group_records
        )
        city_rows.append({
            "city": city,
            **{f"k11_{key}": value for key, value in m11.items()},
            **{f"collapsed_k6_{key}": value for key, value in collapsed_metrics.items()},
        })
        group_rows.extend({"city": city, **row} for row in collapsed_per_group)
        k11_labels.append(y11); k11_predictions.append(p11)
        collapsed_labels.append(collapsed_label); collapsed_predictions.append(collapsed_prediction)
        completed.append(city)
    if not completed:
        return
    city_frame = pd.DataFrame(city_rows).sort_values("city")
    city_frame.to_csv(result_root / "loco_city_summary.csv", index=False)
    pd.DataFrame(group_rows).to_csv(result_root / "loco_city_collapsed_k6_per_group_metrics.csv", index=False)
    y11, p11 = np.concatenate(k11_labels), np.concatenate(k11_predictions)
    collapsed_label = np.concatenate(collapsed_labels)
    collapsed_prediction = np.concatenate(collapsed_predictions)
    m11, pg11, cm11 = metric_set(y11, p11, 11, minimum_group_records)
    collapsed_metrics, collapsed_per_group, collapsed_confusion = metric_set(
        collapsed_label, collapsed_prediction, 6, minimum_group_records
    )
    pooled = {
        "completed_cities": completed,
        "qa_screen": "excluded qa_flag_low_vegetated_height_with_lidar records",
        "minimum_group_records_for_macro_f1": minimum_group_records,
        "k11": m11,
        "collapsed_k6": collapsed_metrics,
        "collapse_map": collapse_map,
    }
    (result_root / "loco_pooled_summary.json").write_text(json.dumps(pooled, indent=2), encoding="utf-8")
    pd.DataFrame(pg11).to_csv(result_root / "loco_pooled_k11_per_group_metrics.csv", index=False)
    pd.DataFrame(collapsed_per_group).to_csv(result_root / "loco_pooled_collapsed_k6_per_group_metrics.csv", index=False)
    pd.DataFrame(cm11).to_csv(result_root / "loco_pooled_k11_confusion_counts.csv", index=False)
    pd.DataFrame(collapsed_confusion).to_csv(result_root / "loco_pooled_collapsed_k6_confusion_counts.csv", index=False)


def training_command(a: argparse.Namespace, city: str, run_name: str) -> list[str]:
    """Construct a LOCO training command from frozen scientific settings."""
    configuration = load_geographic_config(a.geographic_config)
    cmd = [
        sys.executable, str(TRAINER),
        "--complete-shard-root", str(a.complete_shard_root),
        "--complete-shard-pattern", "*_part*_tree_centered_complete_inputs.npz",
        "--prepared-mmap-root", str(a.prepared_mmap_root),
        "--output-dir", str(a.model_output_dir), "--run-name", run_name,
        "--split-manifest", str(a.split_manifest), "--leave-one-city-out-test-city", city,
        "--epochs", str(a.epochs), "--early-stopping-patience", str(a.early_stopping_patience),
        "--batch-size", str(a.batch_size), "--num-workers", str(a.num_workers), "--eval-num-workers", "0",
        "--prefetch-factor", str(a.prefetch_factor), "--persistent-workers", "--shard-local-train-batches",
        "--shard-window-size", str(a.shard_window_size), "--max-shard-cache", str(a.max_shard_cache),
        "--cache-val-in-memory", "--no-cache-test-in-memory",
        *scientific_training_arguments(configuration), "--device", a.device,
        "--export-validation-predictions", "--validation-only-pilot",
    ]
    if a.overwrite:
        cmd.append("--overwrite")
    return cmd


def evaluation_command(a: argparse.Namespace, city: str, model_run: Path) -> list[str]:
    """Construct independent focal-city ensemble inference."""
    cmd = [
        sys.executable, str(EVALUATOR), "--model-run-dir", str(model_run), "--top-k", "5",
        "--complete-shard-root", str(a.complete_shard_root),
        "--complete-shard-pattern", "*_part*_tree_centered_complete_inputs.npz",
        "--prepared-mmap-root", str(a.prepared_mmap_root), "--output-dir", str(a.result_output_dir),
        "--run-name", f"loco_{city}_test_top5_ensemble",
        "--split-manifest", str(a.split_manifest),
        "--city-token", city, "--split", "test",
        "--batch-size", str(a.batch_size), "--num-workers", "2", "--max-shard-cache", str(a.max_shard_cache),
        "--device", a.device, "--export-predictions",
    ]
    if a.overwrite:
        cmd.append("--overwrite")
    return cmd


def main() -> int:
    """Run, resume, or inspect the configured LOCO campaign."""
    a = parse_args()
    required = [
        TRAINER, EVALUATOR, a.complete_shard_root, a.prepared_mmap_root,
        a.grouping_config, a.geographic_config, a.split_manifest,
        a.reference_test_predictions, a.qa_metadata,
    ]
    for path in required:
        if not path.exists():
            raise FileNotFoundError(path)
    validate_split_manifest(a.split_manifest, a.geographic_config)
    eligible = eligible_city_support(
        a.reference_test_predictions, a.qa_metadata, a.minimum_test_records
    )
    all_eligible_cities = eligible.city.tolist()
    if a.city:
        requested = {value.strip().lower() for value in a.city}
        unknown = sorted(requested.difference(set(eligible.city)))
        if unknown:
            raise SystemExit(f"Requested cities are not eligible at threshold {a.minimum_test_records}: {unknown}")
        eligible = eligible.loc[eligible.city.isin(requested)].copy()
    if a.start_at:
        eligible = eligible.loc[eligible.city.ge(a.start_at.strip().lower())].copy()
    print(eligible.to_string(index=False), flush=True)
    print(f"Eligible focal cities: {len(eligible)}", flush=True)
    if a.list_cities:
        return 0
    if not a.dry_run:
        a.model_output_dir.mkdir(parents=True, exist_ok=True)
        a.result_output_dir.mkdir(parents=True, exist_ok=True)
    cities = eligible.city.tolist()
    for index, city in enumerate(cities, 1):
        run_name = f"loco_{city}_pruned_k11_v1"
        model_run = a.model_output_dir / run_name
        result_run = a.result_output_dir / f"loco_{city}_test_top5_ensemble"
        print(f"\n[{index}/{len(cities)}] focal city: {city}", flush=True)
        if (result_run / "ensemble_summary.json").exists() and not a.overwrite:
            print(f"Completed result exists; skipping {city}: {result_run}", flush=True)
            update_status_manifest(a.result_output_dir / "run_status.csv", city, "ensemble", "complete")
            continue
        train_cmd = training_command(a, city, run_name)
        eval_cmd = evaluation_command(a, city, model_run)
        if a.dry_run:
            print(subprocess.list2cmdline(train_cmd), flush=True)
            print(subprocess.list2cmdline(eval_cmd), flush=True)
            continue
        if not (model_run / "run_config.json").exists():
            if model_run.exists() and any(model_run.iterdir()) and not a.overwrite:
                raise SystemExit(
                    f"Incomplete model directory exists for {city}: {model_run}. "
                    "Inspect it, then rerun with --overwrite to restart that city."
                )
            update_status_manifest(a.result_output_dir / "run_status.csv", city, "training", "running")
            try:
                run_streamed(train_cmd, a.result_output_dir / "logs" / f"{city}_training.log", ROOT)
            except Exception:
                update_status_manifest(a.result_output_dir / "run_status.csv", city, "training", "failed")
                raise
            update_status_manifest(a.result_output_dir / "run_status.csv", city, "training", "complete")
        else:
            print(f"Training already complete; proceeding to ensemble: {model_run}", flush=True)
            update_status_manifest(a.result_output_dir / "run_status.csv", city, "training", "complete")
        update_status_manifest(a.result_output_dir / "run_status.csv", city, "ensemble", "running")
        try:
            run_streamed(eval_cmd, a.result_output_dir / "logs" / f"{city}_ensemble.log", ROOT)
        except Exception:
            update_status_manifest(a.result_output_dir / "run_status.csv", city, "ensemble", "failed")
            raise
        update_status_manifest(a.result_output_dir / "run_status.csv", city, "ensemble", "complete")
        aggregate(
            a.result_output_dir, all_eligible_cities, a.qa_metadata,
            a.minimum_group_records, collapse_map_string(a.grouping_config),
        )
    if not a.dry_run:
        aggregate(
            a.result_output_dir, all_eligible_cities, a.qa_metadata,
            a.minimum_group_records, collapse_map_string(a.grouping_config),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
