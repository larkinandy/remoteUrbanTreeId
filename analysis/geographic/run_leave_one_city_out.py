#!/usr/bin/env python3
"""Run leave-one-city-out training for well-supported focal cities.

For each focal city, the inherited stage-two split is used as follows:

* train: inherited training records from every other city;
* validation: inherited validation records from every other city;
* test: inherited test records from the focal city only.

The finalized pruned k=11 architecture is trained from scratch. Its five best
validation checkpoints are ensembled, then probabilities are summed into the
fixed six reporting groups. Runs are sequential to preserve RAM and GPU
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


ROOT = Path(__file__).resolve().parents[2]
TRAINER = ROOT / "model" / "stage2" / "train_fixed_partition_classifier.py"
EVALUATOR = ROOT / "model" / "stage2" / "evaluate_checkpoint_ensemble.py"
DEFAULT_SHARDS = Path(r"H:\TreeCenteredModelInputs\tree_centered_complete_sharded100k_prism_daily3_clean")
DEFAULT_MMAP = Path(r"H:\TreeCenteredModelInputs\tree_centered_prepared_float32_v1")
DEFAULT_MODEL_OUTPUT = Path(
    r"H:\TreeCenteredModelInputs\collapsed_group_models_clean\reviewer_revision\k11_loco_pruned_v1"
)
DEFAULT_RESULT_OUTPUT = ROOT / "outputs" / "reviewer_revision" / "k11_loco_pruned_v1"
DEFAULT_PARTITION = ROOT / "config" / "final_k11_partition.npz"
DEFAULT_SPLITS = (
    ROOT / "outputs" / "reviewer_revision" / "stage2_partitions_k11_groupnorm_v1"
    / "reviewer_stage2_loader_manifest.csv.gz"
)
DEFAULT_REFERENCE_PREDICTIONS = (
    ROOT / "outputs" / "reviewer_revision" / "final_pruned_k11_test_top5_ensemble"
    / "ensemble_predictions.npz"
)
COLLAPSE_MAP = "1,1,2,2,3,3,4,5,3,6,2"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--complete-shard-root", type=Path, default=DEFAULT_SHARDS)
    p.add_argument("--prepared-mmap-root", type=Path, default=DEFAULT_MMAP)
    p.add_argument("--species-partition", type=Path, default=DEFAULT_PARTITION)
    p.add_argument("--split-manifest", type=Path, default=DEFAULT_SPLITS)
    p.add_argument("--reference-test-predictions", type=Path, default=DEFAULT_REFERENCE_PREDICTIONS)
    p.add_argument("--model-output-dir", type=Path, default=DEFAULT_MODEL_OUTPUT)
    p.add_argument("--result-output-dir", type=Path, default=DEFAULT_RESULT_OUTPUT)
    p.add_argument("--minimum-test-records", type=int, default=1000)
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
    p.add_argument("--force", action="store_true", help="Allow trainer/evaluator reuse of nonempty output directories.")
    return p.parse_args()


def eligible_cities(path: Path, minimum: int) -> pd.DataFrame:
    with np.load(path, allow_pickle=True) as pred:
        cities = pd.Series(pred["city_token"].astype(str), name="city")
    counts = cities.value_counts().rename_axis("city").reset_index(name="test_records")
    return counts.loc[counts.test_records.ge(minimum)].sort_values("city").reset_index(drop=True)


def run_streamed(command: list[str], log_path: Path, cwd: Path) -> None:
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


def metric_set(labels: np.ndarray, predictions: np.ndarray, class_count: int) -> tuple[dict, list[dict], np.ndarray]:
    cm = np.bincount(labels * class_count + predictions, minlength=class_count**2).reshape(class_count, class_count)
    support = cm.sum(axis=1)
    predicted = cm.sum(axis=0)
    tp = np.diag(cm)
    precision = np.divide(tp, predicted, out=np.zeros(class_count, dtype=float), where=predicted > 0)
    recall = np.divide(tp, support, out=np.zeros(class_count, dtype=float), where=support > 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros(class_count), where=(precision + recall) > 0)
    represented = support > 0
    overall = {
        "records": int(support.sum()),
        "represented_groups": int(represented.sum()),
        "accuracy": float(tp.sum() / support.sum()),
        "macro_f1": float(f1[represented].mean()),
        "weighted_f1": float(np.average(f1[represented], weights=support[represented])),
    }
    per_group = [
        {
            "group": index + 1, "support": int(support[index]), "predicted": int(predicted[index]),
            "precision": float(precision[index]), "recall": float(recall[index]), "f1": float(f1[index]),
        }
        for index in range(class_count)
    ]
    return overall, per_group, cm


def aggregate(result_root: Path, cities: list[str]) -> None:
    city_rows, group_rows = [], []
    native_labels, native_predictions, collapsed_labels, collapsed_predictions = [], [], [], []
    completed = []
    for city in cities:
        run_dir = result_root / f"loco_{city}_test_top5_ensemble"
        pred_path = run_dir / "ensemble_predictions.npz"
        if not pred_path.exists():
            continue
        with np.load(pred_path, allow_pickle=True) as pred:
            y11, p11 = pred["labels"].astype(np.int64), pred["predictions"].astype(np.int64)
            y6, p6 = pred["collapsed_labels"].astype(np.int64), pred["collapsed_predictions"].astype(np.int64)
        m11, _, _ = metric_set(y11, p11, 11)
        m6, pg6, _ = metric_set(y6, p6, 6)
        city_rows.append({"city": city, **{f"k11_{k}": v for k, v in m11.items()}, **{f"k6_{k}": v for k, v in m6.items()}})
        group_rows.extend({"city": city, **row} for row in pg6)
        native_labels.append(y11); native_predictions.append(p11)
        collapsed_labels.append(y6); collapsed_predictions.append(p6)
        completed.append(city)
    if not completed:
        return
    city_frame = pd.DataFrame(city_rows).sort_values("city")
    city_frame.to_csv(result_root / "loco_city_summary.csv", index=False)
    pd.DataFrame(group_rows).to_csv(result_root / "loco_city_k6_per_group_metrics.csv", index=False)
    y11, p11 = np.concatenate(native_labels), np.concatenate(native_predictions)
    y6, p6 = np.concatenate(collapsed_labels), np.concatenate(collapsed_predictions)
    m11, pg11, cm11 = metric_set(y11, p11, 11)
    m6, pg6, cm6 = metric_set(y6, p6, 6)
    pooled = {"completed_cities": completed, "k11": m11, "collapsed_k6": m6, "collapse_map": COLLAPSE_MAP}
    (result_root / "loco_pooled_summary.json").write_text(json.dumps(pooled, indent=2), encoding="utf-8")
    pd.DataFrame(pg11).to_csv(result_root / "loco_pooled_k11_per_group_metrics.csv", index=False)
    pd.DataFrame(pg6).to_csv(result_root / "loco_pooled_k6_per_group_metrics.csv", index=False)
    pd.DataFrame(cm11).to_csv(result_root / "loco_pooled_k11_confusion_counts.csv", index=False)
    pd.DataFrame(cm6).to_csv(result_root / "loco_pooled_k6_confusion_counts.csv", index=False)


def training_command(a: argparse.Namespace, city: str, run_name: str) -> list[str]:
    cmd = [
        sys.executable, str(TRAINER),
        "--complete-shard-root", str(a.complete_shard_root),
        "--complete-shard-pattern", "*_part*_tree_centered_complete_inputs.npz",
        "--prepared-mmap-root", str(a.prepared_mmap_root),
        "--output-dir", str(a.model_output_dir), "--run-name", run_name,
        "--grouping-scheme", "species_partition", "--species-partition", str(a.species_partition),
        "--split-manifest", str(a.split_manifest), "--leave-one-city-out-test-city", city,
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
        "--shard-window-size", str(a.shard_window_size), "--max-shard-cache", str(a.max_shard_cache),
        "--cache-val-in-memory", "--no-cache-test-in-memory", "--image-center-crop-pixels", "28",
        "--image-encoder-normalization", "group", "--naip-encoder-normalization", "group",
        "--chm-encoder-normalization", "group", "--interaction-encoder-normalization", "group",
        "--independent-chm-full-extent", "--no-interaction-naip-full-extent",
        "--interaction-naip-crop-pixels", "30", "--interaction-fusion-mode", "learned_common_grid",
        "--interaction-naip-patch-pixels", "5", "--interaction-chm-patch-pixels", "3",
        "--image-augmentation", "--use-sentinel-phenology", "--use-raw-sentinel",
        "--use-satellite-embedding", "--no-use-prism-normals", "--use-naip-chm-interaction-branch",
        "--exclude-modality", "tree_centered_structure",
        "--phenology-stat-samples", "20000", "--raw-sentinel-stat-samples", "20000",
        "--satellite-embedding-stat-samples", "20000", "--hidden-dim", "128", "--dropout", "0.20",
        "--learning-rate", "0.0005", "--weight-decay", "0.0001", "--grad-clip", "2.0",
        "--class-weighting", "sqrt_inverse", "--class-weight-max", "3.0", "--device", a.device,
        "--export-validation-predictions", "--validation-only-pilot",
    ]
    if a.force:
        cmd.append("--force")
    return cmd


def evaluation_command(a: argparse.Namespace, city: str, model_run: Path) -> list[str]:
    cmd = [
        sys.executable, str(EVALUATOR), "--model-run-dir", str(model_run), "--top-k", "5",
        "--complete-shard-root", str(a.complete_shard_root),
        "--complete-shard-pattern", "*_part*_tree_centered_complete_inputs.npz",
        "--prepared-mmap-root", str(a.prepared_mmap_root), "--output-dir", str(a.result_output_dir),
        "--run-name", f"loco_{city}_test_top5_ensemble", "--grouping-scheme", "species_partition",
        "--species-partition", str(a.species_partition), "--city-token", city, "--split", "test",
        "--batch-size", str(a.batch_size), "--num-workers", "2", "--max-shard-cache", str(a.max_shard_cache),
        "--collapse-map", COLLAPSE_MAP, "--device", a.device, "--export-predictions",
    ]
    if a.force:
        cmd.append("--force")
    return cmd


def main() -> int:
    a = parse_args()
    required = [TRAINER, EVALUATOR, a.complete_shard_root, a.prepared_mmap_root, a.species_partition, a.split_manifest, a.reference_test_predictions]
    for path in required:
        if not path.exists():
            raise FileNotFoundError(path)
    eligible = eligible_cities(a.reference_test_predictions, a.minimum_test_records)
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
        if (result_run / "ensemble_summary.json").exists() and not a.force:
            print(f"Completed result exists; skipping {city}: {result_run}", flush=True)
            continue
        train_cmd = training_command(a, city, run_name)
        eval_cmd = evaluation_command(a, city, model_run)
        if a.dry_run:
            print(subprocess.list2cmdline(train_cmd), flush=True)
            print(subprocess.list2cmdline(eval_cmd), flush=True)
            continue
        if not (model_run / "run_config.json").exists():
            if model_run.exists() and any(model_run.iterdir()) and not a.force:
                raise SystemExit(
                    f"Incomplete model directory exists for {city}: {model_run}. "
                    "Inspect it, then rerun with --force to restart that city."
                )
            run_streamed(train_cmd, a.result_output_dir / "logs" / f"{city}_training.log", ROOT)
        else:
            print(f"Training already complete; proceeding to ensemble: {model_run}", flush=True)
        run_streamed(eval_cmd, a.result_output_dir / "logs" / f"{city}_ensemble.log", ROOT)
        aggregate(a.result_output_dir, cities)
    if not a.dry_run:
        aggregate(a.result_output_dir, cities)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
