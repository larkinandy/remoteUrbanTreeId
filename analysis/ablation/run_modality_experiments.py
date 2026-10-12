#!/usr/bin/env python3
"""Launch publication-quality k=11 modality experiments.

Each experiment retrains from scratch with the frozen k=11 partition and
inherited stage-two splits. Use repeated ``--modality`` arguments to distribute
different jobs across workstations. Test inference is disabled.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
TRAINER = ROOT / "model" / "stage2" / "train_fixed_partition_classifier.py"
DEFAULT_SHARDS = Path(r"H:\TreeCenteredModelInputs\tree_centered_complete_sharded100k_prism_daily3_clean")
DEFAULT_OUTPUT = Path(r"H:\TreeCenteredModelInputs\collapsed_group_models_clean\reviewer_revision\k11_modality_experiments_v1")
DEFAULT_PREPARED_MMAP = Path(r"H:\TreeCenteredModelInputs\tree_centered_prepared_float32_v1")
DEFAULT_PARTITION = ROOT / "config" / "final_k11_partition.npz"
DEFAULT_SPLITS = (
    ROOT / "outputs" / "reviewer_revision" / "stage2_partitions_k11_groupnorm_v1"
    / "reviewer_stage2_loader_manifest.csv.gz"
)

MODALITIES = (
    "tree_centered_naip",
    "tree_centered_chm",
    "tree_centered_structure",
    "naip_chm_interaction",
    "sentinel_phenology",
    "raw_sentinel",
    "satellite_embedding",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", choices=("leave_one_out", "single_modality"), required=True)
    parser.add_argument(
        "--modality", action="append", choices=MODALITIES,
        help="Run only this target modality; repeat to queue several. Omit to run all seven.",
    )
    parser.add_argument("--complete-shard-root", type=Path, default=DEFAULT_SHARDS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--prepared-mmap-root", type=Path, default=DEFAULT_PREPARED_MMAP)
    parser.add_argument("--species-partition", type=Path, default=DEFAULT_PARTITION)
    parser.add_argument("--split-manifest", type=Path, default=DEFAULT_SPLITS)
    parser.add_argument("--run-suffix", default="v1")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--skip-existing", action=argparse.BooleanOptionalAction, default=True,
        help="Skip a run when its output directory already contains history.csv.",
    )
    return parser.parse_args()


def exclusions(experiment: str, modality: str) -> list[str]:
    if experiment == "leave_one_out":
        return [modality]
    return [candidate for candidate in MODALITIES if candidate != modality]


def run_name(experiment: str, modality: str, suffix: str) -> str:
    prefix = "k11_loo_without" if experiment == "leave_one_out" else "k11_single"
    return f"{prefix}_{modality}_{suffix}"


def command(args: argparse.Namespace, modality: str) -> tuple[str, list[str]]:
    name = run_name(args.experiment, modality, args.run_suffix)
    cmd = [
        sys.executable,
        str(TRAINER),
        "--complete-shard-root", str(args.complete_shard_root),
        "--complete-shard-pattern", "*_part*_tree_centered_complete_inputs.npz",
        "--prepared-mmap-root", str(args.prepared_mmap_root),
        "--output-dir", str(args.output_dir),
        "--run-name", name,
        "--grouping-scheme", "species_partition",
        "--species-partition", str(args.species_partition),
        "--split-manifest", str(args.split_manifest),
        "--seed", "13",
        "--reassign-capped-eval-to-train",
        "--exclude-qa-from-model",
        "--no-exclude-qa-warning-from-model",
        "--use-qa-sample-weights",
        "--low-center-vegetation-weight", "0.25",
        "--low-vegetated-height-with-lidar-weight", "0.50",
        "--insufficient-lidar-coverage-weight", "0.75",
        "--use-distance-sample-weights",
        "--distance-weight-sigma-m", "4.0",
        "--min-distance-sample-weight", "0.25",
        "--epochs", "120",
        "--early-stopping-patience", "12",
        "--early-stopping-min-delta", "0.0001",
        "--checkpoint-selection-score", "blended",
        "--blended-score-macro-weight", "0.50",
        "--blended-score-accuracy-weight", "0.30",
        "--blended-score-weighted-weight", "0.20",
        "--save-top-k-checkpoints", "5",
        "--batch-size", "1024",
        "--num-workers", "3",
        "--eval-num-workers", "0",
        "--prefetch-factor", "3",
        "--persistent-workers",
        "--shard-local-train-batches",
        "--shard-window-size", "8",
        "--max-shard-cache", "10",
        "--cache-val-in-memory",
        "--no-cache-test-in-memory",
        "--image-center-crop-pixels", "28",
        "--image-encoder-normalization", "group",
        "--naip-encoder-normalization", "group",
        "--chm-encoder-normalization", "group",
        "--interaction-encoder-normalization", "group",
        "--independent-chm-full-extent",
        "--no-interaction-naip-full-extent",
        "--interaction-naip-crop-pixels", "30",
        "--interaction-fusion-mode", "learned_common_grid",
        "--interaction-naip-patch-pixels", "5",
        "--interaction-chm-patch-pixels", "3",
        "--image-augmentation",
        "--use-sentinel-phenology",
        "--use-raw-sentinel",
        "--use-satellite-embedding",
        "--no-use-prism-normals",
        "--use-naip-chm-interaction-branch",
        "--phenology-stat-samples", "20000",
        "--raw-sentinel-stat-samples", "20000",
        "--satellite-embedding-stat-samples", "20000",
        "--hidden-dim", "128",
        "--dropout", "0.20",
        "--learning-rate", "0.0005",
        "--weight-decay", "0.0001",
        "--grad-clip", "2.0",
        "--class-weighting", "sqrt_inverse",
        "--class-weight-max", "3.0",
        "--device", args.device,
        "--export-validation-predictions",
        "--validation-only-pilot",
    ]
    for excluded in exclusions(args.experiment, modality):
        cmd.extend(["--exclude-modality", excluded])
    return name, cmd


def display(cmd: list[str]) -> str:
    return subprocess.list2cmdline(cmd)


def main() -> int:
    args = parse_args()
    for path, label in (
        (TRAINER, "trainer"),
        (args.complete_shard_root, "complete shard root"),
        (args.prepared_mmap_root, "prepared mmap root"),
        (args.species_partition, "k=11 partition"),
        (args.split_manifest, "split manifest"),
    ):
        if not path.exists():
            raise FileNotFoundError(f"Missing {label}: {path}")
    targets = args.modality or list(MODALITIES)
    if not args.dry_run:
        args.output_dir.mkdir(parents=True, exist_ok=True)
    for index, modality in enumerate(targets, 1):
        name, cmd = command(args, modality)
        run_dir = args.output_dir / name
        if args.skip_existing and (run_dir / "history.csv").exists():
            print(f"[{index}/{len(targets)}] Skipping existing run: {run_dir}", flush=True)
            continue
        if run_dir.exists() and any(run_dir.iterdir()):
            # A setup-only directory can remain when a job is interrupted before
            # epoch 1. The trainer may safely overwrite those deterministic CSVs.
            cmd.append("--force")
            print(f"[{index}/{len(targets)}] Reusing incomplete setup directory: {run_dir}", flush=True)
        print(f"[{index}/{len(targets)}] {args.experiment}: {modality}", flush=True)
        print(display(cmd), flush=True)
        if not args.dry_run:
            subprocess.run(cmd, cwd=ROOT, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
