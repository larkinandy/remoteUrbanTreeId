#!/usr/bin/env python3
"""Run validation-only pilot modality experiments for the frozen k=11 model.

Each experiment retrains from scratch with the frozen k=11 partition and
inherited stage-two splits. Use repeated ``--modality`` arguments to distribute
different jobs across workstations. Checkpoint selection and reporting here are
validation-only; independent-test inference is handled by a separate script.
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from experiment_commands import TRAINER, training_command
from workflow import experiment_config, training_complete, validate_split_manifest


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PARTITION = ROOT / "config" / "final_k11_partition.npz"
CONFIG = experiment_config()
MODALITIES = tuple(CONFIG["modalities"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", choices=("leave_one_out", "single_modality"), required=True)
    parser.add_argument(
        "--modality", action="append", choices=MODALITIES,
        help="Run only this target modality; repeat to queue several. Omit to run all seven.",
    )
    parser.add_argument("--complete-shard-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prepared-mmap-root", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--run-suffix", default="v1")
    parser.add_argument("--epochs", type=int, default=CONFIG["training"]["epochs"])
    parser.add_argument("--early-stopping-patience", type=int, default=CONFIG["training"]["early_stopping_patience"])
    parser.add_argument("--batch-size", type=int, default=CONFIG["training"]["batch_size"])
    parser.add_argument("--num-workers", type=int, default=3)
    parser.add_argument("--prefetch-factor", type=int, default=3)
    parser.add_argument("--max-shard-cache", type=int, default=10)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing run, including complete outputs.")
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
    return name, training_command(args, args.output_dir, name, exclusions(args.experiment, modality), cache_validation=True)


def display(cmd: list[str]) -> str:
    return subprocess.list2cmdline(cmd)


def main() -> int:
    args = parse_args()
    for path, label in (
        (TRAINER, "trainer"),
        (args.complete_shard_root, "complete shard root"),
        (args.prepared_mmap_root, "prepared mmap root"),
        (args.split_manifest, "split manifest"),
    ):
        if not path.exists():
            raise FileNotFoundError(f"Missing {label}: {path}")
    validate_split_manifest(args.split_manifest)
    targets = args.modality or list(MODALITIES)
    if not args.dry_run:
        args.output_dir.mkdir(parents=True, exist_ok=True)
    for index, modality in enumerate(targets, 1):
        name, cmd = command(args, modality)
        run_dir = args.output_dir / name
        complete = training_complete(run_dir, CONFIG["training"]["top_k_checkpoints"])
        if complete and not args.overwrite:
            print(f"[{index}/{len(targets)}] Skipping verified complete run: {run_dir}", flush=True)
            continue
        if run_dir.exists() and any(run_dir.iterdir()) and not args.overwrite:
            raise RuntimeError(f"Incomplete existing run requires --overwrite: {run_dir}")
        print(f"[{index}/{len(targets)}] {args.experiment}: {modality}", flush=True)
        print(display(cmd), flush=True)
        if not args.dry_run:
            subprocess.run(cmd, cwd=ROOT, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
