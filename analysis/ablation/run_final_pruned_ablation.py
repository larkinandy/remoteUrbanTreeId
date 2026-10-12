#!/usr/bin/env python3
"""Run final pruned-model ablations and independent-test ensemble inference.

Unlike pilot runners, this script trains each frozen k=11 leave-one-modality-out
model, selects checkpoints using validation data, and evaluates the ensemble on
the independent test split with the canonical ``collapsed_k6`` mapping.
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from experiment_commands import EVALUATOR, TRAINER, ensemble_command, training_command
from workflow import experiment_config, inference_complete, training_complete, validate_split_manifest

ROOT = Path(__file__).resolve().parents[2]
CONFIG = experiment_config()
FINAL_MODALITIES = tuple(CONFIG["final_pruned_modalities"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--modality", action="append", choices=FINAL_MODALITIES, help="Run only selected branches; repeat as needed.")
    parser.add_argument("--complete-shard-root", type=Path, required=True)
    parser.add_argument("--prepared-mmap-root", type=Path, required=True)
    parser.add_argument("--species-partition", type=Path, default=ROOT / "config" / "final_k11_partition.npz")
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--model-output-dir", type=Path, required=True)
    parser.add_argument("--result-output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=CONFIG["training"]["epochs"])
    parser.add_argument("--early-stopping-patience", type=int, default=CONFIG["training"]["early_stopping_patience"])
    parser.add_argument("--batch-size", type=int, default=CONFIG["training"]["batch_size"])
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--max-shard-cache", type=int, default=3)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing run products.")
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
        exclusions = [*CONFIG["pruned_modalities"], modality]
        train = training_command(args, args.model_output_dir, run_name, exclusions, cache_validation=False)
        evaluate = ensemble_command(args, model_run, args.result_output_dir, f"without_{modality}_test_top5")
        if args.dry_run:
            print(subprocess.list2cmdline(train), flush=True)
            print(subprocess.list2cmdline(evaluate), flush=True)
            continue
        if args.overwrite or not training_complete(model_run, CONFIG["training"]["top_k_checkpoints"]):
            stream(train, args.result_output_dir / "logs" / f"without_{modality}_training.log")
        else:
            print(f"Completed training exists; skipping fit: {model_run}", flush=True)
        if args.overwrite or not inference_complete(result_run):
            stream(evaluate, args.result_output_dir / "logs" / f"without_{modality}_test_inference.log")
        else:
            print(f"Completed test ensemble exists; skipping inference: {result_run}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
