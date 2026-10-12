#!/usr/bin/env python3
"""Launch the frozen final k11 stage-two training configuration.

Data locations remain explicit because the multi-terabyte inputs are not part
of the repository. All scientific and optimization choices are loaded from
``final_model_config.json`` and passed to the general training engine.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import artifacts


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def parse_args() -> argparse.Namespace:
    """Parse data locations and runtime-only launcher options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--complete-shard-root", type=Path, required=True)
    parser.add_argument("--prepared-mmap-root", type=Path, default=None)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-name", default="final_pruned_k11_classifier")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def option(name: str, value: object) -> list[str]:
    """Format one long command-line option pair."""
    return [f"--{name.replace('_', '-')}", str(value)]


def main() -> int:
    """Validate frozen artifacts and launch the final training command."""
    args = parse_args()
    artifacts.load_grouping_manifest()
    artifacts.validate_split_manifest(args.split_manifest)
    config = json.loads((HERE / "final_model_config.json").read_text(encoding="utf-8"))
    command = [
        sys.executable,
        str(HERE / "train_fixed_partition_classifier.py"),
        *option("complete_shard_root", args.complete_shard_root),
        *option("split_manifest", args.split_manifest),
        *option("output_dir", args.output_dir),
        *option("run_name", args.run_name),
        *option("seed", config["seed"]),
        *option("epochs", config["epochs"]),
        *option("early_stopping_patience", config["early_stopping_patience"]),
        *option("early_stopping_min_delta", config["early_stopping_min_delta"]),
        *option("checkpoint_selection_score", config["checkpoint_selection_score"]),
        *option("blended_score_macro_weight", config["blended_score_weights"]["macro_f1"]),
        *option("blended_score_accuracy_weight", config["blended_score_weights"]["accuracy"]),
        *option("blended_score_weighted_weight", config["blended_score_weights"]["weighted_f1"]),
        *option("save_top_k_checkpoints", config["save_top_k_checkpoints"]),
        *option("batch_size", config["batch_size"]),
        *option("hidden_dim", config["hidden_dim"]),
        *option("dropout", config["dropout"]),
        *option("learning_rate", config["learning_rate"]),
        *option("weight_decay", config["weight_decay"]),
        *option("grad_clip", config["gradient_clip"]),
        *option("class_weighting", config["class_weighting"]),
        *option("class_weight_max", config["class_weight_max"]),
        *option("image_center_crop_pixels", config["image_center_crop_pixels"]),
        *option("naip_encoder_normalization", config["raster_normalization"]),
        *option("chm_encoder_normalization", config["raster_normalization"]),
        *option("interaction_encoder_normalization", config["raster_normalization"]),
        "--use-naip-chm-interaction-branch",
        *option("interaction_fusion_mode", config["interaction_fusion_mode"]),
        "--independent-chm-full-extent" if config["independent_chm_full_extent"] else "--no-independent-chm-full-extent",
        "--interaction-naip-full-extent" if config["interaction_naip_full_extent"] else "--no-interaction-naip-full-extent",
        *option("interaction_naip_crop_pixels", config["interaction_naip_crop_pixels"]),
        *option("interaction_naip_patch_pixels", config["interaction_naip_patch_pixels"]),
        *option("interaction_chm_patch_pixels", config["interaction_chm_patch_pixels"]),
        *option("phenology_stat_samples", config["feature_stat_samples"]),
        *option("raw_sentinel_stat_samples", config["feature_stat_samples"]),
        *option("satellite_embedding_stat_samples", config["feature_stat_samples"]),
        *option("shard_window_size", config["shard_window_size"]),
        *option("device", args.device),
    ]
    for modality in config["excluded_modalities"]:
        command.extend(option("exclude_modality", modality))
    if args.prepared_mmap_root is not None:
        command.extend(option("prepared_mmap_root", args.prepared_mmap_root))
    if args.overwrite:
        command.append("--overwrite")
    return subprocess.run(command, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
