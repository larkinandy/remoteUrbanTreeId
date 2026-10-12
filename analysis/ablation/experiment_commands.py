"""Composable command builders for the frozen stage-two ablation experiments."""

from __future__ import annotations

import sys
from argparse import Namespace
from pathlib import Path

from workflow import canonical_collapse_map, experiment_config


ROOT = Path(__file__).resolve().parents[2]
TRAINER = ROOT / "model" / "stage2" / "train_fixed_partition_classifier.py"
EVALUATOR = ROOT / "model" / "stage2" / "evaluate_checkpoint_ensemble.py"


def _pair(name: str, value: object) -> list[str]:
    return [name, str(value)]


def data_arguments(args: Namespace, output_dir: Path, run_name: str) -> list[str]:
    return [
        *_pair("--complete-shard-root", args.complete_shard_root),
        "--complete-shard-pattern", "*_part*_tree_centered_complete_inputs.npz",
        *_pair("--prepared-mmap-root", args.prepared_mmap_root),
        *_pair("--output-dir", output_dir), *_pair("--run-name", run_name),
        "--grouping-scheme", "species_partition",
        *_pair("--species-partition", args.species_partition),
        *_pair("--split-manifest", args.split_manifest),
    ]


def sampling_arguments(config: dict) -> list[str]:
    sampling = config["sampling"]
    return [
        "--reassign-capped-eval-to-train", "--exclude-qa-from-model",
        "--no-exclude-qa-warning-from-model", "--use-qa-sample-weights",
        *_pair("--low-center-vegetation-weight", sampling["low_center_vegetation_weight"]),
        *_pair("--low-vegetated-height-with-lidar-weight", sampling["low_vegetated_height_with_lidar_weight"]),
        *_pair("--insufficient-lidar-coverage-weight", sampling["insufficient_lidar_coverage_weight"]),
        "--use-distance-sample-weights",
        *_pair("--distance-weight-sigma-m", sampling["distance_weight_sigma_m"]),
        *_pair("--min-distance-sample-weight", sampling["minimum_distance_sample_weight"]),
    ]


def optimization_arguments(args: Namespace, config: dict) -> list[str]:
    training = config["training"]
    weights = training["blended_score_weights"]
    return [
        *_pair("--seed", config["seed"]), *_pair("--epochs", args.epochs),
        *_pair("--early-stopping-patience", args.early_stopping_patience),
        *_pair("--early-stopping-min-delta", training["early_stopping_min_delta"]),
        "--checkpoint-selection-score", training["checkpoint_selection_score"],
        *_pair("--blended-score-macro-weight", weights["macro_f1"]),
        *_pair("--blended-score-accuracy-weight", weights["accuracy"]),
        *_pair("--blended-score-weighted-weight", weights["weighted_f1"]),
        *_pair("--save-top-k-checkpoints", training["top_k_checkpoints"]),
        *_pair("--hidden-dim", training["hidden_dim"]), *_pair("--dropout", training["dropout"]),
        *_pair("--learning-rate", training["learning_rate"]), *_pair("--weight-decay", training["weight_decay"]),
        *_pair("--grad-clip", training["gradient_clip"]),
        "--class-weighting", training["class_weighting"], *_pair("--class-weight-max", training["class_weight_max"]),
    ]


def loader_arguments(args: Namespace, config: dict, cache_validation: bool) -> list[str]:
    sampling = config["sampling"]
    return [
        *_pair("--batch-size", args.batch_size), *_pair("--num-workers", args.num_workers),
        "--eval-num-workers", "0", *_pair("--prefetch-factor", args.prefetch_factor),
        "--persistent-workers", "--shard-local-train-batches",
        *_pair("--shard-window-size", sampling["shard_window_size"]),
        *_pair("--max-shard-cache", args.max_shard_cache),
        "--cache-val-in-memory" if cache_validation else "--no-cache-val-in-memory",
        "--no-cache-test-in-memory",
    ]


def architecture_arguments(config: dict) -> list[str]:
    architecture = config["architecture"]
    feature_stat_samples = config["training"]["feature_stat_samples"]
    return [
        *_pair("--image-center-crop-pixels", architecture["image_center_crop_pixels"]),
        "--image-encoder-normalization", architecture["raster_normalization"],
        "--naip-encoder-normalization", architecture["raster_normalization"],
        "--chm-encoder-normalization", architecture["raster_normalization"],
        "--interaction-encoder-normalization", architecture["raster_normalization"],
        "--independent-chm-full-extent", "--no-interaction-naip-full-extent",
        *_pair("--interaction-naip-crop-pixels", architecture["interaction_naip_crop_pixels"]),
        "--interaction-fusion-mode", architecture["interaction_fusion_mode"],
        *_pair("--interaction-naip-patch-pixels", architecture["interaction_naip_patch_pixels"]),
        *_pair("--interaction-chm-patch-pixels", architecture["interaction_chm_patch_pixels"]),
        "--image-augmentation", "--use-sentinel-phenology", "--use-raw-sentinel",
        "--use-satellite-embedding", "--no-use-prism-normals", "--use-naip-chm-interaction-branch",
        *_pair("--phenology-stat-samples", feature_stat_samples),
        *_pair("--raw-sentinel-stat-samples", feature_stat_samples),
        *_pair("--satellite-embedding-stat-samples", feature_stat_samples),
    ]


def training_command(args: Namespace, output_dir: Path, run_name: str, exclusions: list[str], cache_validation: bool) -> list[str]:
    config = experiment_config()
    command = [sys.executable, str(TRAINER)]
    command += data_arguments(args, output_dir, run_name)
    command += sampling_arguments(config)
    command += optimization_arguments(args, config)
    command += loader_arguments(args, config, cache_validation)
    command += architecture_arguments(config)
    for modality in exclusions:
        command += ["--exclude-modality", modality]
    command += ["--device", args.device, "--export-validation-predictions", "--validation-only-pilot"]
    if args.overwrite:
        command.append("--force")  # Legacy option in the lower-level trainer.
    return command


def ensemble_command(args: Namespace, model_run: Path, output_dir: Path, run_name: str) -> list[str]:
    top_k = experiment_config()["training"]["top_k_checkpoints"]
    command = [
        sys.executable, str(EVALUATOR), *_pair("--model-run-dir", model_run), *_pair("--top-k", top_k),
        *_pair("--complete-shard-root", args.complete_shard_root),
        "--complete-shard-pattern", "*_part*_tree_centered_complete_inputs.npz",
        *_pair("--prepared-mmap-root", args.prepared_mmap_root), *_pair("--output-dir", output_dir),
        *_pair("--run-name", run_name), "--grouping-scheme", "species_partition",
        *_pair("--species-partition", args.species_partition), *_pair("--split-manifest", args.split_manifest),
        "--split", "test", *_pair("--batch-size", args.batch_size), *_pair("--num-workers", args.num_workers),
        *_pair("--max-shard-cache", args.max_shard_cache), "--collapse-map", canonical_collapse_map(),
        "--device", args.device, "--export-predictions",
    ]
    if args.overwrite:
        command.append("--force")  # Legacy option in the lower-level evaluator.
    return command
