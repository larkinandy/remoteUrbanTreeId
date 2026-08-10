#!/usr/bin/env python3
"""Train clean tree_id-centered k=6 collapsed-group classifiers.

This script uses the clean standalone complete shards and maps each record's
scientific label through a saved centroid partition. It supports the two k=6
diagnostics we want to compare:

* species_k6: species/binomial-level partition, genera may be split.
* genus_k6: genus-constrained partition, each genus appears in one group.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import train_clean_tree_id_centered_taxon_discriminator as clean_disc


CLEAN_ROOT = Path(r"H:\TreeCenteredModelInputs")
DEFAULT_SHARD_ROOT = CLEAN_ROOT / "tree_centered_complete_sharded100k_clean"
DEFAULT_OUTPUT_DIR = CLEAN_ROOT / "collapsed_group_models_clean"
DEFAULT_PARTITION_BASE = (
    CLEAN_ROOT
    / "taxon_discrimination_clean"
    / "clean_abq_atl_taxon_discriminator"
)
DEFAULT_SPECIES_K6 = (
    DEFAULT_PARTITION_BASE
    / "global_centroid_taxon_partitions_clean_exploratory"
    / "k06_partition.npz"
)
DEFAULT_GENUS_K6 = (
    DEFAULT_PARTITION_BASE
    / "genus_constrained_centroid_partitions_clean_exploratory"
    / "k06_partition.npz"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--complete-shard-root", type=Path, default=DEFAULT_SHARD_ROOT)
    parser.add_argument("--complete-shard-pattern", default="*_part*_tree_centered_complete_inputs.npz")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--run-name", default="clean_tree_id_centered_k6_classifier")
    parser.add_argument("--grouping-scheme", choices=("species_k6", "genus_k6"), required=True)
    parser.add_argument("--species-k6-partition", type=Path, default=DEFAULT_SPECIES_K6)
    parser.add_argument("--genus-k6-partition", type=Path, default=DEFAULT_GENUS_K6)
    parser.add_argument("--city-token", action="append", default=None)
    parser.add_argument("--exclude-city-token", action="append", default=[])
    parser.add_argument("--max-records-per-city", type=int, default=0)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--val-fraction", type=float, default=0.10)
    parser.add_argument("--test-fraction", type=float, default=0.10)
    parser.add_argument("--max-val-samples", type=int, default=150000)
    parser.add_argument("--max-test-samples", type=int, default=150000)
    parser.add_argument("--reassign-capped-eval-to-train", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--exclude-qa-from-model", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--exclude-qa-warning-from-model",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Also remove retained QA warning records from train/val/test. Use this for a "
            "high-confidence product where the model is only applied to no-QA-flag trees."
        ),
    )
    parser.add_argument("--use-qa-sample-weights", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--low-center-vegetation-weight", type=float, default=0.25)
    parser.add_argument("--low-vegetated-height-with-lidar-weight", type=float, default=0.50)
    parser.add_argument("--insufficient-lidar-coverage-weight", type=float, default=0.75)
    parser.add_argument("--use-distance-sample-weights", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--distance-weight-sigma-m", type=float, default=4.0)
    parser.add_argument("--min-distance-sample-weight", type=float, default=0.25)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--early-stopping-patience", type=int, default=12)
    parser.add_argument("--early-stopping-min-delta", type=float, default=1.0e-4)
    parser.add_argument("--resume-from-checkpoint", type=Path, default=None)
    parser.add_argument(
        "--checkpoint-selection-score",
        choices=("macro_f1", "accuracy", "weighted_f1", "blended"),
        default="macro_f1",
        help="Validation metric used for best_model.pt, early stopping, and optional top-k checkpoint retention.",
    )
    parser.add_argument("--blended-score-macro-weight", type=float, default=0.50)
    parser.add_argument("--blended-score-accuracy-weight", type=float, default=0.30)
    parser.add_argument("--blended-score-weighted-weight", type=float, default=0.20)
    parser.add_argument("--save-top-k-checkpoints", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--num-workers", type=int, default=3)
    parser.add_argument("--eval-num-workers", type=int, default=0)
    parser.add_argument("--prefetch-factor", type=int, default=3)
    parser.add_argument("--persistent-workers", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--shard-local-train-batches", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--shard-window-size", type=int, default=8)
    parser.add_argument("--max-shard-cache", type=int, default=10)
    parser.add_argument("--cache-val-in-memory", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--cache-test-in-memory", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--image-center-crop-pixels", type=int, default=32)
    parser.add_argument(
        "--independent-chm-full-extent",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Keep the independent CHM encoder at the full native chip extent while "
            "using the NAIP-aligned CHM center crop for the interaction encoder."
        ),
    )
    parser.add_argument(
        "--interaction-naip-full-extent",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Use the full native NAIP chip in the NAIP-CHM interaction branch. "
            "The independent NAIP branch still uses --image-center-crop-pixels."
        ),
    )
    parser.add_argument(
        "--interaction-naip-crop-pixels",
        type=int,
        default=0,
        help=(
            "Optional NAIP center-crop size used only by the NAIP-CHM interaction "
            "branch. Zero reuses --image-center-crop-pixels. Ignored when "
            "--interaction-naip-full-extent is enabled."
        ),
    )
    parser.add_argument(
        "--interaction-fusion-mode",
        choices=("bilinear_upsample", "learned_common_grid", "shared_naip_common_grid"),
        default="bilinear_upsample",
        help=(
            "How the interaction branch aligns NAIP and CHM. The legacy mode "
            "bilinearly upsamples CHM to the NAIP grid. learned_common_grid uses "
            "separate learned non-overlapping convolutions to map the native grids "
            "to a shared spatial feature grid before pixel-aligned feature fusion. "
            "shared_naip_common_grid also reuses the aligned NAIP feature map for "
            "the independent NAIP modality."
        ),
    )
    parser.add_argument(
        "--interaction-naip-patch-pixels",
        type=int,
        default=5,
        help="NAIP pixels per learned common-grid cell.",
    )
    parser.add_argument(
        "--interaction-chm-patch-pixels",
        type=int,
        default=3,
        help="CHM pixels per learned common-grid cell.",
    )
    parser.add_argument("--image-augmentation", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--tree-image-channel-dropout", type=float, default=0.0)
    parser.add_argument("--tree-image-branch-dropout", type=float, default=0.0)
    parser.add_argument("--satellite-embedding-dropout", type=float, default=0.0)
    parser.add_argument("--satellite-embedding-value-dropout", type=float, default=0.0)
    parser.add_argument("--prism-normals-dropout", type=float, default=0.0)
    parser.add_argument("--prism-normals-value-dropout", type=float, default=0.0)
    parser.add_argument("--use-sentinel-phenology", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--zero-phenology-feature-regex",
        action="append",
        default=[],
        help=(
            "Regex selecting standardized Sentinel phenology columns that are "
            "always replaced by zero (the training mean). Repeat for multiple "
            "families. The matched names and indices are saved with the run."
        ),
    )
    parser.add_argument("--use-raw-sentinel", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use-satellite-embedding", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use-prism-normals", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--use-naip-chm-interaction-branch", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--phenology-stat-samples", type=int, default=20000)
    parser.add_argument("--raw-sentinel-stat-samples", type=int, default=20000)
    parser.add_argument("--satellite-embedding-stat-samples", type=int, default=20000)
    parser.add_argument("--prism-normals-stat-samples", type=int, default=20000)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.20)
    parser.add_argument("--learning-rate", type=float, default=5.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--grad-clip", type=float, default=2.0)
    parser.add_argument("--class-weighting", choices=("none", "sqrt_inverse", "inverse"), default="sqrt_inverse")
    parser.add_argument("--class-weight-max", type=float, default=3.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--export-predictions", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def selection_score(metrics: dict[str, Any], args: argparse.Namespace) -> float:
    if args.checkpoint_selection_score == "accuracy":
        return float(metrics["accuracy"])
    if args.checkpoint_selection_score == "weighted_f1":
        return float(metrics["weighted_f1"])
    if args.checkpoint_selection_score == "blended":
        return float(
            args.blended_score_macro_weight * metrics["macro_f1"]
            + args.blended_score_accuracy_weight * metrics["accuracy"]
            + args.blended_score_weighted_weight * metrics["weighted_f1"]
        )
    return float(metrics["macro_f1"])


def checkpoint_payload(
    model,
    group_names: list[str],
    label_to_group: dict[str, int],
    structure_mean: np.ndarray,
    structure_std: np.ndarray,
    phenology_mean: np.ndarray | None,
    phenology_std: np.ndarray | None,
    raw_sentinel_mean: np.ndarray | None,
    raw_sentinel_std: np.ndarray | None,
    satellite_embedding_mean: np.ndarray | None,
    satellite_embedding_std: np.ndarray | None,
    prism_mean: np.ndarray | None,
    prism_std: np.ndarray | None,
    args: argparse.Namespace,
    epoch: int,
    val_metrics: dict[str, Any],
    score: float,
) -> dict[str, Any]:
    return {
        "model_state_dict": model.state_dict(),
        "group_names": group_names,
        "label_to_group": label_to_group,
        "structure_mean": clean_disc.train_base.torch.as_tensor(structure_mean),
        "structure_std": clean_disc.train_base.torch.as_tensor(structure_std),
        "phenology_mean": None if phenology_mean is None else clean_disc.train_base.torch.as_tensor(phenology_mean),
        "phenology_std": None if phenology_std is None else clean_disc.train_base.torch.as_tensor(phenology_std),
        "raw_sentinel_mean": None if raw_sentinel_mean is None else clean_disc.train_base.torch.as_tensor(raw_sentinel_mean),
        "raw_sentinel_std": None if raw_sentinel_std is None else clean_disc.train_base.torch.as_tensor(raw_sentinel_std),
        "satellite_embedding_mean": None if satellite_embedding_mean is None else clean_disc.train_base.torch.as_tensor(satellite_embedding_mean),
        "satellite_embedding_std": None if satellite_embedding_std is None else clean_disc.train_base.torch.as_tensor(satellite_embedding_std),
        "prism_mean": None if prism_mean is None else clean_disc.train_base.torch.as_tensor(prism_mean),
        "prism_std": None if prism_std is None else clean_disc.train_base.torch.as_tensor(prism_std),
        "epoch": int(epoch),
        "selection_score": float(score),
        "selection_metric": args.checkpoint_selection_score,
        "val_metrics": {key: value for key, value in val_metrics.items() if key != "per_class"},
        "args": clean_disc.safe_json_args(args),
    }


def save_top_k_checkpoint(
    payload: dict[str, Any],
    run_dir: Path,
    top_k: int,
    saved_top: list[tuple[float, Path]],
) -> list[tuple[float, Path]]:
    if top_k <= 0:
        return saved_top
    ckpt_dir = run_dir / "top_checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    score = float(payload["selection_score"])
    epoch = int(payload["epoch"])
    path = ckpt_dir / f"checkpoint_epoch_{epoch:03d}_score_{score:.6f}.pt"
    clean_disc.train_base.torch.save(payload, path)
    saved_top.append((score, path))
    saved_top = sorted(saved_top, key=lambda item: item[0], reverse=True)
    while len(saved_top) > top_k:
        _old_score, old_path = saved_top.pop(-1)
        if old_path.exists():
            old_path.unlink()
    manifest = pd.DataFrame(
        [{"rank": rank + 1, "score": value, "checkpoint": str(path)} for rank, (value, path) in enumerate(saved_top)]
    )
    manifest.to_csv(ckpt_dir / "top_checkpoints.csv", index=False)
    return saved_top


def partition_path(args: argparse.Namespace) -> Path:
    return args.genus_k6_partition if args.grouping_scheme == "genus_k6" else args.species_k6_partition


def load_partition(path: Path) -> tuple[dict[str, int], list[str]]:
    if not path.exists():
        raise FileNotFoundError(path)
    data = np.load(path, allow_pickle=True)
    labels = [str(value) for value in data["label_columns"]]
    if "label_group" in data.files:
        groups = np.asarray(data["label_group"], dtype=np.int64)
    elif "assignments" in data.files:
        groups = np.asarray(data["assignments"], dtype=np.int64)
    else:
        raise KeyError(f"{path} does not contain label_group or assignments.")
    if len(labels) != len(groups):
        raise RuntimeError(f"{path} has mismatched label/group lengths.")
    group_ids = sorted(int(value) for value in np.unique(groups))
    group_names = [f"group{group_id + 1:02d}" for group_id in group_ids]
    group_remap = {group_id: index for index, group_id in enumerate(group_ids)}
    label_to_group = {label: group_remap[int(group)] for label, group in zip(labels, groups)}
    return label_to_group, group_names


def ensure_shard_pattern(args: argparse.Namespace) -> None:
    """Use a known clean-shard filename pattern before calling shared discovery."""
    root = Path(args.complete_shard_root)
    if list(root.glob(f"*/{args.complete_shard_pattern}")):
        return
    fallback_patterns = (
        "*_part*_tree_centered_complete_inputs.npz",
        "*_tree_centered_complete_inputs.npz",
        "*complete_inputs.npz",
    )
    for pattern in fallback_patterns:
        if pattern == args.complete_shard_pattern:
            continue
        if list(root.glob(f"*/{pattern}")):
            print(
                f"No shards matched --complete-shard-pattern={args.complete_shard_pattern!r}; "
                f"using {pattern!r} instead.",
                flush=True,
            )
            args.complete_shard_pattern = pattern
            return


class LearnedCommonGridInteractionEncoder(clean_disc.train_base.nn.Module):
    """Fuse native-resolution NAIP and CHM on a learned common spatial grid."""

    def __init__(
        self,
        hidden_dim: int,
        dropout: float,
        naip_patch_pixels: int = 5,
        chm_patch_pixels: int = 3,
    ):
        super().__init__()
        if naip_patch_pixels <= 0 or chm_patch_pixels <= 0:
            raise ValueError("Interaction patch dimensions must be positive.")
        self.naip_patch_pixels = int(naip_patch_pixels)
        self.chm_patch_pixels = int(chm_patch_pixels)
        branch_dim = 32
        self.naip_to_common_grid = clean_disc.train_base.nn.Sequential(
            clean_disc.train_base.nn.Conv2d(
                5,
                branch_dim,
                kernel_size=self.naip_patch_pixels,
                stride=self.naip_patch_pixels,
            ),
            clean_disc.train_base.nn.BatchNorm2d(branch_dim),
            clean_disc.train_base.nn.GELU(),
        )
        self.chm_to_common_grid = clean_disc.train_base.nn.Sequential(
            clean_disc.train_base.nn.Conv2d(
                3,
                branch_dim,
                kernel_size=self.chm_patch_pixels,
                stride=self.chm_patch_pixels,
            ),
            clean_disc.train_base.nn.BatchNorm2d(branch_dim),
            clean_disc.train_base.nn.GELU(),
        )
        self.spatial_interaction = clean_disc.train_base.nn.Sequential(
            clean_disc.train_base.nn.Conv2d(branch_dim * 2, 64, 3, padding=1),
            clean_disc.train_base.nn.BatchNorm2d(64),
            clean_disc.train_base.nn.GELU(),
            clean_disc.train_base.nn.Conv2d(64, hidden_dim, 3, padding=1),
            clean_disc.train_base.nn.BatchNorm2d(hidden_dim),
            clean_disc.train_base.nn.GELU(),
            clean_disc.train_base.nn.AdaptiveAvgPool2d(1),
            clean_disc.train_base.nn.Flatten(),
            clean_disc.train_base.nn.Dropout(dropout),
        )

    def forward(
        self,
        naip: clean_disc.train_base.torch.Tensor,
        chm: clean_disc.train_base.torch.Tensor,
    ) -> clean_disc.train_base.torch.Tensor:
        if naip.shape[-2] % self.naip_patch_pixels or naip.shape[-1] % self.naip_patch_pixels:
            raise ValueError(
                f"NAIP interaction shape {tuple(naip.shape[-2:])} is not divisible by "
                f"{self.naip_patch_pixels}."
            )
        if chm.shape[-2] % self.chm_patch_pixels or chm.shape[-1] % self.chm_patch_pixels:
            raise ValueError(
                f"CHM interaction shape {tuple(chm.shape[-2:])} is not divisible by "
                f"{self.chm_patch_pixels}."
            )
        naip_features = self.naip_to_common_grid(naip)
        chm_features = self.chm_to_common_grid(chm)
        if naip_features.shape[-2:] != chm_features.shape[-2:]:
            raise ValueError(
                "Learned interaction grids do not align: "
                f"NAIP={tuple(naip_features.shape[-2:])}, "
                f"CHM={tuple(chm_features.shape[-2:])}. Adjust crop and patch dimensions."
            )
        return self.spatial_interaction(
            clean_disc.train_base.torch.cat([naip_features, chm_features], dim=1)
        )


class SharedNaipCommonGridEncoder(clean_disc.train_base.nn.Module):
    """Produce independent and interaction features from one aligned NAIP map."""

    def __init__(
        self,
        hidden_dim: int,
        dropout: float,
        naip_patch_pixels: int = 5,
        chm_patch_pixels: int = 3,
    ):
        super().__init__()
        if naip_patch_pixels <= 0 or chm_patch_pixels <= 0:
            raise ValueError("Interaction patch dimensions must be positive.")
        self.naip_patch_pixels = int(naip_patch_pixels)
        self.chm_patch_pixels = int(chm_patch_pixels)
        branch_dim = 32
        self.naip_to_common_grid = clean_disc.train_base.nn.Sequential(
            clean_disc.train_base.nn.Conv2d(
                5,
                branch_dim,
                kernel_size=self.naip_patch_pixels,
                stride=self.naip_patch_pixels,
            ),
            clean_disc.train_base.nn.BatchNorm2d(branch_dim),
            clean_disc.train_base.nn.GELU(),
        )
        self.independent_naip_head = clean_disc.train_base.nn.Sequential(
            clean_disc.train_base.nn.Conv2d(branch_dim, 64, 3, padding=1),
            clean_disc.train_base.nn.BatchNorm2d(64),
            clean_disc.train_base.nn.GELU(),
            clean_disc.train_base.nn.Conv2d(64, hidden_dim, 3, padding=1),
            clean_disc.train_base.nn.BatchNorm2d(hidden_dim),
            clean_disc.train_base.nn.GELU(),
            clean_disc.train_base.nn.AdaptiveAvgPool2d(1),
            clean_disc.train_base.nn.Flatten(),
            clean_disc.train_base.nn.Dropout(dropout),
        )
        self.chm_to_common_grid = clean_disc.train_base.nn.Sequential(
            clean_disc.train_base.nn.Conv2d(
                3,
                branch_dim,
                kernel_size=self.chm_patch_pixels,
                stride=self.chm_patch_pixels,
            ),
            clean_disc.train_base.nn.BatchNorm2d(branch_dim),
            clean_disc.train_base.nn.GELU(),
        )
        self.spatial_interaction = clean_disc.train_base.nn.Sequential(
            clean_disc.train_base.nn.Conv2d(branch_dim * 2, 64, 3, padding=1),
            clean_disc.train_base.nn.BatchNorm2d(64),
            clean_disc.train_base.nn.GELU(),
            clean_disc.train_base.nn.Conv2d(64, hidden_dim, 3, padding=1),
            clean_disc.train_base.nn.BatchNorm2d(hidden_dim),
            clean_disc.train_base.nn.GELU(),
            clean_disc.train_base.nn.AdaptiveAvgPool2d(1),
            clean_disc.train_base.nn.Flatten(),
            clean_disc.train_base.nn.Dropout(dropout),
        )

    def forward(
        self,
        naip: clean_disc.train_base.torch.Tensor,
        chm: clean_disc.train_base.torch.Tensor,
    ) -> tuple[clean_disc.train_base.torch.Tensor, clean_disc.train_base.torch.Tensor]:
        if naip.shape[-2] % self.naip_patch_pixels or naip.shape[-1] % self.naip_patch_pixels:
            raise ValueError(
                f"NAIP interaction shape {tuple(naip.shape[-2:])} is not divisible by "
                f"{self.naip_patch_pixels}."
            )
        if chm.shape[-2] % self.chm_patch_pixels or chm.shape[-1] % self.chm_patch_pixels:
            raise ValueError(
                f"CHM interaction shape {tuple(chm.shape[-2:])} is not divisible by "
                f"{self.chm_patch_pixels}."
            )
        naip_features = self.naip_to_common_grid(naip)
        chm_features = self.chm_to_common_grid(chm)
        if naip_features.shape[-2:] != chm_features.shape[-2:]:
            raise ValueError(
                "Shared interaction grids do not align: "
                f"NAIP={tuple(naip_features.shape[-2:])}, "
                f"CHM={tuple(chm_features.shape[-2:])}."
            )
        independent_naip = self.independent_naip_head(naip_features)
        interaction = self.spatial_interaction(
            clean_disc.train_base.torch.cat([naip_features, chm_features], dim=1)
        )
        return independent_naip, interaction


class CleanK6Classifier(clean_disc.train_base.nn.Module):
    def __init__(
        self,
        class_count: int,
        structure_dim: int,
        hidden_dim: int,
        dropout: float,
        phenology_dim: int = 0,
        raw_sentinel_dim: int = 0,
        satellite_embedding_dim: int = 0,
        prism_normals_dim: int = 0,
        tree_image_branch_dropout: float = 0.0,
        use_naip_chm_interaction_branch: bool = False,
        independent_naip_crop_pixels: int = 0,
        interaction_naip_crop_pixels: int = 0,
        interaction_fusion_mode: str = "bilinear_upsample",
        interaction_naip_patch_pixels: int = 5,
        interaction_chm_patch_pixels: int = 3,
        phenology_zero_indices: list[int] | tuple[int, ...] | None = None,
    ):
        super().__init__()
        self.tree_image_branch_dropout = float(tree_image_branch_dropout)
        self.tree_image_modality_index = 0
        self.use_naip_chm_interaction_branch = bool(use_naip_chm_interaction_branch)
        self.independent_naip_crop_pixels = max(0, int(independent_naip_crop_pixels))
        self.interaction_naip_crop_pixels = max(0, int(interaction_naip_crop_pixels))
        self.interaction_fusion_mode = str(interaction_fusion_mode)
        self.phenology_zero_indices = tuple(int(value) for value in (phenology_zero_indices or ()))
        self.modality_names = ("tree_centered_naip", "tree_centered_chm", "tree_centered_structure")
        if self.use_naip_chm_interaction_branch:
            self.modality_names += ("naip_chm_interaction",)
        if phenology_dim > 0:
            self.modality_names += ("sentinel_phenology",)
        if raw_sentinel_dim > 0:
            self.modality_names += ("raw_sentinel",)
        if satellite_embedding_dim > 0:
            self.modality_names += ("satellite_embedding",)
        if prism_normals_dim > 0:
            self.modality_names += ("prism_normals",)

        self.image_encoder = (
            None
            if self.use_naip_chm_interaction_branch
            and self.interaction_fusion_mode == "shared_naip_common_grid"
            else clean_disc.train_base.ImageEncoder(5, hidden_dim, dropout)
        )
        self.chm_encoder = clean_disc.train_base.ImageEncoder(3, hidden_dim, dropout)
        if self.use_naip_chm_interaction_branch and self.interaction_fusion_mode == "shared_naip_common_grid":
            self.interaction_encoder = SharedNaipCommonGridEncoder(
                hidden_dim=hidden_dim,
                dropout=dropout,
                naip_patch_pixels=interaction_naip_patch_pixels,
                chm_patch_pixels=interaction_chm_patch_pixels,
            )
        elif self.use_naip_chm_interaction_branch and self.interaction_fusion_mode == "learned_common_grid":
            self.interaction_encoder = LearnedCommonGridInteractionEncoder(
                hidden_dim=hidden_dim,
                dropout=dropout,
                naip_patch_pixels=interaction_naip_patch_pixels,
                chm_patch_pixels=interaction_chm_patch_pixels,
            )
        elif self.use_naip_chm_interaction_branch and self.interaction_fusion_mode == "bilinear_upsample":
            self.interaction_encoder = clean_disc.train_base.ImageEncoder(8, hidden_dim, dropout)
        elif self.use_naip_chm_interaction_branch:
            raise ValueError(f"Unknown interaction_fusion_mode={self.interaction_fusion_mode!r}")
        else:
            self.interaction_encoder = None
        self.structure_encoder = clean_disc.PhenologyEncoder(structure_dim, hidden_dim, dropout)
        self.phenology_encoder = clean_disc.PhenologyEncoder(phenology_dim, hidden_dim, dropout) if phenology_dim > 0 else None
        self.raw_sentinel_encoder = clean_disc.train_base.MaskedSequenceEncoder(raw_sentinel_dim, hidden_dim, dropout) if raw_sentinel_dim > 0 else None
        self.satellite_embedding_encoder = clean_disc.PhenologyEncoder(satellite_embedding_dim, hidden_dim, dropout) if satellite_embedding_dim > 0 else None
        self.prism_normals_encoder = clean_disc.PhenologyEncoder(prism_normals_dim, hidden_dim, dropout) if prism_normals_dim > 0 else None
        modality_count = len(self.modality_names)
        self.gate = clean_disc.train_base.nn.Sequential(
            clean_disc.train_base.nn.LayerNorm(hidden_dim * modality_count),
            clean_disc.train_base.nn.Linear(hidden_dim * modality_count, hidden_dim),
            clean_disc.train_base.nn.GELU(),
            clean_disc.train_base.nn.Dropout(dropout),
            clean_disc.train_base.nn.Linear(hidden_dim, modality_count),
        )
        self.head = clean_disc.train_base.nn.Sequential(
            clean_disc.train_base.nn.LayerNorm(hidden_dim),
            clean_disc.train_base.nn.Linear(hidden_dim, hidden_dim),
            clean_disc.train_base.nn.GELU(),
            clean_disc.train_base.nn.Dropout(dropout),
            clean_disc.train_base.nn.Linear(hidden_dim, class_count),
        )

    def forward(self, batch: dict[str, clean_disc.train_base.torch.Tensor]):
        tree_image = batch["tree_image"]
        tree_chm_image = batch["tree_chm_image"]
        independent_tree_image = tree_image
        if (
            self.interaction_naip_crop_pixels > 0
            and self.independent_naip_crop_pixels > 0
            and self.independent_naip_crop_pixels < min(tree_image.shape[-2:])
        ):
            crop = self.independent_naip_crop_pixels
            y0 = (tree_image.shape[-2] - crop) // 2
            x0 = (tree_image.shape[-1] - crop) // 2
            independent_tree_image = tree_image[:, :, y0 : y0 + crop, x0 : x0 + crop]
        chm_vector = self.chm_encoder(tree_chm_image)
        structure_vector = self.structure_encoder(batch["tree_centered_structure"])
        if self.interaction_encoder is not None and self.interaction_fusion_mode == "shared_naip_common_grid":
            independent_naip_vector, interaction_vector = self.interaction_encoder(
                tree_image,
                batch["tree_chm_interaction_image"],
            )
            features = [
                independent_naip_vector,
                chm_vector,
                structure_vector,
                interaction_vector,
            ]
        else:
            assert self.image_encoder is not None
            features = [
                self.image_encoder(independent_tree_image),
                chm_vector,
                structure_vector,
            ]
        if self.interaction_encoder is not None and self.interaction_fusion_mode != "shared_naip_common_grid":
            interaction_naip = tree_image
            interaction_chm = batch["tree_chm_interaction_image"]
            if self.interaction_fusion_mode == "learned_common_grid":
                features.append(self.interaction_encoder(interaction_naip, interaction_chm))
            else:
                resized_chm = clean_disc.train_base.F.interpolate(
                    interaction_chm,
                    size=interaction_naip.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
                features.append(
                    self.interaction_encoder(
                        clean_disc.train_base.torch.cat([interaction_naip, resized_chm], dim=1)
                    )
                )
        if self.phenology_encoder is not None:
            phenology = batch["sentinel_phenology"]
            if self.phenology_zero_indices:
                phenology = phenology.clone()
                phenology[:, self.phenology_zero_indices] = 0.0
            features.append(self.phenology_encoder(phenology))
        if self.raw_sentinel_encoder is not None:
            features.append(self.raw_sentinel_encoder(batch["sentinel_sequence"], batch["sentinel_sequence_mask"]))
        if self.satellite_embedding_encoder is not None:
            features.append(self.satellite_embedding_encoder(batch["satellite_embedding"]))
        if self.prism_normals_encoder is not None:
            features.append(self.prism_normals_encoder(batch["prism_normals"]))
        encoded = clean_disc.train_base.torch.stack(features, dim=1)
        gate_logits = self.gate(encoded.flatten(1))
        if self.training and self.tree_image_branch_dropout > 0:
            dropped = clean_disc.train_base.torch.rand(encoded.shape[0], device=encoded.device) < self.tree_image_branch_dropout
            if dropped.any():
                encoded = encoded.clone()
                gate_logits = gate_logits.clone()
                encoded[dropped, self.tree_image_modality_index, :] = 0.0
                gate_logits[dropped, self.tree_image_modality_index] = -1.0e4
        gates = clean_disc.train_base.torch.softmax(gate_logits, dim=1)
        fused = (encoded * gates.unsqueeze(-1)).sum(dim=1)
        return self.head(fused), gates


def train_epoch(model, loader, optimizer, device, args, ce_weights):
    model.train()
    total_loss = 0.0
    total_n = 0
    for batch in loader:
        batch = {key: value.to(device, non_blocking=True) if hasattr(value, "to") else value for key, value in batch.items()}
        labels = batch["label"]
        weights = batch["sample_weight"].to(device=device, dtype=clean_disc.train_base.torch.float32)
        optimizer.zero_grad(set_to_none=True)
        logits, _gates = model(batch)
        ce = clean_disc.train_base.F.cross_entropy(logits, labels, weight=ce_weights, reduction="none")
        loss = (ce * weights).sum() / clean_disc.train_base.torch.clamp(weights.sum(), min=1.0e-6)
        loss.backward()
        if args.grad_clip > 0:
            clean_disc.train_base.torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        n = int(labels.numel())
        total_loss += float(loss.detach().cpu()) * n
        total_n += n
    return {"loss": total_loss / max(total_n, 1)}


@clean_disc.train_base.torch.no_grad()
def evaluate(model, loader, device, class_count: int, group_names: list[str], ce_weights=None, collect: bool = False):
    model.eval()
    y_true, y_pred, prob_rows = [], [], []
    gate_sum = None
    gate_n = 0
    total_loss = 0.0
    total_n = 0
    cities, tree_ids, crop_indices = [], [], []
    for batch in loader:
        city_token = batch.pop("city_token")
        batch = {key: value.to(device, non_blocking=True) if hasattr(value, "to") else value for key, value in batch.items()}
        logits, gates = model(batch)
        ce = clean_disc.train_base.F.cross_entropy(logits, batch["label"], weight=ce_weights, reduction="none")
        sample_weights = batch["sample_weight"].to(device=device, dtype=clean_disc.train_base.torch.float32)
        loss = (ce * sample_weights).sum() / clean_disc.train_base.torch.clamp(sample_weights.sum(), min=1.0e-6)
        probs = clean_disc.train_base.torch.softmax(logits, dim=1)
        pred = probs.argmax(dim=1)
        n = int(pred.numel())
        total_loss += float(loss.detach().cpu()) * n
        total_n += n
        y_true.append(batch["label"].detach().cpu().numpy())
        y_pred.append(pred.detach().cpu().numpy())
        gate_values = gates.detach().cpu().numpy()
        gate_sum = gate_values.sum(axis=0) if gate_sum is None else gate_sum + gate_values.sum(axis=0)
        gate_n += gate_values.shape[0]
        if collect:
            prob_rows.append(probs.detach().cpu().numpy())
            cities.extend(list(city_token))
            tree_ids.append(batch["tree_centered_index"].detach().cpu().numpy())
            crop_indices.append(batch["crop_index"].detach().cpu().numpy())
    true = np.concatenate(y_true) if y_true else np.empty(0, dtype=np.int64)
    pred = np.concatenate(y_pred) if y_pred else np.empty(0, dtype=np.int64)
    metrics = clean_disc.f1_metrics(true, pred, class_count)
    metrics["loss"] = total_loss / max(total_n, 1)
    per_class = []
    for cls, label in enumerate(group_names):
        true_mask = true == cls
        pred_mask = pred == cls
        tp = int(np.sum(true_mask & pred_mask))
        fp = int(np.sum(~true_mask & pred_mask))
        fn = int(np.sum(true_mask & ~pred_mask))
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1.0e-12)
        per_class.append(
            {
                "group": label,
                "support": int(true_mask.sum()),
                "predicted": int(pred_mask.sum()),
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
            }
        )
    metrics["per_class"] = per_class
    if gate_sum is not None and gate_n > 0:
        for name, value in zip(model.modality_names, gate_sum / gate_n):
            metrics[f"gate_{name}"] = float(value)
    if collect:
        metrics["probabilities"] = np.concatenate(prob_rows) if prob_rows else np.empty((0, class_count), dtype=np.float32)
        metrics["labels"] = true
        metrics["predictions"] = pred
        metrics["city_token"] = np.asarray(cities, dtype="U64")
        metrics["tree_id"] = np.concatenate(tree_ids) if tree_ids else np.empty(0, dtype=np.int64)
        metrics["crop_index"] = np.concatenate(crop_indices) if crop_indices else np.empty(0, dtype=np.int64)
    return metrics


def main() -> int:
    args = parse_args()
    if clean_disc.train_base.TORCH_IMPORT_ERROR is not None:
        raise SystemExit(f"PyTorch is required. Original error: {clean_disc.train_base.TORCH_IMPORT_ERROR}")
    for label in ("complete_shard_root", "output_dir"):
        resolved = Path(getattr(args, label)).resolve()
        if not str(resolved).lower().startswith(str(CLEAN_ROOT).lower()):
            raise SystemExit(f"{label} must point inside {CLEAN_ROOT}; got {resolved}")

    random.seed(args.seed)
    np.random.seed(args.seed)
    clean_disc.train_base.torch.manual_seed(args.seed)
    if str(args.device).startswith("cuda") and not clean_disc.train_base.torch.cuda.is_available():
        raise RuntimeError(
            f"CUDA was requested with --device {args.device}, but this Python environment "
            "has no CUDA-enabled PyTorch installation. Refusing to fall back silently to CPU."
        )
    device = clean_disc.train_base.torch.device(args.device)

    run_dir = args.output_dir / args.run_name
    if run_dir.exists() and any(run_dir.iterdir()) and not args.force:
        raise SystemExit(f"Run directory exists and is not empty: {run_dir}; pass --force.")
    run_dir.mkdir(parents=True, exist_ok=True)

    label_to_group, group_names = load_partition(partition_path(args))
    ensure_shard_pattern(args)
    manifest, paths = clean_disc.discover_shards(args)
    args.phenology_zero_indices = []
    args.phenology_zero_matched_columns = []
    if args.zero_phenology_feature_regex:
        if not args.use_sentinel_phenology:
            raise SystemExit(
                "--zero-phenology-feature-regex requires --use-sentinel-phenology."
            )
        sample_shard_path = next(iter(paths.values()))
        with np.load(sample_shard_path, allow_pickle=True) as sample_shard:
            phenology_columns = (
                [str(value) for value in sample_shard["sentinel_phenology_columns"]]
                if "sentinel_phenology_columns" in sample_shard.files
                else []
            )
        patterns = [
            re.compile(pattern, flags=re.IGNORECASE)
            for pattern in args.zero_phenology_feature_regex
        ]
        args.phenology_zero_indices = [
            index
            for index, column in enumerate(phenology_columns)
            if any(pattern.search(column) for pattern in patterns)
        ]
        args.phenology_zero_matched_columns = [
            phenology_columns[index] for index in args.phenology_zero_indices
        ]
        if not args.phenology_zero_indices:
            raise SystemExit(
                "--zero-phenology-feature-regex matched no Sentinel phenology columns."
            )
        print(
            "Always-zero standardized phenology inputs: "
            f"features={len(args.phenology_zero_indices)}; "
            f"columns={','.join(args.phenology_zero_matched_columns)}",
            flush=True,
        )
    manifest["fine_label"] = clean_disc.scientific_label_from_name(manifest["scientific_name"], "scientific_binomial_or_genus")
    before = len(manifest)
    manifest = manifest.loc[manifest["fine_label"].isin(label_to_group)].copy()
    dropped = before - len(manifest)
    if dropped:
        print(f"Dropped records not represented in {args.grouping_scheme} partition: {before:,}->{len(manifest):,}", flush=True)
    if not len(manifest):
        raise SystemExit("No records remain after partition label mapping.")
    manifest["label_index"] = manifest["fine_label"].map(label_to_group).astype(np.int64)
    if args.exclude_qa_warning_from_model:
        if "qa_any_warning_flag" not in manifest.columns:
            raise SystemExit("--exclude-qa-warning-from-model requires qa_any_warning_flag in clean shard metadata.")
        before_warning = len(manifest)
        warning = clean_disc.bool_series(manifest["qa_any_warning_flag"])
        manifest = manifest.loc[~warning].copy()
        print(
            f"Dropped QA-warning records before model splits: {before_warning:,}->{len(manifest):,}",
            flush=True,
        )
        if not len(manifest):
            raise SystemExit("No records remain after QA-warning filtering.")
    manifest["sample_weight"] = clean_disc.compute_sample_weights(manifest, args)
    weights = manifest["sample_weight"].to_numpy(dtype=np.float32)
    print(
        f"Clean k=6 classifier data before split: rows={len(manifest):,}; cities={manifest['city_token'].nunique():,}; "
        f"groups={len(group_names)}; sample_weight_mean={weights.mean():.3f}; p05={np.quantile(weights, 0.05):.3f}",
        flush=True,
    )
    pd.DataFrame({"group": group_names, "group_index": range(len(group_names))}).to_csv(run_dir / "group_label_map.csv", index=False)
    group_counts = manifest["label_index"].value_counts().sort_index()
    pd.DataFrame(
        {
            "group": [group_names[i] for i in group_counts.index],
            "records": group_counts.values.astype(int),
            "fraction": group_counts.values / max(len(manifest), 1),
        }
    ).to_csv(run_dir / "group_counts.csv", index=False)

    manifest = clean_disc.split_manifest(manifest, args)
    manifest = clean_disc.cap_eval_splits(manifest, args)
    train_manifest = manifest.loc[manifest["split"].eq("train")].reset_index(drop=True)
    val_manifest = manifest.loc[manifest["split"].eq("val")].reset_index(drop=True)
    test_manifest = manifest.loc[manifest["split"].eq("test")].reset_index(drop=True)
    print(
        f"Clean {args.grouping_scheme} classifier split: train={len(train_manifest):,}; "
        f"val={len(val_manifest):,}; test={len(test_manifest):,}",
        flush=True,
    )

    store = clean_disc.CleanShardStore(paths, max_cached_shards=args.max_shard_cache)
    structure_mean, structure_std = clean_disc.estimate_array_standardizer(train_manifest, store, "structure", 0, args.seed)
    phenology_mean = phenology_std = raw_sentinel_mean = raw_sentinel_std = None
    satellite_embedding_mean = satellite_embedding_std = prism_mean = prism_std = None
    if args.use_sentinel_phenology:
        print("Using Sentinel phenology arrays from clean complete shards.", flush=True)
        phenology_mean, phenology_std = clean_disc.estimate_array_standardizer(train_manifest, store, "sentinel_phenology", args.phenology_stat_samples, args.seed + 11)
    if args.use_raw_sentinel:
        print("Using raw Sentinel sequence arrays from clean complete shards.", flush=True)
        sample_shard_path = next(iter(paths.values()))
        with np.load(sample_shard_path, allow_pickle=True) as sample_shard:
            sentinel_feature_columns = (
                [str(value) for value in sample_shard["sentinel_feature_columns"]]
                if "sentinel_feature_columns" in sample_shard.files
                else []
            )
        daily_prism_columns = [
            column for column in sentinel_feature_columns if column.startswith("prism_daily_")
        ]
        print(
            f"Raw temporal encoder input: features={len(sentinel_feature_columns)}; "
            f"daily_PRISM={','.join(daily_prism_columns) if daily_prism_columns else 'none'}",
            flush=True,
        )
        raw_sentinel_mean, raw_sentinel_std = clean_disc.estimate_raw_sentinel_standardizer(train_manifest, store, args.raw_sentinel_stat_samples, args.seed + 17)
    if args.use_satellite_embedding:
        print("Using GEE/Satlas embedding arrays from clean complete shards.", flush=True)
        satellite_embedding_mean, satellite_embedding_std = clean_disc.estimate_array_standardizer(
            train_manifest, store, "satellite_embedding", args.satellite_embedding_stat_samples, args.seed + 23
        )
    if args.use_prism_normals:
        print("Using PRISM normals arrays from clean complete shards.", flush=True)
        prism_mean, prism_std = clean_disc.estimate_array_standardizer(train_manifest, store, "prism_normals", args.prism_normals_stat_samples, args.seed + 29)

    train_ds = clean_disc.CleanTreeDataset(
        train_manifest, store, structure_mean, structure_std, phenology_mean, phenology_std,
        raw_sentinel_mean, raw_sentinel_std, satellite_embedding_mean, satellite_embedding_std,
        prism_mean, prism_std, args
    )
    val_ds = clean_disc.CleanTreeDataset(
        val_manifest, store, structure_mean, structure_std, phenology_mean, phenology_std,
        raw_sentinel_mean, raw_sentinel_std, satellite_embedding_mean, satellite_embedding_std,
        prism_mean, prism_std, args, preload_in_memory=args.cache_val_in_memory
    )
    test_ds = clean_disc.CleanTreeDataset(
        test_manifest, store, structure_mean, structure_std, phenology_mean, phenology_std,
        raw_sentinel_mean, raw_sentinel_std, satellite_embedding_mean, satellite_embedding_std,
        prism_mean, prism_std, args, preload_in_memory=args.cache_test_in_memory
    )
    if args.num_workers > 0:
        store.clear()
        print("Cleared clean shard cache before DataLoader worker spawn.", flush=True)
    train_loader = clean_disc.make_loader(train_ds, args, train=True)
    val_loader = clean_disc.make_loader(val_ds, args, train=False)
    test_loader = clean_disc.make_loader(test_ds, args, train=False)

    model = CleanK6Classifier(
        class_count=len(group_names),
        structure_dim=int(structure_mean.shape[0]),
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        phenology_dim=0 if phenology_mean is None else int(phenology_mean.shape[0]),
        raw_sentinel_dim=0 if raw_sentinel_mean is None else int(raw_sentinel_mean.shape[0]),
        satellite_embedding_dim=0 if satellite_embedding_mean is None else int(satellite_embedding_mean.shape[0]),
        prism_normals_dim=0 if prism_mean is None else int(prism_mean.shape[0]),
        tree_image_branch_dropout=args.tree_image_branch_dropout,
        use_naip_chm_interaction_branch=args.use_naip_chm_interaction_branch,
        independent_naip_crop_pixels=args.image_center_crop_pixels,
        interaction_naip_crop_pixels=args.interaction_naip_crop_pixels,
        interaction_fusion_mode=args.interaction_fusion_mode,
        interaction_naip_patch_pixels=args.interaction_naip_patch_pixels,
        interaction_chm_patch_pixels=args.interaction_chm_patch_pixels,
        phenology_zero_indices=args.phenology_zero_indices,
    ).to(device)
    if args.use_naip_chm_interaction_branch:
        shared_text = (
            "shared NAIP common-grid map feeds independent and interaction heads"
            if args.interaction_fusion_mode == "shared_naip_common_grid"
            else "independent NAIP and interaction encoders"
        )
        print(
            f"NAIP-CHM interaction architecture: mode={args.interaction_fusion_mode}; "
            f"NAIP crop={args.interaction_naip_crop_pixels or args.image_center_crop_pixels}px; "
            f"NAIP patch={args.interaction_naip_patch_pixels}px; "
            f"CHM patch={args.interaction_chm_patch_pixels}px; {shared_text}.",
            flush=True,
        )
    counts = np.bincount(train_manifest["label_index"].to_numpy(dtype=np.int64), minlength=len(group_names))
    ce_weights = clean_disc.train_base.torch.as_tensor(clean_disc.class_weights(counts, args), dtype=clean_disc.train_base.torch.float32, device=device)
    if args.resume_from_checkpoint is not None:
        if not args.resume_from_checkpoint.exists():
            raise FileNotFoundError(args.resume_from_checkpoint)
        resume_checkpoint = clean_disc.train_base.torch.load(args.resume_from_checkpoint, map_location=device)
        model.load_state_dict(resume_checkpoint["model_state_dict"])
        print(
            f"Resumed model weights from {args.resume_from_checkpoint}; "
            f"checkpoint_epoch={resume_checkpoint.get('epoch', 'unknown')}; "
            f"checkpoint_score={resume_checkpoint.get('selection_score', 'unknown')}",
            flush=True,
        )

    optimizer = clean_disc.train_base.torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}", flush=True)

    best_score = -1.0
    best_macro = -1.0
    best_epoch = 0
    no_improve = 0
    history: list[dict[str, Any]] = []
    saved_top: list[tuple[float, Path]] = []
    best_path = run_dir / "best_model.pt"
    for epoch in range(1, args.epochs + 1):
        train_metrics = train_epoch(model, train_loader, optimizer, device, args, ce_weights)
        val_metrics = evaluate(model, val_loader, device, len(group_names), group_names, ce_weights=ce_weights, collect=False)
        score = selection_score(val_metrics, args)
        payload = checkpoint_payload(
            model,
            group_names,
            label_to_group,
            structure_mean,
            structure_std,
            phenology_mean,
            phenology_std,
            raw_sentinel_mean,
            raw_sentinel_std,
            satellite_embedding_mean,
            satellite_embedding_std,
            prism_mean,
            prism_std,
            args,
            epoch,
            val_metrics,
            score,
        )
        saved_top = save_top_k_checkpoint(payload, run_dir, args.save_top_k_checkpoints, saved_top)
        improved = score > best_score + float(args.early_stopping_min_delta)
        if improved:
            best_score = score
            best_macro = val_metrics["macro_f1"]
            best_epoch = epoch
            no_improve = 0
            clean_disc.train_base.torch.save(payload, best_path)
        else:
            no_improve += 1
        row = {
            "epoch": epoch,
            **{f"train_{k}": v for k, v in train_metrics.items()},
            **{f"val_{k}": v for k, v in val_metrics.items() if k != "per_class"},
            "val_selection_score": score,
            "selection_metric": args.checkpoint_selection_score,
        }
        history.append(row)
        pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)
        gate_text = ",".join(f"{name}:{val_metrics.get(f'gate_{name}', 0.0):.3f}" for name in model.modality_names)
        print(
            f"epoch {epoch:03d}: train_loss={train_metrics['loss']:.4f}; val_loss={val_metrics['loss']:.4f}; "
            f"val_acc={val_metrics['accuracy']:.3f}; val_macro_f1={val_metrics['macro_f1']:.3f}; "
            f"val_weighted_f1={val_metrics['weighted_f1']:.3f}; score={score:.4f}; gates={gate_text}; "
            f"improved={'yes' if improved else 'no'}; no_improve={no_improve}/{args.early_stopping_patience}",
            flush=True,
        )
        if no_improve >= args.early_stopping_patience:
            break

    if best_path.exists():
        checkpoint = clean_disc.train_base.torch.load(best_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
    test_metrics = evaluate(model, test_loader, device, len(group_names), group_names, ce_weights=ce_weights, collect=args.export_predictions)
    pd.DataFrame(test_metrics["per_class"]).to_csv(run_dir / "test_per_group_metrics.csv", index=False)
    summary = {
        "grouping_scheme": args.grouping_scheme,
        "partition_path": str(partition_path(args)),
        "best_epoch": best_epoch,
        "checkpoint_selection_score": args.checkpoint_selection_score,
        "best_val_score": best_score,
        "best_val_macro_f1": best_macro,
        "test_accuracy": test_metrics["accuracy"],
        "test_macro_f1": test_metrics["macro_f1"],
        "test_weighted_f1": test_metrics["weighted_f1"],
        "group_names": group_names,
        "args": clean_disc.safe_json_args(args),
    }
    (run_dir / "run_config.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if args.export_predictions:
        np.savez_compressed(
            run_dir / "test_predictions.npz",
            probabilities=test_metrics["probabilities"],
            labels=test_metrics["labels"],
            predictions=test_metrics["predictions"],
            city_token=test_metrics["city_token"],
            tree_id=test_metrics["tree_id"],
            crop_index=test_metrics["crop_index"],
            group_names=np.asarray(group_names, dtype="<U64"),
        )
    print(
        f"Finished clean {args.grouping_scheme} classifier: best_epoch={best_epoch}; "
        f"test_acc={test_metrics['accuracy']:.3f}; test_macro_f1={test_metrics['macro_f1']:.3f}; "
        f"test_weighted_f1={test_metrics['weighted_f1']:.3f}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
