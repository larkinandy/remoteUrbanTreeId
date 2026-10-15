#!/usr/bin/env python3
"""Train the stage-two classifier for a frozen taxon partition.

This script uses the clean standalone complete shards and maps each record's
scientific label through a fixed species-level partition. The repository
default is the frozen `k11` manuscript mapping including taxa with at least 50
records. Reported `collapsed_k6` results are produced after checkpoint
ensembling; they are not trained here as a separate label space.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
MODEL_DIR = HERE.parent
DISCRIMINATOR_DIR = MODEL_DIR / "discriminator"
for module_dir in (HERE, MODEL_DIR, DISCRIMINATOR_DIR):
    if str(module_dir) not in sys.path:
        sys.path.insert(0, str(module_dir))

import train_taxon_discriminator as clean_disc
import clean_mmap_loader
import artifacts
import checkpointing
import cli as stage2_cli

DEFAULT_MANUSCRIPT_PARTITION = artifacts.FINAL_PARTITION


def parse_args() -> argparse.Namespace:
    """Parse the general fixed-k11 training and experiment options."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    stage2_cli.add_shard_io_arguments(parser)
    parser.add_argument("--run-name", default="fixed_k11_stage2_classifier")
    parser.add_argument("--city-token", action="append", default=None)
    parser.add_argument("--exclude-city-token", action="append", default=[])
    parser.add_argument(
        "--leave-one-city-out-test-city",
        default=None,
        help=(
            "Cross-city sensitivity mode used with --split-manifest. Training and "
            "validation retain only records outside this city, while testing retains "
            "only this city's inherited test records. The focal city's inherited "
            "train/validation records and other cities' test records are excluded."
        ),
    )
    parser.add_argument("--max-records-per-city", type=int, default=0)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--val-fraction", type=float, default=0.10)
    parser.add_argument("--test-fraction", type=float, default=0.10)
    parser.add_argument("--max-val-samples", type=int, default=150000)
    parser.add_argument("--max-test-samples", type=int, default=150000)
    stage2_cli.add_split_manifest_argument(parser, required=False)
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
        "--initialize-from-checkpoint",
        type=Path,
        default=None,
        help=(
            "Warm-start matching parameters from a checkpoint while creating a fresh "
            "optimizer and training history. Unlike --resume-from-checkpoint, unmatched "
            "keys and shapes are skipped."
        ),
    )
    parser.add_argument(
        "--initialize-chm-from-checkpoint",
        type=Path,
        default=None,
        help="Override chm_encoder parameters from a CHM-only checkpoint.",
    )
    parser.add_argument(
        "--initialize-interaction-from-checkpoint",
        type=Path,
        default=None,
        help="Override interaction_encoder parameters from an interaction-only checkpoint.",
    )
    parser.add_argument(
        "--reinitialize-gate-on-initialize",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Keep the newly initialized modality gate instead of loading gate parameters.",
    )
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
    parser.add_argument(
        "--prepared-mmap-root",
        type=Path,
        default=None,
        help=(
            "Optional root containing prepared raster and vector NPY sidecars. "
            "When supplied, training uses shared memory maps instead of loading NPZ shards."
        ),
    )
    parser.add_argument("--cache-val-in-memory", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--cache-test-in-memory", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--image-center-crop-pixels", type=int, default=32)
    parser.add_argument(
        "--image-encoder-normalization",
        choices=("batch", "group"),
        default="batch",
        help=(
            "Normalization used by the standard NAIP, CHM, and bilinear interaction "
            "image encoders. Batch preserves the published-model architecture; group "
            "avoids running-statistic dependence in shard-local single-modality screens."
        ),
    )
    parser.add_argument(
        "--naip-encoder-normalization",
        choices=("batch", "group"),
        default=None,
        help=(
            "Optional normalization override for only the independent raw-NAIP "
            "encoder. When omitted, --image-encoder-normalization is used."
        ),
    )
    parser.add_argument(
        "--naip-encoder-architecture",
        choices=("compact_cnn", "maestro"),
        default="compact_cnn",
        help=(
            "Independent NAIP encoder. compact_cnn is the existing five-channel "
            "CNN (RGB, NIR, NDVI); maestro uses the pretrained four-band MAESTRO "
            "S2-NAIP urban encoder (NIR, red, green, blue)."
        ),
    )
    parser.add_argument("--maestro-model-id", default="BiliSakura/MAESTRO-transformers")
    parser.add_argument("--maestro-subfolder", default="maestro-s2-naip-urban-base")
    parser.add_argument(
        "--maestro-revision",
        default="243b1bc253b72980f1bf6f70354c1f84d958105c",
        help="Pinned revision of the Transformers-compatible MAESTRO conversion.",
    )
    parser.add_argument(
        "--maestro-freeze-backbone",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Freeze MAESTRO and train only its projection and classifier (recommended screen).",
    )
    parser.add_argument(
        "--maestro-local-files-only",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Require MAESTRO files to already exist in the Hugging Face cache.",
    )
    parser.add_argument(
        "--chm-encoder-normalization",
        choices=("batch", "group"),
        default=None,
        help=(
            "Optional normalization override for only the independent CHM encoder. "
            "When omitted, --image-encoder-normalization is used."
        ),
    )
    parser.add_argument(
        "--interaction-encoder-normalization",
        choices=("batch", "group"),
        default="batch",
        help="Normalization used inside the NAIP-CHM interaction encoder.",
    )
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
    parser.add_argument(
        "--exclude-modality",
        action="append",
        choices=(
            "tree_centered_naip", "tree_centered_chm", "tree_centered_structure",
            "naip_chm_interaction", "sentinel_phenology", "raw_sentinel",
            "satellite_embedding", "prism_normals",
        ),
        default=[],
        help=(
            "Remove one modality branch from the architecture before training. Repeat to "
            "remove multiple branches. Intended for fully retrained modality ablations."
        ),
    )
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
    parser.add_argument(
        "--export-validation-predictions",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Export validation probabilities and identifiers from the selected checkpoint.",
    )
    parser.add_argument(
        "--validation-only-pilot",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Stop after training and validation; do not run final test inference.",
    )
    stage2_cli.add_overwrite_argument(parser)
    return parser.parse_args()


def partition_path(args: argparse.Namespace) -> Path:
    """Return the sole supported frozen manuscript partition."""
    return DEFAULT_MANUSCRIPT_PARTITION


def load_partition(path: Path) -> tuple[dict[str, int], list[str]]:
    """Load a non-object taxon partition and normalize group indices."""
    if not path.exists():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as data:
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
        normalization: str = "batch",
    ) -> None:
        """Create independently learned NAIP and CHM common-grid projections."""
        super().__init__()
        if naip_patch_pixels <= 0 or chm_patch_pixels <= 0:
            raise ValueError("Interaction patch dimensions must be positive.")
        self.naip_patch_pixels = int(naip_patch_pixels)
        self.chm_patch_pixels = int(chm_patch_pixels)
        branch_dim = 32

        def norm(channels: int) -> Any:
            """Construct the shared spatial normalization layer."""
            return clean_disc.train_base.make_spatial_normalization(channels, normalization)

        self.naip_to_common_grid = clean_disc.train_base.nn.Sequential(
            clean_disc.train_base.nn.Conv2d(
                5,
                branch_dim,
                kernel_size=self.naip_patch_pixels,
                stride=self.naip_patch_pixels,
            ),
            norm(branch_dim),
            clean_disc.train_base.nn.GELU(),
        )
        self.chm_to_common_grid = clean_disc.train_base.nn.Sequential(
            clean_disc.train_base.nn.Conv2d(
                3,
                branch_dim,
                kernel_size=self.chm_patch_pixels,
                stride=self.chm_patch_pixels,
            ),
            norm(branch_dim),
            clean_disc.train_base.nn.GELU(),
        )
        self.spatial_interaction = clean_disc.train_base.nn.Sequential(
            clean_disc.train_base.nn.Conv2d(branch_dim * 2, 64, 3, padding=1),
            norm(64),
            clean_disc.train_base.nn.GELU(),
            clean_disc.train_base.nn.Conv2d(64, hidden_dim, 3, padding=1),
            norm(hidden_dim),
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
        """Encode aligned NAIP and CHM tensors into one interaction vector."""
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
        normalization: str = "batch",
    ) -> None:
        """Create a shared NAIP map with independent and interaction heads."""
        super().__init__()
        if naip_patch_pixels <= 0 or chm_patch_pixels <= 0:
            raise ValueError("Interaction patch dimensions must be positive.")
        self.naip_patch_pixels = int(naip_patch_pixels)
        self.chm_patch_pixels = int(chm_patch_pixels)
        branch_dim = 32

        def norm(channels: int) -> Any:
            """Construct the shared spatial normalization layer."""
            return clean_disc.train_base.make_spatial_normalization(channels, normalization)

        self.naip_to_common_grid = clean_disc.train_base.nn.Sequential(
            clean_disc.train_base.nn.Conv2d(
                5,
                branch_dim,
                kernel_size=self.naip_patch_pixels,
                stride=self.naip_patch_pixels,
            ),
            norm(branch_dim),
            clean_disc.train_base.nn.GELU(),
        )
        self.independent_naip_head = clean_disc.train_base.nn.Sequential(
            clean_disc.train_base.nn.Conv2d(branch_dim, 64, 3, padding=1),
            norm(64),
            clean_disc.train_base.nn.GELU(),
            clean_disc.train_base.nn.Conv2d(64, hidden_dim, 3, padding=1),
            norm(hidden_dim),
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
            norm(branch_dim),
            clean_disc.train_base.nn.GELU(),
        )
        self.spatial_interaction = clean_disc.train_base.nn.Sequential(
            clean_disc.train_base.nn.Conv2d(branch_dim * 2, 64, 3, padding=1),
            norm(64),
            clean_disc.train_base.nn.GELU(),
            clean_disc.train_base.nn.Conv2d(64, hidden_dim, 3, padding=1),
            norm(hidden_dim),
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
        """Return independent NAIP and joint NAIP–CHM representations."""
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


class MaestroNaipEncoder(clean_disc.train_base.nn.Module):
    """Adapt the pretrained MAESTRO aerial encoder to this model's hidden width."""

    def __init__(
        self,
        hidden_dim: int,
        dropout: float,
        model_id: str,
        subfolder: str,
        revision: str,
        freeze_backbone: bool,
        local_files_only: bool,
    ) -> None:
        """Load a pinned MAESTRO backbone and add the stage-two projection."""
        super().__init__()
        try:
            from transformers import AutoModel
            from huggingface_hub import snapshot_download
        except ImportError as exc:
            raise ImportError(
                "MAESTRO requires transformers. Install it in the training environment "
                "before using --naip-encoder-architecture maestro."
            ) from exc
        model_path = Path(model_id)
        if model_path.exists():
            local_model_dir = model_path / subfolder if subfolder else model_path
        else:
            # Transformers resolves auto_map remote-code files at a Hub
            # repository's root even when config.json is in a subfolder. A
            # pinned snapshot makes the checkpoint subfolder a conventional
            # local model directory and keeps configuration_maestro.py and
            # modeling_maestro.py beside config.json.
            snapshot_root = Path(
                snapshot_download(
                    repo_id=model_id,
                    revision=revision,
                    allow_patterns=[f"{subfolder}/*"] if subfolder else None,
                    local_files_only=local_files_only,
                )
            )
            local_model_dir = snapshot_root / subfolder if subfolder else snapshot_root
        self.backbone = AutoModel.from_pretrained(
            local_model_dir,
            trust_remote_code=True,
            local_files_only=True,
        )
        self.freeze_backbone = bool(freeze_backbone)
        if self.freeze_backbone:
            self.backbone.requires_grad_(False)
            self.backbone.eval()
        output_dim = int(getattr(self.backbone.config, "embed_dim", 768))
        self.projection = clean_disc.train_base.nn.Sequential(
            clean_disc.train_base.nn.LayerNorm(output_dim),
            clean_disc.train_base.nn.Linear(output_dim, hidden_dim),
            clean_disc.train_base.nn.GELU(),
            clean_disc.train_base.nn.Dropout(dropout),
        )

    def train(self, mode: bool = True) -> Any:
        """Preserve evaluation mode for a frozen pretrained backbone."""
        super().train(mode)
        if self.freeze_backbone:
            self.backbone.eval()
        return self

    def forward(
        self,
        image: clean_disc.train_base.torch.Tensor,
    ) -> clean_disc.train_base.torch.Tensor:
        """Convert stored NAIP channel order and encode one image batch."""
        torch = clean_disc.train_base.torch
        # Stored tree_image is RGB, NIR, NDVI in [0, 1]. MAESTRO expects
        # four reflectance channels ordered NIR, red, green, blue.
        aerial = image[:, [3, 0, 1, 2]].unsqueeze(1)
        batch_size = aerial.shape[0]
        dates = torch.tensor(
            [2020.0, 180.0, 12.0], dtype=aerial.dtype, device=aerial.device
        ).view(1, 1, 3).expand(batch_size, 1, 3)
        # A fixed neutral date prevents acquisition-date information from
        # entering this image-encoder comparison; relative time is zero.
        context = torch.no_grad() if self.freeze_backbone else torch.enable_grad()
        with context:
            output = self.backbone(aerial=aerial, aerial_dates=dates, ref_date=dates)
            pooled = output.pooler_output
        return self.projection(pooled)


class FixedPartitionClassifier(clean_disc.train_base.nn.Module):
    """Encode modalities, fuse them with learned gates, and classify k11."""

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
        image_encoder_normalization: str = "batch",
        naip_encoder_normalization: str | None = None,
        naip_encoder_architecture: str = "compact_cnn",
        maestro_model_id: str = "BiliSakura/MAESTRO-transformers",
        maestro_subfolder: str = "maestro-s2-naip-urban-base",
        maestro_revision: str = "243b1bc253b72980f1bf6f70354c1f84d958105c",
        maestro_freeze_backbone: bool = True,
        maestro_local_files_only: bool = False,
        chm_encoder_normalization: str | None = None,
        interaction_encoder_normalization: str = "batch",
        phenology_zero_indices: list[int] | tuple[int, ...] | None = None,
        excluded_modalities: list[str] | tuple[str, ...] | None = None,
    ) -> None:
        """Build only the modality branches retained by the configuration."""
        super().__init__()
        self.tree_image_branch_dropout = float(tree_image_branch_dropout)
        self.tree_image_modality_index = 0
        self.use_naip_chm_interaction_branch = bool(use_naip_chm_interaction_branch)
        self.independent_naip_crop_pixels = max(0, int(independent_naip_crop_pixels))
        self.interaction_naip_crop_pixels = max(0, int(interaction_naip_crop_pixels))
        self.interaction_fusion_mode = str(interaction_fusion_mode)
        self.phenology_zero_indices = tuple(int(value) for value in (phenology_zero_indices or ()))
        self.excluded_modalities = frozenset(excluded_modalities or ())
        naip_normalization = naip_encoder_normalization or image_encoder_normalization
        chm_normalization = chm_encoder_normalization or image_encoder_normalization
        if self.interaction_fusion_mode == "shared_naip_common_grid" and "tree_centered_naip" in self.excluded_modalities:
            raise ValueError("Cannot independently exclude NAIP in shared_naip_common_grid mode")
        names = []
        for name, available in (
            ("tree_centered_naip", True), ("tree_centered_chm", True),
            ("tree_centered_structure", True),
            ("naip_chm_interaction", self.use_naip_chm_interaction_branch),
            ("sentinel_phenology", phenology_dim > 0), ("raw_sentinel", raw_sentinel_dim > 0),
            ("satellite_embedding", satellite_embedding_dim > 0), ("prism_normals", prism_normals_dim > 0),
        ):
            if available and name not in self.excluded_modalities:
                names.append(name)
        if not names:
            raise ValueError("At least one modality must remain")
        self.modality_names = tuple(names)

        if self.use_naip_chm_interaction_branch and self.interaction_fusion_mode == "shared_naip_common_grid":
            if naip_encoder_architecture == "maestro":
                raise ValueError("MAESTRO is incompatible with shared_naip_common_grid interaction mode")
            self.image_encoder = None
        elif naip_encoder_architecture == "maestro":
            self.image_encoder = MaestroNaipEncoder(
                hidden_dim=hidden_dim,
                dropout=dropout,
                model_id=maestro_model_id,
                subfolder=maestro_subfolder,
                revision=maestro_revision,
                freeze_backbone=maestro_freeze_backbone,
                local_files_only=maestro_local_files_only,
            )
        else:
            self.image_encoder = clean_disc.train_base.ImageEncoder(
                5, hidden_dim, dropout, normalization=naip_normalization
            )
        self.chm_encoder = (
            None
            if "tree_centered_chm" in self.excluded_modalities
            else clean_disc.train_base.ImageEncoder(
                3, hidden_dim, dropout, normalization=chm_normalization
            )
        )
        if self.use_naip_chm_interaction_branch and self.interaction_fusion_mode == "shared_naip_common_grid":
            self.interaction_encoder = SharedNaipCommonGridEncoder(
                hidden_dim=hidden_dim,
                dropout=dropout,
                naip_patch_pixels=interaction_naip_patch_pixels,
                chm_patch_pixels=interaction_chm_patch_pixels,
                normalization=interaction_encoder_normalization,
            )
        elif self.use_naip_chm_interaction_branch and self.interaction_fusion_mode == "learned_common_grid":
            self.interaction_encoder = LearnedCommonGridInteractionEncoder(
                hidden_dim=hidden_dim,
                dropout=dropout,
                naip_patch_pixels=interaction_naip_patch_pixels,
                chm_patch_pixels=interaction_chm_patch_pixels,
                normalization=interaction_encoder_normalization,
            )
        elif self.use_naip_chm_interaction_branch and self.interaction_fusion_mode == "bilinear_upsample":
            self.interaction_encoder = clean_disc.train_base.ImageEncoder(
                8, hidden_dim, dropout, normalization=interaction_encoder_normalization
            )
        elif self.use_naip_chm_interaction_branch:
            raise ValueError(f"Unknown interaction_fusion_mode={self.interaction_fusion_mode!r}")
        else:
            self.interaction_encoder = None
        self.structure_encoder = None if "tree_centered_structure" in self.excluded_modalities else clean_disc.PhenologyEncoder(structure_dim, hidden_dim, dropout)
        self.phenology_encoder = clean_disc.PhenologyEncoder(phenology_dim, hidden_dim, dropout) if phenology_dim > 0 and "sentinel_phenology" not in self.excluded_modalities else None
        self.raw_sentinel_encoder = clean_disc.train_base.MaskedSequenceEncoder(raw_sentinel_dim, hidden_dim, dropout) if raw_sentinel_dim > 0 and "raw_sentinel" not in self.excluded_modalities else None
        self.satellite_embedding_encoder = clean_disc.PhenologyEncoder(satellite_embedding_dim, hidden_dim, dropout) if satellite_embedding_dim > 0 and "satellite_embedding" not in self.excluded_modalities else None
        self.prism_normals_encoder = clean_disc.PhenologyEncoder(prism_normals_dim, hidden_dim, dropout) if prism_normals_dim > 0 and "prism_normals" not in self.excluded_modalities else None
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

    def encode(
        self,
        batch: dict[str, clean_disc.train_base.torch.Tensor],
    ) -> tuple[clean_disc.train_base.torch.Tensor, clean_disc.train_base.torch.Tensor]:
        """Return the gated fused representation and modality weights."""
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
        chm_vector = self.chm_encoder(tree_chm_image) if self.chm_encoder is not None else None
        structure_vector = self.structure_encoder(batch["tree_centered_structure"]) if self.structure_encoder is not None else None
        if self.interaction_encoder is not None and self.interaction_fusion_mode == "shared_naip_common_grid":
            independent_naip_vector, interaction_vector = self.interaction_encoder(
                tree_image,
                batch["tree_chm_interaction_image"],
            )
            vectors = {"tree_centered_naip": independent_naip_vector, "tree_centered_chm": chm_vector,
                       "tree_centered_structure": structure_vector, "naip_chm_interaction": interaction_vector}
            features = [vectors[name] for name in self.modality_names if name in vectors]
        else:
            vectors = {"tree_centered_chm": chm_vector, "tree_centered_structure": structure_vector}
            if "tree_centered_naip" in self.modality_names:
                assert self.image_encoder is not None
                vectors["tree_centered_naip"] = self.image_encoder(independent_tree_image)
            features = [vectors[name] for name in self.modality_names if name in vectors]
        if self.interaction_encoder is not None and self.interaction_fusion_mode != "shared_naip_common_grid":
            if "naip_chm_interaction" not in self.excluded_modalities:
                interaction_naip = tree_image
                interaction_chm = batch["tree_chm_interaction_image"]
                if self.interaction_fusion_mode == "learned_common_grid":
                    features.append(self.interaction_encoder(interaction_naip, interaction_chm))
                else:
                    resized_chm = clean_disc.train_base.F.interpolate(
                        interaction_chm, size=interaction_naip.shape[-2:], mode="bilinear", align_corners=False,
                    )
                    features.append(self.interaction_encoder(clean_disc.train_base.torch.cat([interaction_naip, resized_chm], dim=1)))
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
        return fused, gates

    def forward(self, batch: dict[str, clean_disc.train_base.torch.Tensor]):
        """Return class logits and sample-specific modality gates."""
        fused, gates = self.encode(batch)
        return self.head(fused), gates


def build_classifier(
    args: argparse.Namespace,
    class_count: int,
    dimensions: dict[str, int],
    device: Any,
    *,
    inference: bool = False,
) -> FixedPartitionClassifier:
    """Construct the configured classifier from compact dimension metadata."""
    model = FixedPartitionClassifier(
        class_count=class_count,
        structure_dim=dimensions["structure"],
        hidden_dim=int(getattr(args, "hidden_dim", 128)),
        dropout=float(getattr(args, "dropout", 0.2)),
        phenology_dim=dimensions["phenology"],
        raw_sentinel_dim=dimensions["raw_sentinel"],
        satellite_embedding_dim=dimensions["satellite_embedding"],
        prism_normals_dim=dimensions["prism_normals"],
        tree_image_branch_dropout=0.0 if inference else float(getattr(args, "tree_image_branch_dropout", 0.0)),
        use_naip_chm_interaction_branch=bool(getattr(args, "use_naip_chm_interaction_branch", False)),
        independent_naip_crop_pixels=int(getattr(args, "image_center_crop_pixels", 0)),
        interaction_naip_crop_pixels=int(getattr(args, "interaction_naip_crop_pixels", 0)),
        interaction_fusion_mode=str(getattr(args, "interaction_fusion_mode", "bilinear_upsample")),
        interaction_naip_patch_pixels=int(getattr(args, "interaction_naip_patch_pixels", 5)),
        interaction_chm_patch_pixels=int(getattr(args, "interaction_chm_patch_pixels", 3)),
        image_encoder_normalization=str(getattr(args, "image_encoder_normalization", "batch")),
        naip_encoder_normalization=getattr(args, "naip_encoder_normalization", None),
        naip_encoder_architecture=str(getattr(args, "naip_encoder_architecture", "compact_cnn")),
        maestro_model_id=str(getattr(args, "maestro_model_id", "BiliSakura/MAESTRO-transformers")),
        maestro_subfolder=str(getattr(args, "maestro_subfolder", "maestro-s2-naip-urban-base")),
        maestro_revision=str(getattr(args, "maestro_revision", "243b1bc253b72980f1bf6f70354c1f84d958105c")),
        maestro_freeze_backbone=bool(getattr(args, "maestro_freeze_backbone", True)),
        maestro_local_files_only=bool(getattr(args, "maestro_local_files_only", False)),
        chm_encoder_normalization=getattr(args, "chm_encoder_normalization", None),
        interaction_encoder_normalization=str(getattr(args, "interaction_encoder_normalization", "batch")),
        phenology_zero_indices=list(getattr(args, "phenology_zero_indices", [])),
        excluded_modalities=list(getattr(args, "exclude_modality", [])),
    )
    return model.to(device)


def train_epoch(
    model: Any,
    loader: Any,
    optimizer: Any,
    device: Any,
    args: argparse.Namespace,
    ce_weights: Any,
) -> dict[str, float]:
    """Train the classifier for one epoch and return its weighted mean loss."""
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
def evaluate(
    model: Any,
    loader: Any,
    device: Any,
    class_count: int,
    group_names: list[str],
    ce_weights: Any = None,
    collect: bool = False,
) -> dict[str, Any]:
    """Evaluate one checkpoint and optionally retain record-level predictions."""
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
    """Train, validate, checkpoint, and independently evaluate the fixed-k11 model."""
    args = parse_args()
    artifacts.load_grouping_manifest()
    if clean_disc.train_base.TORCH_IMPORT_ERROR is not None:
        raise SystemExit(f"PyTorch is required. Original error: {clean_disc.train_base.TORCH_IMPORT_ERROR}")
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
    if run_dir.exists() and any(run_dir.iterdir()) and not args.overwrite:
        raise SystemExit(f"Run directory exists and is not empty: {run_dir}; pass --overwrite.")
    run_dir.mkdir(parents=True, exist_ok=True)

    label_to_group, group_names = load_partition(partition_path(args))
    ensure_shard_pattern(args)
    manifest, paths = clean_disc.discover_shards(args)
    first_shard_path = next(iter(paths.values()))
    args.input_schema_sha256 = artifacts.npz_schema_sha256(first_shard_path)
    args.input_feature_schema_sha256 = artifacts.npz_feature_schema_sha256(first_shard_path)
    args.phenology_zero_indices = []
    args.phenology_zero_matched_columns = []
    if args.zero_phenology_feature_regex:
        if not args.use_sentinel_phenology:
            raise SystemExit(
                "--zero-phenology-feature-regex requires --use-sentinel-phenology."
            )
        sample_shard_path = first_shard_path
        with np.load(sample_shard_path, allow_pickle=False) as sample_shard:
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
        print(f"Dropped records not represented in the fixed partition: {before:,}->{len(manifest):,}", flush=True)
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
        f"Stage-two k11 classifier data before split: rows={len(manifest):,}; cities={manifest['city_token'].nunique():,}; "
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

    if args.split_manifest is not None:
        if args.exclude_qa_from_model and "qa_exclude_from_model" in manifest.columns:
            qa_excluded = clean_disc.bool_series(manifest["qa_exclude_from_model"])
            removed = int(qa_excluded.sum())
            manifest = manifest.loc[~qa_excluded].copy()
            print(f"Dropped QA-excluded records before inherited splits: {removed:,}", flush=True)
        split_frame = pd.read_csv(args.split_manifest, usecols=["record_key", "split"])
        if split_frame["record_key"].duplicated().any():
            raise SystemExit(f"Duplicate record_key values in --split-manifest {args.split_manifest}")
        tree_id = pd.to_numeric(manifest["tree_id"], errors="raise").astype("int64").astype(str)
        crop_index = pd.to_numeric(manifest["crop_index"], errors="raise").astype("int64").astype(str)
        manifest["record_key"] = manifest["city_token"].astype(str) + "|" + tree_id + "|" + crop_index
        manifest = manifest.merge(split_frame, on="record_key", how="left", validate="one_to_one")
        missing_split = int(manifest["split"].isna().sum())
        if missing_split:
            raise SystemExit(
                f"--split-manifest is missing {missing_split:,} mapped model records; "
                "refusing to create implicit assignments."
            )
        invalid = sorted(set(manifest["split"].astype(str)).difference({"train", "val", "test"}))
        if invalid:
            raise SystemExit(f"Invalid split values in --split-manifest: {invalid}")
        print(f"Using inherited split assignments from {args.split_manifest}", flush=True)
    else:
        manifest = clean_disc.split_manifest(manifest, args)
        manifest = clean_disc.cap_eval_splits(manifest, args)
    train_manifest = manifest.loc[manifest["split"].eq("train")].reset_index(drop=True)
    val_manifest = manifest.loc[manifest["split"].eq("val")].reset_index(drop=True)
    test_manifest = manifest.loc[manifest["split"].eq("test")].reset_index(drop=True)
    if args.leave_one_city_out_test_city is not None:
        if args.split_manifest is None:
            raise SystemExit("--leave-one-city-out-test-city requires --split-manifest.")
        focal_city = str(args.leave_one_city_out_test_city).strip().lower()
        available_cities = set(manifest["city_token"].astype(str).str.lower())
        if focal_city not in available_cities:
            raise SystemExit(
                f"Focal city {focal_city!r} is absent from the mapped model records."
            )
        train_manifest = train_manifest.loc[
            train_manifest["city_token"].astype(str).str.lower().ne(focal_city)
        ].reset_index(drop=True)
        val_manifest = val_manifest.loc[
            val_manifest["city_token"].astype(str).str.lower().ne(focal_city)
        ].reset_index(drop=True)
        test_manifest = test_manifest.loc[
            test_manifest["city_token"].astype(str).str.lower().eq(focal_city)
        ].reset_index(drop=True)
        if not len(train_manifest) or not len(val_manifest) or not len(test_manifest):
            raise SystemExit(
                "Leave-one-city-out split produced an empty train, validation, or test set: "
                f"city={focal_city}; train={len(train_manifest):,}; "
                f"val={len(val_manifest):,}; test={len(test_manifest):,}."
            )
        print(
            "Leave-one-city-out split enforced: "
            f"focal_test_city={focal_city}; nonfocal_train={len(train_manifest):,}; "
            f"nonfocal_val={len(val_manifest):,}; focal_inherited_test={len(test_manifest):,}",
            flush=True,
        )
    print(
        f"Fixed-partition classifier split: train={len(train_manifest):,}; "
        f"val={len(val_manifest):,}; test={len(test_manifest):,}",
        flush=True,
    )

    if args.prepared_mmap_root is None:
        store = clean_disc.CleanShardStore(paths, max_cached_shards=args.max_shard_cache)
        dataset_class = clean_disc.CleanTreeDataset
        print("Input loader: legacy NPZ shard cache.", flush=True)
    else:
        store = clean_mmap_loader.MmapShardStore(paths, args.prepared_mmap_root)
        dataset_class = clean_mmap_loader.MmapTreeDataset
        print("Input loader: prepared standalone NPY memory maps.", flush=True)
    structure_mean, structure_std = clean_disc.estimate_array_standardizer(train_manifest, store, "structure", 0, args.seed)
    phenology_mean = phenology_std = raw_sentinel_mean = raw_sentinel_std = None
    satellite_embedding_mean = satellite_embedding_std = prism_mean = prism_std = None
    if args.use_sentinel_phenology:
        print("Using Sentinel phenology arrays from clean complete shards.", flush=True)
        phenology_mean, phenology_std = clean_disc.estimate_array_standardizer(train_manifest, store, "sentinel_phenology", args.phenology_stat_samples, args.seed + 11)
    if args.use_raw_sentinel:
        print("Using raw Sentinel sequence arrays from clean complete shards.", flush=True)
        sample_shard_path = first_shard_path
        with np.load(sample_shard_path, allow_pickle=False) as sample_shard:
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

    train_ds = dataset_class(
        train_manifest, store, structure_mean, structure_std, phenology_mean, phenology_std,
        raw_sentinel_mean, raw_sentinel_std, satellite_embedding_mean, satellite_embedding_std,
        prism_mean, prism_std, args
    )
    val_ds = dataset_class(
        val_manifest, store, structure_mean, structure_std, phenology_mean, phenology_std,
        raw_sentinel_mean, raw_sentinel_std, satellite_embedding_mean, satellite_embedding_std,
        prism_mean, prism_std, args, preload_in_memory=args.cache_val_in_memory
    )
    test_ds = dataset_class(
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

    dimensions = {
        "structure": int(structure_mean.shape[0]),
        "phenology": 0 if phenology_mean is None else int(phenology_mean.shape[0]),
        "raw_sentinel": 0 if raw_sentinel_mean is None else int(raw_sentinel_mean.shape[0]),
        "satellite_embedding": 0 if satellite_embedding_mean is None else int(satellite_embedding_mean.shape[0]),
        "prism_normals": 0 if prism_mean is None else int(prism_mean.shape[0]),
    }
    model = build_classifier(args, len(group_names), dimensions, device)
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
    if args.resume_from_checkpoint is not None and args.initialize_from_checkpoint is not None:
        raise SystemExit("Use either --resume-from-checkpoint or --initialize-from-checkpoint, not both.")
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
    elif args.initialize_from_checkpoint is not None:
        initialization_paths = {
            "base": args.initialize_from_checkpoint,
            "chm_encoder": args.initialize_chm_from_checkpoint,
            "interaction_encoder": args.initialize_interaction_from_checkpoint,
        }
        for label, path in initialization_paths.items():
            if path is not None and not path.exists():
                raise FileNotFoundError(f"{label} initialization checkpoint: {path}")

        target_state = model.state_dict()
        merged_state = {key: value for key, value in target_state.items()}

        def merge_checkpoint(path: Path, label: str, allowed_prefix: str | None = None) -> None:
            """Copy shape-compatible parameters from one initialization checkpoint."""
            checkpoint = clean_disc.train_base.torch.load(path, map_location=device)
            source_state = checkpoint["model_state_dict"]
            loaded = 0
            skipped_shape = 0
            skipped_gate = 0
            for key, value in source_state.items():
                if allowed_prefix is not None and not key.startswith(allowed_prefix + "."):
                    continue
                if allowed_prefix is None and key.startswith(("chm_encoder.", "interaction_encoder.")):
                    continue
                if allowed_prefix is None and args.reinitialize_gate_on_initialize and key.startswith("gate."):
                    skipped_gate += 1
                    continue
                if key not in target_state or tuple(value.shape) != tuple(target_state[key].shape):
                    skipped_shape += 1
                    continue
                merged_state[key] = value.to(device=target_state[key].device, dtype=target_state[key].dtype)
                loaded += 1
            print(
                f"Selective initialization from {label}: path={path}; loaded={loaded}; "
                f"shape_or_key_skips={skipped_shape}; gate_skips={skipped_gate}",
                flush=True,
            )

        merge_checkpoint(args.initialize_from_checkpoint, "base")
        if args.initialize_chm_from_checkpoint is not None:
            merge_checkpoint(args.initialize_chm_from_checkpoint, "CHM branch", "chm_encoder")
        if args.initialize_interaction_from_checkpoint is not None:
            merge_checkpoint(
                args.initialize_interaction_from_checkpoint,
                "interaction branch",
                "interaction_encoder",
            )
        model.load_state_dict(merged_state)
        print("Warm-start complete; optimizer and early-stopping history start fresh.", flush=True)

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
        score = checkpointing.selection_score(val_metrics, args)
        standardizers = {
            "structure_mean": structure_mean,
            "structure_std": structure_std,
            "phenology_mean": phenology_mean,
            "phenology_std": phenology_std,
            "raw_sentinel_mean": raw_sentinel_mean,
            "raw_sentinel_std": raw_sentinel_std,
            "satellite_embedding_mean": satellite_embedding_mean,
            "satellite_embedding_std": satellite_embedding_std,
            "prism_mean": prism_mean,
            "prism_std": prism_std,
        }
        payload = checkpointing.checkpoint_payload(
            model,
            group_names,
            label_to_group,
            standardizers,
            args,
            epoch,
            val_metrics,
            score,
            clean_disc.train_base.torch,
        )
        saved_top = checkpointing.save_top_k_checkpoint(
            payload,
            run_dir,
            args.save_top_k_checkpoints,
            saved_top,
            clean_disc.train_base.torch,
        )
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
    if args.export_validation_predictions:
        validation_metrics = evaluate(
            model,
            val_loader,
            device,
            len(group_names),
            group_names,
            ce_weights=ce_weights,
            collect=True,
        )
        np.savez_compressed(
            run_dir / "validation_predictions.npz",
            probabilities=validation_metrics["probabilities"],
            labels=validation_metrics["labels"],
            predictions=validation_metrics["predictions"],
            city_token=validation_metrics["city_token"],
            tree_id=validation_metrics["tree_id"],
            crop_index=validation_metrics["crop_index"],
            group_names=np.asarray(group_names, dtype="<U64"),
            modality_names=np.asarray(model.modality_names, dtype="<U64"),
        )
        print(f"Saved validation predictions: {run_dir / 'validation_predictions.npz'}", flush=True)
    if args.validation_only_pilot:
        summary = {
            "label_space": "fixed_k11",
            "partition_path": str(partition_path(args)),
            "validation_only_pilot": True,
            "test_evaluation_performed": False,
            "best_epoch": best_epoch,
            "checkpoint_selection_score": args.checkpoint_selection_score,
            "best_val_score": best_score,
            "best_val_macro_f1": best_macro,
            "group_names": group_names,
            "provenance": artifacts.provenance(
                args.split_manifest,
                args.input_schema_sha256,
                args.input_feature_schema_sha256,
            ),
            "args": clean_disc.safe_json_args(args),
        }
        (run_dir / "run_config.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(
            f"Finished validation-only pilot: best_epoch={best_epoch}; "
            f"best_val_score={best_score:.4f}; best_val_macro_f1={best_macro:.3f}; "
            "test_evaluation=skipped",
            flush=True,
        )
        return 0
    test_metrics = evaluate(model, test_loader, device, len(group_names), group_names, ce_weights=ce_weights, collect=args.export_predictions)
    pd.DataFrame(test_metrics["per_class"]).to_csv(run_dir / "test_per_group_metrics.csv", index=False)
    summary = {
        "label_space": "fixed_k11",
        "partition_path": str(partition_path(args)),
        "best_epoch": best_epoch,
        "checkpoint_selection_score": args.checkpoint_selection_score,
        "best_val_score": best_score,
        "best_val_macro_f1": best_macro,
        "test_accuracy": test_metrics["accuracy"],
        "test_macro_f1": test_metrics["macro_f1"],
        "test_weighted_f1": test_metrics["weighted_f1"],
        "group_names": group_names,
        "provenance": artifacts.provenance(
            args.split_manifest,
            args.input_schema_sha256,
            args.input_feature_schema_sha256,
        ),
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
        f"Finished fixed-k11 classifier: best_epoch={best_epoch}; "
        f"test_acc={test_metrics['accuracy']:.3f}; test_macro_f1={test_metrics['macro_f1']:.3f}; "
        f"test_weighted_f1={test_metrics['weighted_f1']:.3f}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
