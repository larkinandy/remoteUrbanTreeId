#!/usr/bin/env python3
"""Shared data reconstruction helpers for stage-two ensemble evaluation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
DISCRIMINATOR_DIR = HERE.parent / "discriminator"
for module_dir in (HERE, DISCRIMINATOR_DIR):
    if str(module_dir) not in sys.path:
        sys.path.insert(0, str(module_dir))

import train_fixed_partition_classifier as stage2
import train_taxon_discriminator as discriminator


def load_json(path: Path) -> dict[str, Any]:
    """Read a UTF-8 JSON object."""
    return json.loads(path.read_text(encoding="utf-8"))


def as_numpy(value: Any) -> np.ndarray | None:
    """Convert a tensor-like checkpoint value to a float32 NumPy array."""
    if value is None:
        return None
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy().astype(np.float32)
    return np.asarray(value, dtype=np.float32)


def runtime_args(args: argparse.Namespace, checkpoint: dict[str, Any]) -> argparse.Namespace:
    """Recreate data/model arguments from a checkpoint with evaluation overrides."""
    run_config = load_json(args.model_run_dir / "run_config.json").get("args", {})
    saved = dict(checkpoint.get("args") or run_config)
    saved.update(
        complete_shard_root=args.complete_shard_root,
        complete_shard_pattern=args.complete_shard_pattern,
        city_token=args.city_token,
        exclude_city_token=args.exclude_city_token,
        max_records_per_city=0,
        max_shard_cache=args.max_shard_cache,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        eval_num_workers=args.num_workers,
        image_augmentation=False,
        split_manifest=args.split_manifest,
    )
    for name in (
        "tree_image_channel_dropout",
        "tree_image_branch_dropout",
        "satellite_embedding_dropout",
        "satellite_embedding_value_dropout",
        "prism_normals_dropout",
        "prism_normals_value_dropout",
    ):
        saved[name] = 0.0
    return argparse.Namespace(**saved)


def build_manifest(
    args: argparse.Namespace,
    run_args: argparse.Namespace,
) -> tuple[pd.DataFrame, dict[str, Path], list[str]]:
    """Reconstruct the frozen labeled manifest and inherited data splits."""
    stage2.ensure_shard_pattern(run_args)
    label_to_group, group_names = stage2.load_partition(stage2.DEFAULT_MANUSCRIPT_PARTITION)
    manifest, paths = discriminator.discover_shards(run_args)
    manifest["fine_label"] = discriminator.scientific_label_from_name(
        manifest["scientific_name"], "scientific_binomial_or_genus"
    )
    before = len(manifest)
    manifest = manifest.loc[manifest["fine_label"].isin(label_to_group)].copy()
    print(
        "Dropped records not represented in the fixed k11 partition: "
        f"{before:,}->{len(manifest):,}",
        flush=True,
    )
    manifest["label_index"] = manifest["fine_label"].map(label_to_group).astype(np.int64)
    manifest["sample_weight"] = discriminator.compute_sample_weights(manifest, run_args)

    if bool(getattr(run_args, "exclude_qa_from_model", False)) and "qa_exclude_from_model" in manifest.columns:
        qa_excluded = discriminator.bool_series(manifest["qa_exclude_from_model"])
        removed = int(qa_excluded.sum())
        manifest = manifest.loc[~qa_excluded].copy()
        print(f"Dropped QA-excluded records before inherited splits: {removed:,}", flush=True)

    split_frame = pd.read_csv(args.split_manifest, usecols=["record_key", "split"])
    if split_frame["record_key"].duplicated().any():
        raise SystemExit(f"Duplicate record_key values in {args.split_manifest}")
    tree_id = pd.to_numeric(manifest["tree_id"], errors="raise").astype("int64").astype(str)
    crop_index = pd.to_numeric(manifest["crop_index"], errors="raise").astype("int64").astype(str)
    manifest["record_key"] = manifest["city_token"].astype(str) + "|" + tree_id + "|" + crop_index
    manifest = manifest.merge(split_frame, on="record_key", how="left", validate="one_to_one")
    missing = int(manifest["split"].isna().sum())
    if missing:
        raise SystemExit(
            f"Inherited split manifest is missing {missing:,} mapped model records; "
            "refusing to create implicit assignments."
        )
    invalid = sorted(set(manifest["split"].astype(str)).difference({"train", "val", "test"}))
    if invalid:
        raise SystemExit(f"Invalid inherited split values: {invalid}")
    print(f"Using inherited split assignments from {args.split_manifest}", flush=True)
    return manifest.reset_index(drop=True), paths, group_names


def make_loader(dataset: Any, args: argparse.Namespace) -> Any:
    """Create a deterministic non-shuffled evaluation DataLoader."""
    torch = discriminator.train_base.torch
    kwargs: dict[str, Any] = {
        "batch_size": int(args.batch_size),
        "shuffle": False,
        "num_workers": int(args.num_workers),
        "pin_memory": True,
    }
    if int(args.num_workers) > 0:
        kwargs["prefetch_factor"] = 3
        kwargs["persistent_workers"] = True
    return torch.utils.data.DataLoader(dataset, **kwargs)
