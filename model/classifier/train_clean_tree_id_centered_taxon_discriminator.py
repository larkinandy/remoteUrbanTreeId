#!/usr/bin/env python3
"""Train a clean tree_id-centered taxon discriminator from complete shards.

This script is intentionally standalone for the clean H:\\TreeCenteredModelInputs
pipeline. It does not import the previous discriminator trainer and it does not
try to infer row alignment from old cell-centered IDs.

Clean shard contract:
  * each ``*_inputs.npz`` is a compact table of already-screened records
  * its companion ``*_metadata.csv`` has the same row order
  * ``tree_id`` is the stable record identifier
  * arrays are indexed by local shard row, not by original crop_index
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from model_components import PhenologyEncoder
import model_components as train_base


CLEAN_ROOT = Path(r"H:\TreeCenteredModelInputs")
DEFAULT_SHARD_ROOT = CLEAN_ROOT / "tree_centered_complete_sharded100k_clean"
DEFAULT_OUTPUT_DIR = CLEAN_ROOT / "taxon_discrimination_clean"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--complete-shard-root", type=Path, default=DEFAULT_SHARD_ROOT)
    parser.add_argument("--complete-shard-pattern", default="*_part*_tree_centered_complete_inputs.npz")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--run-name", default="clean_tree_id_taxon_discriminator_full_context")
    parser.add_argument("--city-token", action="append", default=None)
    parser.add_argument("--exclude-city-token", action="append", default=[])
    parser.add_argument("--label-column", default="taxon_label")
    parser.add_argument(
        "--label-mode",
        choices=("column", "scientific_genus", "scientific_binomial", "scientific_binomial_or_genus"),
        default="scientific_binomial_or_genus",
        help=(
            "Default uses scientific_name so rows with coarse taxon_label values like other_broadleaf "
            "can still contribute genus/species-level supervision."
        ),
    )
    parser.add_argument("--exclude-label", action="append", default=[])
    parser.add_argument("--exclude-catch-all-labels", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--catch-all-label-prefix", action="append", default=["other_"])
    parser.add_argument(
        "--exclude-source-catch-all-labels",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Also drop rows whose original taxon_label is catch-all when using scientific label modes.",
    )
    parser.add_argument("--min-class-train-samples", type=int, default=200)
    parser.add_argument("--max-records-per-city", type=int, default=0)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--test-fraction", type=float, default=0.15)
    parser.add_argument("--max-val-samples", type=int, default=150_000)
    parser.add_argument("--max-test-samples", type=int, default=150_000)
    parser.add_argument("--reassign-capped-eval-to-train", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--exclude-qa-from-model", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use-qa-sample-weights", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--low-center-vegetation-weight", type=float, default=0.5)
    parser.add_argument("--low-vegetated-height-with-lidar-weight", type=float, default=0.5)
    parser.add_argument("--insufficient-lidar-coverage-weight", type=float, default=1.0)
    parser.add_argument("--use-distance-sample-weights", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--distance-weight-sigma-m", type=float, default=4.0)
    parser.add_argument("--min-distance-sample-weight", type=float, default=0.25)

    parser.add_argument("--use-sentinel-phenology", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--phenology-stat-samples", type=int, default=20_000)
    parser.add_argument("--use-raw-sentinel", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--raw-sentinel-stat-samples", type=int, default=20_000)
    parser.add_argument("--use-satellite-embedding", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--satellite-embedding-stat-samples", type=int, default=20_000)
    parser.add_argument("--satellite-embedding-dropout", type=float, default=0.0)
    parser.add_argument("--satellite-embedding-value-dropout", type=float, default=0.0)
    parser.add_argument("--use-prism-normals", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--prism-normals-stat-samples", type=int, default=20_000)
    parser.add_argument("--prism-normals-dropout", type=float, default=0.0)
    parser.add_argument("--prism-normals-value-dropout", type=float, default=0.0)

    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=768)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--eval-num-workers", type=int, default=0)
    parser.add_argument("--prefetch-factor", type=int, default=3)
    parser.add_argument("--persistent-workers", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--shard-local-train-batches", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--shard-window-size", type=int, default=8)
    parser.add_argument("--max-shard-cache", type=int, default=12)
    parser.add_argument("--cache-val-in-memory", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--cache-test-in-memory", action=argparse.BooleanOptionalAction, default=False)

    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.20)
    parser.add_argument("--tree-image-branch-dropout", type=float, default=0.0)
    parser.add_argument("--tree-image-channel-dropout", type=float, default=0.0)
    parser.add_argument("--temperature", type=float, default=0.12)
    parser.add_argument("--classification-loss-weight", type=float, default=0.35)
    parser.add_argument("--class-weighting", choices=("none", "sqrt_inverse", "inverse"), default="sqrt_inverse")
    parser.add_argument("--class-weight-max", type=float, default=3.0)
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
    parser.add_argument("--image-augmentation", action="store_true")
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--early-stopping-patience", type=int, default=12)
    parser.add_argument("--early-stopping-min-delta", type=float, default=0.0005)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--export-embeddings", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def bool_series(values: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(values):
        return values.fillna(False)
    if pd.api.types.is_numeric_dtype(values):
        return pd.to_numeric(values, errors="coerce").fillna(0).ne(0)
    return values.astype(str).str.strip().str.lower().isin({"true", "1", "yes", "y"})


def normalized_label_text(values: pd.Series) -> pd.Series:
    return values.fillna("").astype(str).str.strip().str.lower().str.replace(r"\s+", " ", regex=True)


def scientific_label_from_name(scientific_name: pd.Series, mode: str) -> pd.Series:
    sci = scientific_name.fillna("").astype(str).str.strip().str.replace(r"\s+", " ", regex=True)
    tokens = sci.str.extract(r"^([A-Z][A-Za-z-]{2,})(?:\s+([a-z][a-z-]{2,}))?", expand=True)
    genus = tokens[0].fillna("").str.lower()
    species = tokens[1].fillna("").str.lower()
    plausible_genus = genus.ne("")
    plausible_species = species.ne("") & ~species.isin({"sp", "spp", "species", "hybrid", "unknown"})
    if mode == "scientific_genus":
        return genus.where(plausible_genus, "")
    if mode == "scientific_binomial":
        return (genus + "_" + species).where(plausible_genus & plausible_species, "")
    if mode == "scientific_binomial_or_genus":
        return (genus + "_" + species).where(plausible_genus & plausible_species, genus.where(plausible_genus, ""))
    raise ValueError(mode)


def build_target_labels(manifest: pd.DataFrame, args: argparse.Namespace) -> pd.Series:
    if args.label_mode == "column":
        if args.label_column not in manifest.columns:
            raise SystemExit(f"Missing label column: {args.label_column}")
        return normalized_label_text(manifest[args.label_column])
    if "scientific_name" not in manifest.columns:
        raise SystemExit(f"--label-mode {args.label_mode} requires scientific_name in metadata.")
    return scientific_label_from_name(manifest["scientific_name"], args.label_mode)


def compute_sample_weights(manifest: pd.DataFrame, args: argparse.Namespace) -> np.ndarray:
    weights = np.ones(len(manifest), dtype=np.float32)
    if args.use_qa_sample_weights:
        for column, weight in {
            "qa_flag_low_center_vegetation": args.low_center_vegetation_weight,
            "qa_flag_low_vegetated_height_with_lidar": args.low_vegetated_height_with_lidar_weight,
            "qa_flag_insufficient_lidar_coverage_for_height": args.insufficient_lidar_coverage_weight,
        }.items():
            if column in manifest.columns:
                weights[bool_series(manifest[column]).to_numpy(dtype=bool)] *= np.float32(max(0.0, float(weight)))
    if args.use_distance_sample_weights:
        distance_column = "match_distance_m" if "match_distance_m" in manifest.columns else "nearest_crown_distance_m"
        if distance_column in manifest.columns:
            distance = pd.to_numeric(manifest[distance_column], errors="coerce").to_numpy(dtype=np.float32)
            finite = np.isfinite(distance)
            sigma = max(float(args.distance_weight_sigma_m), 1.0e-6)
            decay = np.ones(len(manifest), dtype=np.float32)
            decay[finite] = np.exp(-0.5 * np.square(distance[finite] / sigma)).astype(np.float32)
            decay = np.maximum(decay, np.float32(max(0.0, float(args.min_distance_sample_weight))))
            weights *= decay
    return np.clip(weights, 0.0, 1.0).astype(np.float32)


def city_from_input_path(path: Path) -> str:
    return path.name.split("_part", 1)[0].lower()


def discover_shards(args: argparse.Namespace) -> tuple[pd.DataFrame, dict[str, Path]]:
    selected = None if args.city_token is None else {value.strip().lower() for value in args.city_token if value.strip()}
    excluded = {value.strip().lower() for value in args.exclude_city_token if value.strip()}
    rows: list[pd.DataFrame] = []
    paths: dict[str, Path] = {}
    for input_path in sorted(args.complete_shard_root.glob(f"*/{args.complete_shard_pattern}")):
        city = city_from_input_path(input_path)
        if selected is not None and city not in selected:
            continue
        if city in excluded:
            continue
        metadata_path = input_path.with_name(input_path.name.replace("_inputs.npz", "_metadata.csv"))
        if not metadata_path.exists():
            print(f"SKIP {input_path.name}: missing metadata {metadata_path}", flush=True)
            continue
        frame = pd.read_csv(metadata_path, low_memory=False)
        frame = frame.copy()
        source_key = input_path.stem.replace("_inputs", "")
        frame["city_token"] = city
        frame["source_key"] = source_key
        frame["local_row"] = np.arange(len(frame), dtype=np.int64)
        frame["source_complete_shard_path"] = str(input_path)
        frame["source_metadata_path"] = str(metadata_path)
        if "tree_id" not in frame.columns:
            raise RuntimeError(f"{metadata_path} is missing required clean key tree_id")
        if "crop_index" not in frame.columns:
            frame["crop_index"] = frame["local_row"]
        if args.max_records_per_city > 0 and len(frame) > args.max_records_per_city:
            frame = frame.sample(n=args.max_records_per_city, random_state=args.seed).sort_index().reset_index(drop=True)
        frame["local_row"] = (
            pd.to_numeric(frame["source_sample_index"], errors="coerce")
            .fillna(pd.Series(frame.index, index=frame.index, dtype=np.int64))
            .astype(np.int64)
            if "source_sample_index" in frame.columns
            else frame["local_row"]
        )
        rows.append(frame)
        paths[source_key] = input_path
    if not rows:
        raise SystemExit(f"No clean complete shards found under {args.complete_shard_root}")
    manifest = pd.concat(rows, ignore_index=True)
    print(f"Discovered clean complete tree_id shards: shards={len(paths):,}; rows={len(manifest):,}", flush=True)
    return manifest, paths


@dataclass
class CleanShardArrays:
    path: Path
    crops: np.ndarray
    chm: np.ndarray
    chm_valid_mask: np.ndarray
    vegetation_chm: np.ndarray
    structure: np.ndarray
    sentinel_phenology: np.ndarray | None
    sentinel_sequence: np.ndarray | None
    sentinel_sequence_mask: np.ndarray | None
    satellite_embedding: np.ndarray | None
    prism_normals: np.ndarray | None


class CleanShardStore:
    def __init__(self, paths: dict[str, Path], max_cached_shards: int = 12):
        self.paths = dict(paths)
        self.max_cached_shards = max(1, int(max_cached_shards))
        self.cache: dict[str, CleanShardArrays] = {}
        self.order: list[str] = []

    def get(self, source_key: str) -> CleanShardArrays:
        if source_key in self.cache:
            return self.cache[source_key]
        path = self.paths[source_key]
        data = np.load(path, allow_pickle=True)
        required = [
            "tree_centered_naip",
            "tree_centered_chm",
            "tree_centered_chm_valid_mask",
            "tree_centered_vegetation_chm",
            "tree_centered_naip_chm_structure",
        ]
        missing = [key for key in required if key not in data.files]
        if missing:
            raise RuntimeError(f"{path} is missing required clean shard arrays: {missing}")
        def metric_array(name: str) -> np.ndarray:
            values = np.asarray(data[name])
            scale_key = f"{name}_scale_m"
            if scale_key in data.files:
                scale = float(np.asarray(data[scale_key]).reshape(-1)[0])
                return values.astype(np.float32) * scale
            return values.astype(np.float32, copy=False)

        arrays = CleanShardArrays(
            path=path,
            crops=np.asarray(data["tree_centered_naip"]),
            chm=metric_array("tree_centered_chm"),
            chm_valid_mask=np.asarray(data["tree_centered_chm_valid_mask"], dtype=bool),
            vegetation_chm=metric_array("tree_centered_vegetation_chm"),
            structure=np.asarray(data["tree_centered_naip_chm_structure"], dtype=np.float32),
            sentinel_phenology=np.asarray(data["sentinel_phenology"], dtype=np.float32) if "sentinel_phenology" in data.files else None,
            sentinel_sequence=np.asarray(data["sentinel_sequence"], dtype=np.float32) if "sentinel_sequence" in data.files else None,
            sentinel_sequence_mask=np.asarray(data["sentinel_sequence_mask"], dtype=bool) if "sentinel_sequence_mask" in data.files else None,
            satellite_embedding=np.asarray(data["satellite_embedding"], dtype=np.float32) if "satellite_embedding" in data.files else None,
            prism_normals=np.asarray(data["prism_normals"], dtype=np.float32) if "prism_normals" in data.files else None,
        )
        row_count = len(arrays.crops)
        for name in ("chm", "chm_valid_mask", "vegetation_chm", "structure"):
            if len(getattr(arrays, name)) != row_count:
                raise RuntimeError(f"{path}: {name} rows do not match NAIP rows")
        for name in ("sentinel_phenology", "sentinel_sequence", "sentinel_sequence_mask", "satellite_embedding", "prism_normals"):
            value = getattr(arrays, name)
            if value is not None and len(value) != row_count:
                raise RuntimeError(f"{path}: {name} rows do not match NAIP rows")
        self.cache[source_key] = arrays
        self.order.append(source_key)
        while len(self.order) > self.max_cached_shards:
            old = self.order.pop(0)
            self.cache.pop(old, None)
        return arrays

    def clear(self) -> None:
        self.cache.clear()
        self.order.clear()


def split_manifest(manifest: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    manifest = manifest.reset_index(drop=True)
    rng = np.random.default_rng(args.seed)
    split = np.full(len(manifest), "train", dtype=object)
    for _label, group in manifest.groupby("label_index", sort=False):
        indices = group.index.to_numpy(copy=True)
        rng.shuffle(indices)
        test_n = int(round(len(indices) * args.test_fraction))
        val_n = int(round(len(indices) * args.val_fraction))
        split[indices[:test_n]] = "test"
        split[indices[test_n : test_n + val_n]] = "val"
    manifest = manifest.copy()
    manifest["split"] = split
    if args.exclude_qa_from_model and "qa_exclude_from_model" in manifest.columns:
        bad = bool_series(manifest["qa_exclude_from_model"])
        removed = int(bad.sum())
        if removed:
            print(f"Dropped QA-excluded records before model splits: {removed:,}", flush=True)
        manifest = manifest.loc[~bad].copy()
    return manifest.reset_index(drop=True)


def cap_eval_split(manifest: pd.DataFrame, split_name: str, max_samples: int, args: argparse.Namespace) -> pd.DataFrame:
    if max_samples <= 0:
        return manifest
    mask = manifest["split"].eq(split_name)
    count = int(mask.sum())
    if count <= max_samples:
        return manifest
    rng = np.random.default_rng(args.seed + (101 if split_name == "val" else 202))
    split_indices = manifest.index[mask].to_numpy(copy=True)
    rng.shuffle(split_indices)
    capped = split_indices[max_samples:]
    manifest = manifest.copy()
    if args.reassign_capped_eval_to_train:
        manifest.loc[capped, "split"] = "train"
        print(f"Capped {split_name} samples: {count:,}->{max_samples:,}; reassigned {len(capped):,} to train.", flush=True)
    else:
        manifest = manifest.drop(index=capped)
        print(f"Capped {split_name} samples: {count:,}->{max_samples:,}; discarded {len(capped):,}.", flush=True)
    return manifest.reset_index(drop=True)


def cap_eval_splits(manifest: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    manifest = cap_eval_split(manifest, "val", int(args.max_val_samples), args)
    manifest = cap_eval_split(manifest, "test", int(args.max_test_samples), args)
    return manifest


def sample_rows_for_stats(frame: pd.DataFrame, max_samples: int, seed: int) -> pd.DataFrame:
    if max_samples > 0 and len(frame) > max_samples:
        rng = np.random.default_rng(seed)
        return frame.iloc[rng.choice(np.arange(len(frame)), size=max_samples, replace=False)].copy()
    return frame


def estimate_array_standardizer(
    frame: pd.DataFrame,
    store: CleanShardStore,
    attr: str,
    max_samples: int,
    seed: int,
    flatten_valid_mask: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    frame = sample_rows_for_stats(frame, max_samples, seed)
    rows: list[np.ndarray] = []
    for source_key, group in frame.groupby("source_key", sort=False):
        arrays = store.get(str(source_key))
        values = getattr(arrays, attr)
        if values is None:
            raise RuntimeError(f"{arrays.path} is missing required array {attr}")
        local_rows = pd.to_numeric(group["local_row"], errors="raise").to_numpy(dtype=np.int64)
        selected = np.asarray(values[local_rows], dtype=np.float32)
        if flatten_valid_mask is not None:
            mask_values = getattr(arrays, flatten_valid_mask)
            if mask_values is None:
                raise RuntimeError(f"{arrays.path} is missing required mask {flatten_valid_mask}")
            mask = np.asarray(mask_values[local_rows], dtype=bool)
            selected = selected[mask]
            if selected.size:
                rows.append(selected.reshape(-1, selected.shape[-1] if selected.ndim > 1 else 1))
        else:
            rows.append(selected.reshape(selected.shape[0], -1))
    values = np.concatenate(rows, axis=0) if rows else np.empty((0, 0), dtype=np.float32)
    mean = values.mean(axis=0).astype(np.float32) if len(values) else np.zeros(0, dtype=np.float32)
    std = values.std(axis=0).astype(np.float32) if len(values) else np.ones_like(mean, dtype=np.float32)
    std = np.where(std < 1.0e-6, 1.0, std).astype(np.float32)
    return mean, std


def estimate_raw_sentinel_standardizer(
    frame: pd.DataFrame,
    store: CleanShardStore,
    max_samples: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    frame = sample_rows_for_stats(frame, max_samples, seed)
    rows: list[np.ndarray] = []
    for source_key, group in frame.groupby("source_key", sort=False):
        arrays = store.get(str(source_key))
        if arrays.sentinel_sequence is None or arrays.sentinel_sequence_mask is None:
            raise RuntimeError(f"{arrays.path} is missing raw Sentinel sequence arrays")
        local_rows = pd.to_numeric(group["local_row"], errors="raise").to_numpy(dtype=np.int64)
        sequence = np.asarray(arrays.sentinel_sequence[local_rows], dtype=np.float32)
        mask = np.asarray(arrays.sentinel_sequence_mask[local_rows], dtype=bool)
        valid = sequence[mask]
        if valid.size:
            rows.append(valid.reshape(-1, sequence.shape[-1]))
    values = np.concatenate(rows, axis=0) if rows else np.empty((0, 0), dtype=np.float32)
    mean = values.mean(axis=0).astype(np.float32) if len(values) else np.zeros(0, dtype=np.float32)
    std = values.std(axis=0).astype(np.float32) if len(values) else np.ones_like(mean, dtype=np.float32)
    std = np.where(std < 1.0e-6, 1.0, std).astype(np.float32)
    return mean, std


class CleanTreeDataset(train_base.Dataset):
    def __init__(
        self,
        manifest: pd.DataFrame,
        store: CleanShardStore,
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
        preload_in_memory: bool = False,
    ):
        self.manifest = manifest.reset_index(drop=True)
        self.store = store
        self.structure_mean = np.asarray(structure_mean, dtype=np.float32)
        self.structure_std = np.asarray(structure_std, dtype=np.float32)
        self.phenology_mean = phenology_mean
        self.phenology_std = phenology_std
        self.raw_sentinel_mean = raw_sentinel_mean
        self.raw_sentinel_std = raw_sentinel_std
        self.satellite_embedding_mean = satellite_embedding_mean
        self.satellite_embedding_std = satellite_embedding_std
        self.prism_mean = prism_mean
        self.prism_std = prism_std
        self.args = args
        self.source_keys = self.manifest["source_key"].astype(str).to_numpy()
        self.local_rows = pd.to_numeric(self.manifest["local_row"], errors="raise").to_numpy(dtype=np.int64)
        self.tree_ids = pd.to_numeric(self.manifest["tree_id"], errors="raise").to_numpy(dtype=np.int64)
        crop_index_values = pd.to_numeric(self.manifest["crop_index"], errors="coerce")
        local_row_values = pd.Series(self.local_rows, index=self.manifest.index)
        self.crop_indices = crop_index_values.fillna(local_row_values).to_numpy(dtype=np.int64)
        self.labels = pd.to_numeric(self.manifest["label_index"], errors="raise").to_numpy(dtype=np.int64)
        self.sample_weights = pd.to_numeric(self.manifest["sample_weight"], errors="coerce").fillna(1.0).to_numpy(dtype=np.float32)
        self.preloaded: list[dict[str, Any]] | None = None
        if preload_in_memory:
            old_aug = self.args.image_augmentation
            self.args.image_augmentation = False
            self.preloaded = [self._build_item(i) for i in range(len(self))]
            self.args.image_augmentation = old_aug

    def __len__(self) -> int:
        return len(self.manifest)

    def _apply_raster_transform(self, image: train_base.torch.Tensor, transform_index: int) -> train_base.torch.Tensor:
        if transform_index == 1:
            return train_base.torch.flip(image, dims=(-1,))
        if transform_index == 2:
            return train_base.torch.flip(image, dims=(-2,))
        if transform_index in (3, 4, 5):
            return train_base.torch.rot90(image, k={3: 1, 4: 2, 5: 3}[transform_index], dims=(-2, -1))
        return image

    def _image(
        self,
        chip_raw: np.ndarray,
        transform_index: int,
        center_crop: bool = True,
        crop_pixels: int | None = None,
    ) -> train_base.torch.Tensor:
        requested_crop = (
            int(self.args.image_center_crop_pixels)
            if crop_pixels is None
            else int(crop_pixels)
        )
        if (
            center_crop
            and requested_crop > 0
            and requested_crop < min(chip_raw.shape[0], chip_raw.shape[1])
        ):
            size = requested_crop
            y0 = (chip_raw.shape[0] - size) // 2
            x0 = (chip_raw.shape[1] - size) // 2
            chip_raw = chip_raw[y0 : y0 + size, x0 : x0 + size, :]
        chip = np.asarray(chip_raw, dtype=np.float32)
        red = chip[:, :, 0]
        nir = chip[:, :, 3]
        denom = nir + red
        ndvi = np.divide(nir - red, denom, out=np.zeros_like(nir, dtype=np.float32), where=denom > 0)
        image_np = np.empty((5, chip.shape[0], chip.shape[1]), dtype=np.float32)
        image_np[:4] = np.transpose(chip / 255.0, (2, 0, 1))
        image_np[4] = np.nan_to_num(ndvi, nan=0.0, posinf=0.0, neginf=0.0)
        image = train_base.torch.from_numpy(image_np)
        if self.args.image_augmentation:
            image = self._apply_raster_transform(image, transform_index)
            if self.args.tree_image_channel_dropout > 0:
                keep = train_base.torch.rand(image.shape[0]) >= self.args.tree_image_channel_dropout
                if not bool(keep.any()):
                    keep[int(train_base.torch.randint(0, image.shape[0], ()).item())] = True
                image = image * keep.to(dtype=image.dtype).view(-1, 1, 1) / max(1.0 - self.args.tree_image_channel_dropout, 1.0e-6)
        return image

    def _chm_image(
        self,
        arrays: CleanShardArrays,
        local_row: int,
        transform_index: int,
        align_to_naip_crop: bool = False,
        naip_crop_pixels: int | None = None,
    ) -> train_base.torch.Tensor:
        chm = np.nan_to_num(np.asarray(arrays.chm[local_row], dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        valid = np.asarray(arrays.chm_valid_mask[local_row], dtype=bool)
        vegetation = np.nan_to_num(np.asarray(arrays.vegetation_chm[local_row], dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        naip_shape = np.asarray(arrays.crops[local_row]).shape[:2]
        requested_naip_crop = (
            int(self.args.image_center_crop_pixels)
            if naip_crop_pixels is None
            else int(naip_crop_pixels)
        )
        if (
            align_to_naip_crop
            and requested_naip_crop > 0
            and len(naip_shape) == 2
            and requested_naip_crop < min(naip_shape)
        ):
            # The interaction branch must cover the same ground footprint as
            # the NAIP center crop. Keep the independent CHM branch at its
            # native 38 x 38 extent for broader structural context, but apply
            # the NAIP crop fraction here before interaction-branch resampling.
            # Example: 28/64 of a 38 x 38 CHM -> 17 x 17.
            crop_fraction = float(requested_naip_crop) / float(min(naip_shape))
            out_h = min(chm.shape[0], max(1, int(round(chm.shape[0] * crop_fraction))))
            out_w = min(chm.shape[1], max(1, int(round(chm.shape[1] * crop_fraction))))
            y0 = max(0, (chm.shape[0] - out_h) // 2)
            x0 = max(0, (chm.shape[1] - out_w) // 2)
            crop = np.s_[y0 : y0 + out_h, x0 : x0 + out_w]
            chm = chm[crop]
            valid = valid[crop]
            vegetation = vegetation[crop]
        valid_f = valid.astype(np.float32)
        chm_np = np.stack(
            [
                np.clip(chm, 0.0, 40.0) / 40.0 * valid_f,
                valid_f,
                np.clip(vegetation, 0.0, 40.0) / 40.0 * valid_f,
            ],
            axis=0,
        ).astype(np.float32)
        tensor = train_base.torch.from_numpy(chm_np)
        if self.args.image_augmentation:
            tensor = self._apply_raster_transform(tensor, transform_index)
        return tensor

    def _normalize_vector(self, value: np.ndarray, mean: np.ndarray | None, std: np.ndarray | None) -> np.ndarray:
        value = np.asarray(value, dtype=np.float32).reshape(-1)
        if mean is not None and std is not None:
            value = (value - mean) / std
        return np.clip(np.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0), -12.0, 12.0).astype(np.float32)

    def _build_item(self, index: int) -> dict[str, Any]:
        source_key = self.source_keys[index]
        local_row = int(self.local_rows[index])
        arrays = self.store.get(source_key)
        transform_index = int(train_base.torch.randint(0, 6, ()).item()) if self.args.image_augmentation else 0
        structure = self._normalize_vector(arrays.structure[local_row], self.structure_mean, self.structure_std)
        interaction_enabled = bool(getattr(self.args, "use_naip_chm_interaction_branch", False))
        interaction_full_extent = bool(getattr(self.args, "interaction_naip_full_extent", False))
        interaction_crop_pixels = int(getattr(self.args, "interaction_naip_crop_pixels", 0))
        if interaction_enabled and interaction_full_extent:
            tree_image = self._image(arrays.crops[local_row], transform_index, center_crop=False)
        elif interaction_enabled and interaction_crop_pixels > 0:
            # Build one NAIP tensor per record. The classifier center-crops this
            # tensor for the independent NAIP encoder and passes it unchanged to
            # the interaction encoder.
            tree_image = self._image(
                arrays.crops[local_row],
                transform_index,
                crop_pixels=interaction_crop_pixels,
            )
        else:
            tree_image = self._image(arrays.crops[local_row], transform_index)
        item: dict[str, Any] = {
            "tree_image": tree_image,
            "tree_chm_image": self._chm_image(
                arrays,
                local_row,
                transform_index,
                align_to_naip_crop=not bool(getattr(self.args, "independent_chm_full_extent", False)),
            ),
            "tree_centered_structure": train_base.torch.from_numpy(structure),
            "label": train_base.torch.tensor(int(self.labels[index]), dtype=train_base.torch.long),
            "sample_weight": train_base.torch.tensor(float(self.sample_weights[index]), dtype=train_base.torch.float32),
            "city_token": str(self.manifest["city_token"].iloc[index]),
            "tree_centered_index": train_base.torch.tensor(int(self.tree_ids[index]), dtype=train_base.torch.long),
            "crop_index": train_base.torch.tensor(int(self.crop_indices[index]), dtype=train_base.torch.long),
        }
        if interaction_enabled:
            item["tree_chm_interaction_image"] = self._chm_image(
                arrays,
                local_row,
                transform_index,
                align_to_naip_crop=not interaction_full_extent,
                naip_crop_pixels=interaction_crop_pixels if interaction_crop_pixels > 0 else None,
            )
        if self.phenology_mean is not None:
            item["sentinel_phenology"] = train_base.torch.from_numpy(
                self._normalize_vector(arrays.sentinel_phenology[local_row], self.phenology_mean, self.phenology_std)
            )
        if self.raw_sentinel_mean is not None:
            if arrays.sentinel_sequence is None or arrays.sentinel_sequence_mask is None:
                raise RuntimeError(f"{arrays.path} is missing raw Sentinel arrays")
            sequence = np.asarray(arrays.sentinel_sequence[local_row], dtype=np.float32)
            mask = np.asarray(arrays.sentinel_sequence_mask[local_row], dtype=np.float32)
            sequence = (sequence - self.raw_sentinel_mean.reshape(1, -1)) / self.raw_sentinel_std.reshape(1, -1)
            sequence = np.clip(np.nan_to_num(sequence, nan=0.0, posinf=0.0, neginf=0.0), -12.0, 12.0).astype(np.float32)
            item["sentinel_sequence"] = train_base.torch.from_numpy(sequence)
            item["sentinel_sequence_mask"] = train_base.torch.from_numpy(mask)
        if self.satellite_embedding_mean is not None:
            emb = self._normalize_vector(arrays.satellite_embedding[local_row], self.satellite_embedding_mean, self.satellite_embedding_std)
            if self.args.image_augmentation and self.args.satellite_embedding_value_dropout > 0:
                emb = emb * (np.random.random(emb.shape) >= self.args.satellite_embedding_value_dropout).astype(np.float32)
            if self.args.image_augmentation and self.args.satellite_embedding_dropout > 0 and float(np.random.random()) < self.args.satellite_embedding_dropout:
                emb = np.zeros_like(emb, dtype=np.float32)
            item["satellite_embedding"] = train_base.torch.from_numpy(emb)
        if self.prism_mean is not None:
            prism = self._normalize_vector(arrays.prism_normals[local_row], self.prism_mean, self.prism_std)
            if self.args.image_augmentation and self.args.prism_normals_value_dropout > 0:
                prism = prism * (np.random.random(prism.shape) >= self.args.prism_normals_value_dropout).astype(np.float32)
            if self.args.image_augmentation and self.args.prism_normals_dropout > 0 and float(np.random.random()) < self.args.prism_normals_dropout:
                prism = np.zeros_like(prism, dtype=np.float32)
            item["prism_normals"] = train_base.torch.from_numpy(prism)
        return item

    def __getitem__(self, index: int) -> dict[str, Any]:
        if self.preloaded is not None:
            return self.preloaded[index]
        return self._build_item(index)


class CleanTreeDiscriminator(train_base.nn.Module):
    def __init__(
        self,
        class_count: int,
        structure_dim: int,
        hidden_dim: int,
        embedding_dim: int,
        dropout: float,
        phenology_dim: int = 0,
        raw_sentinel_dim: int = 0,
        satellite_embedding_dim: int = 0,
        prism_normals_dim: int = 0,
        tree_image_branch_dropout: float = 0.0,
    ):
        super().__init__()
        self.tree_image_branch_dropout = float(tree_image_branch_dropout)
        self.tree_image_modality_index = 0
        self.modality_names = ("tree_centered_naip", "tree_centered_chm", "tree_centered_structure")
        if phenology_dim > 0:
            self.modality_names += ("sentinel_phenology",)
        if raw_sentinel_dim > 0:
            self.modality_names += ("raw_sentinel",)
        if satellite_embedding_dim > 0:
            self.modality_names += ("satellite_embedding",)
        if prism_normals_dim > 0:
            self.modality_names += ("prism_normals",)
        self.image_encoder = train_base.ImageEncoder(5, hidden_dim, dropout)
        self.chm_encoder = train_base.ImageEncoder(3, hidden_dim, dropout)
        self.structure_encoder = PhenologyEncoder(structure_dim, hidden_dim, dropout)
        self.phenology_encoder = PhenologyEncoder(phenology_dim, hidden_dim, dropout) if phenology_dim > 0 else None
        self.raw_sentinel_encoder = train_base.MaskedSequenceEncoder(raw_sentinel_dim, hidden_dim, dropout) if raw_sentinel_dim > 0 else None
        self.satellite_embedding_encoder = PhenologyEncoder(satellite_embedding_dim, hidden_dim, dropout) if satellite_embedding_dim > 0 else None
        self.prism_normals_encoder = PhenologyEncoder(prism_normals_dim, hidden_dim, dropout) if prism_normals_dim > 0 else None
        modality_count = len(self.modality_names)
        self.gate = train_base.nn.Sequential(
            train_base.nn.LayerNorm(hidden_dim * modality_count),
            train_base.nn.Linear(hidden_dim * modality_count, hidden_dim),
            train_base.nn.GELU(),
            train_base.nn.Dropout(dropout),
            train_base.nn.Linear(hidden_dim, modality_count),
        )
        self.projector = train_base.nn.Sequential(
            train_base.nn.LayerNorm(hidden_dim),
            train_base.nn.Linear(hidden_dim, hidden_dim),
            train_base.nn.GELU(),
            train_base.nn.Dropout(dropout),
            train_base.nn.Linear(hidden_dim, embedding_dim),
        )
        self.classifier = train_base.nn.Linear(embedding_dim, class_count)

    def encode(self, batch: dict[str, train_base.torch.Tensor]):
        features = [
            self.image_encoder(batch["tree_image"]),
            self.chm_encoder(batch["tree_chm_image"]),
            self.structure_encoder(batch["tree_centered_structure"]),
        ]
        if self.phenology_encoder is not None:
            features.append(self.phenology_encoder(batch["sentinel_phenology"]))
        if self.raw_sentinel_encoder is not None:
            features.append(self.raw_sentinel_encoder(batch["sentinel_sequence"], batch["sentinel_sequence_mask"]))
        if self.satellite_embedding_encoder is not None:
            features.append(self.satellite_embedding_encoder(batch["satellite_embedding"]))
        if self.prism_normals_encoder is not None:
            features.append(self.prism_normals_encoder(batch["prism_normals"]))
        encoded = train_base.torch.stack(features, dim=1)
        gate_logits = self.gate(encoded.flatten(1))
        if self.training and self.tree_image_branch_dropout > 0:
            dropped = train_base.torch.rand(encoded.shape[0], device=encoded.device) < self.tree_image_branch_dropout
            if dropped.any():
                encoded = encoded.clone()
                gate_logits = gate_logits.clone()
                encoded[dropped, self.tree_image_modality_index, :] = 0.0
                gate_logits[dropped, self.tree_image_modality_index] = -1.0e4
        gate_weights = train_base.torch.softmax(gate_logits, dim=1)
        fused = (encoded * gate_weights.unsqueeze(-1)).sum(dim=1)
        z = train_base.F.normalize(self.projector(fused), dim=1)
        return z, gate_weights

    def forward(self, batch: dict[str, train_base.torch.Tensor]):
        z, gates = self.encode(batch)
        return z, self.classifier(z), gates


def supervised_contrastive_loss(z, labels, temperature: float, sample_weights=None):
    labels = labels.view(-1, 1)
    mask = train_base.torch.eq(labels, labels.T).float().to(z.device)
    logits = train_base.torch.matmul(z, z.T) / temperature
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()
    logits_mask = train_base.torch.ones_like(mask) - train_base.torch.eye(mask.shape[0], device=z.device)
    mask = mask * logits_mask
    exp_logits = train_base.torch.exp(logits) * logits_mask
    log_prob = logits - train_base.torch.log(exp_logits.sum(1, keepdim=True) + 1.0e-12)
    positives = mask.sum(1)
    valid = positives > 0
    loss = -((mask * log_prob).sum(1) / train_base.torch.clamp(positives, min=1.0))[valid]
    if loss.numel() == 0:
        return z.new_tensor(0.0)
    if sample_weights is None:
        return loss.mean()
    weights = sample_weights.to(device=z.device, dtype=z.dtype)[valid]
    return (loss * weights).sum() / train_base.torch.clamp(weights.sum(), min=1.0e-6)


def class_weights(counts: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    if args.class_weighting == "none":
        return np.ones_like(counts, dtype=np.float32)
    positive = counts[counts > 0]
    reference = float(np.median(positive)) if len(positive) else 1.0
    weights = np.ones_like(counts, dtype=np.float32)
    for i, count in enumerate(counts):
        if count <= 0:
            weights[i] = 0.0
        elif args.class_weighting == "sqrt_inverse":
            weights[i] = math.sqrt(reference / float(count))
        else:
            weights[i] = reference / float(count)
    if args.class_weight_max > 0:
        weights = np.clip(weights, 0.0, float(args.class_weight_max))
    supported = weights[counts > 0]
    if len(supported):
        weights = weights / max(float(supported.mean()), 1.0e-6)
    return weights.astype(np.float32)


def f1_metrics(y_true: np.ndarray, y_pred: np.ndarray, class_count: int) -> dict[str, float]:
    f1 = []
    support = []
    for cls in range(class_count):
        tp = np.sum((y_true == cls) & (y_pred == cls))
        fp = np.sum((y_true != cls) & (y_pred == cls))
        fn = np.sum((y_true == cls) & (y_pred != cls))
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1.append(2 * precision * recall / max(precision + recall, 1.0e-12))
        support.append(np.sum(y_true == cls))
    f1 = np.asarray(f1, dtype=np.float64)
    support = np.asarray(support, dtype=np.float64)
    return {
        "accuracy": float(np.mean(y_true == y_pred)) if len(y_true) else 0.0,
        "macro_f1": float(f1.mean()) if len(f1) else 0.0,
        "weighted_f1": float((f1 * support).sum() / max(support.sum(), 1.0)),
    }


class SourceShardBatchSampler:
    def __init__(self, ds: CleanTreeDataset, batch_size: int, seed: int, shard_window_size: int = 8):
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.shard_window_size = max(1, int(shard_window_size))
        self.epoch = 0
        by_source: dict[str, list[int]] = {}
        for pos, source_key in enumerate(ds.source_keys.tolist()):
            by_source.setdefault(str(source_key), []).append(pos)
        self.by_source = by_source
        self.batch_count = sum((len(values) + self.batch_size - 1) // self.batch_size for values in by_source.values())

    def __len__(self) -> int:
        return self.batch_count

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        self.epoch += 1
        source_keys = list(self.by_source)
        rng.shuffle(source_keys)
        for window_start in range(0, len(source_keys), self.shard_window_size):
            window_keys = source_keys[window_start : window_start + self.shard_window_size]
            batches_by_source: list[list[list[int]]] = []
            for key in window_keys:
                shuffled = np.asarray(self.by_source[key], dtype=np.int64)
                rng.shuffle(shuffled)
                batches_by_source.append(
                    [shuffled[start : start + self.batch_size].astype(int).tolist() for start in range(0, len(shuffled), self.batch_size)]
                )
            active = [i for i, batches in enumerate(batches_by_source) if batches]
            while active:
                cycle = np.asarray(active, dtype=np.int64)
                rng.shuffle(cycle)
                for index in cycle.astype(int).tolist():
                    if batches_by_source[index]:
                        yield batches_by_source[index].pop()
                active = [i for i in active if batches_by_source[i]]


def make_loader(ds: CleanTreeDataset, args: argparse.Namespace, train: bool):
    if train and args.shard_local_train_batches:
        kwargs: dict[str, Any] = {
            "batch_sampler": SourceShardBatchSampler(ds, args.batch_size, args.seed, args.shard_window_size),
            "num_workers": args.num_workers,
            "pin_memory": str(args.device).startswith("cuda"),
        }
        if args.num_workers > 0:
            kwargs["prefetch_factor"] = args.prefetch_factor
            kwargs["persistent_workers"] = bool(args.persistent_workers)
        print(f"Training batches: round-robin complete-shard batches enabled (window_size={args.shard_window_size}).", flush=True)
        return train_base.DataLoader(ds, **kwargs)
    kwargs = {
        "batch_size": args.batch_size,
        "shuffle": train,
        "num_workers": args.num_workers if train else args.eval_num_workers,
        "pin_memory": str(args.device).startswith("cuda"),
    }
    if kwargs["num_workers"] > 0:
        kwargs["prefetch_factor"] = args.prefetch_factor
        kwargs["persistent_workers"] = bool(args.persistent_workers)
    return train_base.DataLoader(ds, **kwargs)


def train_epoch(model, loader, optimizer, device, args, ce_weights):
    model.train()
    total = {"loss": 0.0, "supcon": 0.0, "ce": 0.0, "n": 0}
    for batch in loader:
        batch = {key: value.to(device, non_blocking=True) if hasattr(value, "to") else value for key, value in batch.items()}
        labels = batch["label"]
        sample_weights = batch["sample_weight"].to(device=device, dtype=train_base.torch.float32)
        optimizer.zero_grad(set_to_none=True)
        z, logits, _gates = model(batch)
        supcon = supervised_contrastive_loss(z, labels, args.temperature, sample_weights)
        ce_per_sample = train_base.F.cross_entropy(logits, labels, weight=ce_weights, reduction="none")
        ce = (ce_per_sample * sample_weights).sum() / train_base.torch.clamp(sample_weights.sum(), min=1.0e-6)
        loss = supcon + float(args.classification_loss_weight) * ce
        loss.backward()
        if args.grad_clip > 0:
            train_base.torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        n = int(labels.numel())
        total["loss"] += float(loss.detach().cpu()) * n
        total["supcon"] += float(supcon.detach().cpu()) * n
        total["ce"] += float(ce.detach().cpu()) * n
        total["n"] += n
    n = max(total.pop("n"), 1)
    return {key: value / n for key, value in total.items()}


@train_base.torch.no_grad()
def evaluate(model, loader, device, class_count: int, collect: bool = False):
    model.eval()
    y_true, y_pred = [], []
    gate_sum = None
    gate_n = 0
    zs, cities, tree_ids, crop_indices = [], [], [], []
    for batch in loader:
        city_token = batch.pop("city_token")
        batch = {key: value.to(device, non_blocking=True) if hasattr(value, "to") else value for key, value in batch.items()}
        z, logits, gates = model(batch)
        pred = logits.argmax(dim=1)
        y_true.append(batch["label"].detach().cpu().numpy())
        y_pred.append(pred.detach().cpu().numpy())
        gate_values = gates.detach().cpu().numpy()
        gate_sum = gate_values.sum(axis=0) if gate_sum is None else gate_sum + gate_values.sum(axis=0)
        gate_n += gate_values.shape[0]
        if collect:
            zs.append(z.detach().cpu().numpy())
            cities.extend(list(city_token))
            tree_ids.append(batch["tree_centered_index"].detach().cpu().numpy())
            crop_indices.append(batch["crop_index"].detach().cpu().numpy())
    true = np.concatenate(y_true) if y_true else np.empty(0, dtype=np.int64)
    pred = np.concatenate(y_pred) if y_pred else np.empty(0, dtype=np.int64)
    metrics = f1_metrics(true, pred, class_count)
    if gate_sum is not None and gate_n > 0:
        for name, value in zip(model.modality_names, gate_sum / gate_n):
            metrics[f"gate_{name}"] = float(value)
    if collect:
        metrics["embeddings"] = np.concatenate(zs) if zs else np.empty((0, 0), dtype=np.float32)
        metrics["labels"] = true
        metrics["predictions"] = pred
        metrics["city_token"] = np.asarray(cities, dtype="U64")
        metrics["tree_id"] = np.concatenate(tree_ids) if tree_ids else np.empty(0, dtype=np.int64)
        metrics["crop_index"] = np.concatenate(crop_indices) if crop_indices else np.empty(0, dtype=np.int64)
    return metrics


def safe_json_args(args: argparse.Namespace) -> dict[str, Any]:
    return {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}


def main() -> int:
    args = parse_args()
    for label in ("complete_shard_root", "output_dir"):
        resolved = Path(getattr(args, label)).resolve()
        if not str(resolved).lower().startswith(str(CLEAN_ROOT).lower()):
            raise SystemExit(f"{label} must point inside {CLEAN_ROOT}; got {resolved}")
    if train_base.TORCH_IMPORT_ERROR is not None:
        raise SystemExit(f"PyTorch is required. Original error: {train_base.TORCH_IMPORT_ERROR}")
    random.seed(args.seed)
    np.random.seed(args.seed)
    train_base.torch.manual_seed(args.seed)
    if str(args.device).startswith("cuda") and not train_base.torch.cuda.is_available():
        raise RuntimeError(
            f"CUDA was requested with --device {args.device}, but this Python environment "
            "has no CUDA-enabled PyTorch installation. Refusing to fall back silently to CPU."
        )
    device = train_base.torch.device(args.device)
    run_dir = args.output_dir / args.run_name
    if run_dir.exists() and any(run_dir.iterdir()) and not args.force:
        raise SystemExit(f"Run directory exists and is not empty: {run_dir}; pass --force.")
    run_dir.mkdir(parents=True, exist_ok=True)

    manifest, paths = discover_shards(args)
    manifest["discriminator_label"] = build_target_labels(manifest, args)
    manifest = manifest.loc[manifest["discriminator_label"].ne("") & manifest["discriminator_label"].ne("nan")].copy()
    before_catch_all = len(manifest)
    if args.exclude_catch_all_labels:
        prefixes = tuple(value.strip().lower() for value in args.catch_all_label_prefix if value.strip())
        if prefixes:
            manifest = manifest.loc[~manifest["discriminator_label"].str.startswith(prefixes)].copy()
        if args.exclude_source_catch_all_labels and args.label_mode != "column" and "taxon_label" in manifest.columns:
            taxon_labels = normalized_label_text(manifest["taxon_label"])
            manifest = manifest.loc[~taxon_labels.str.startswith(prefixes)].copy()
        if before_catch_all - len(manifest):
            print(f"Excluded catch-all labels: {before_catch_all:,}->{len(manifest):,}", flush=True)
    excluded_labels = {value.strip().lower() for value in args.exclude_label if value.strip()}
    if excluded_labels:
        before = len(manifest)
        manifest = manifest.loc[~manifest["discriminator_label"].isin(excluded_labels)].copy()
        print(f"Excluded labels {sorted(excluded_labels)}: {before:,}->{len(manifest):,}", flush=True)
    raw_counts = manifest["discriminator_label"].value_counts()
    allowed = sorted(raw_counts[raw_counts >= int(args.min_class_train_samples)].index.tolist())
    label_to_index = {label: index for index, label in enumerate(allowed)}
    before_min = len(manifest)
    manifest = manifest.loc[manifest["discriminator_label"].isin(label_to_index)].copy()
    if before_min - len(manifest):
        print(
            f"Dropped labels below --min-class-train-samples={args.min_class_train_samples}: "
            f"{before_min:,}->{len(manifest):,}; retained_classes={len(label_to_index):,}",
            flush=True,
        )
    if not len(manifest):
        raise SystemExit("No records remain after label filtering.")
    manifest["label_index"] = manifest["discriminator_label"].map(label_to_index).astype(np.int64)
    manifest["sample_weight"] = compute_sample_weights(manifest, args)
    weights = manifest["sample_weight"].to_numpy(dtype=np.float32)
    print(
        "Sample weights: "
        f"mean={weights.mean():.3f}; min={weights.min():.3f}; p05={np.quantile(weights, 0.05):.3f}; "
        f"median={np.median(weights):.3f}; p95={np.quantile(weights, 0.95):.3f}; max={weights.max():.3f}",
        flush=True,
    )
    manifest = split_manifest(manifest, args)
    manifest = cap_eval_splits(manifest, args)
    train_manifest = manifest.loc[manifest["split"].eq("train")].reset_index(drop=True)
    val_manifest = manifest.loc[manifest["split"].eq("val")].reset_index(drop=True)
    test_manifest = manifest.loc[manifest["split"].eq("test")].reset_index(drop=True)
    print(
        f"Clean tree_id discriminator data: train={len(train_manifest):,}; val={len(val_manifest):,}; "
        f"test={len(test_manifest):,}; classes={len(label_to_index):,}; cities={manifest['city_token'].nunique():,}",
        flush=True,
    )

    store = CleanShardStore(paths, max_cached_shards=args.max_shard_cache)
    structure_mean, structure_std = estimate_array_standardizer(train_manifest, store, "structure", 0, args.seed)
    phenology_mean = phenology_std = None
    raw_sentinel_mean = raw_sentinel_std = None
    satellite_embedding_mean = satellite_embedding_std = None
    prism_mean = prism_std = None
    if args.use_sentinel_phenology:
        print(f"Using Sentinel phenology arrays from clean complete shards; standardizer samples={min(len(train_manifest), args.phenology_stat_samples):,}", flush=True)
        phenology_mean, phenology_std = estimate_array_standardizer(train_manifest, store, "sentinel_phenology", args.phenology_stat_samples, args.seed + 11)
    if args.use_raw_sentinel:
        print(f"Using raw Sentinel sequence arrays from clean complete shards; standardizer samples={min(len(train_manifest), args.raw_sentinel_stat_samples):,}", flush=True)
        raw_sentinel_mean, raw_sentinel_std = estimate_raw_sentinel_standardizer(train_manifest, store, args.raw_sentinel_stat_samples, args.seed + 17)
    if args.use_satellite_embedding:
        print(f"Using GEE/Satlas embedding arrays from clean complete shards; standardizer samples={min(len(train_manifest), args.satellite_embedding_stat_samples):,}", flush=True)
        satellite_embedding_mean, satellite_embedding_std = estimate_array_standardizer(
            train_manifest, store, "satellite_embedding", args.satellite_embedding_stat_samples, args.seed + 23
        )
    if args.use_prism_normals:
        print(f"Using PRISM normals arrays from clean complete shards; standardizer samples={min(len(train_manifest), args.prism_normals_stat_samples):,}", flush=True)
        prism_mean, prism_std = estimate_array_standardizer(train_manifest, store, "prism_normals", args.prism_normals_stat_samples, args.seed + 29)

    train_ds = CleanTreeDataset(
        train_manifest,
        store,
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
    )
    val_ds = CleanTreeDataset(
        val_manifest,
        store,
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
        preload_in_memory=args.cache_val_in_memory,
    )
    test_ds = CleanTreeDataset(
        test_manifest,
        store,
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
        preload_in_memory=args.cache_test_in_memory,
    )
    if args.num_workers > 0:
        store.clear()
        print("Cleared clean shard cache before DataLoader worker spawn.", flush=True)
    train_loader = make_loader(train_ds, args, train=True)
    val_loader = make_loader(val_ds, args, train=False)
    test_loader = make_loader(test_ds, args, train=False)

    model = CleanTreeDiscriminator(
        class_count=len(label_to_index),
        structure_dim=int(structure_mean.shape[0]),
        hidden_dim=args.hidden_dim,
        embedding_dim=args.embedding_dim,
        dropout=args.dropout,
        phenology_dim=0 if phenology_mean is None else int(phenology_mean.shape[0]),
        raw_sentinel_dim=0 if raw_sentinel_mean is None else int(raw_sentinel_mean.shape[0]),
        satellite_embedding_dim=0 if satellite_embedding_mean is None else int(satellite_embedding_mean.shape[0]),
        prism_normals_dim=0 if prism_mean is None else int(prism_mean.shape[0]),
        tree_image_branch_dropout=args.tree_image_branch_dropout,
    ).to(device)
    counts = np.bincount(train_manifest["label_index"].to_numpy(dtype=np.int64), minlength=len(label_to_index))
    ce_weights = train_base.torch.as_tensor(class_weights(counts, args), dtype=train_base.torch.float32, device=device)
    optimizer = train_base.torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}", flush=True)

    best_macro = -1.0
    best_epoch = 0
    no_improve = 0
    history: list[dict[str, Any]] = []
    best_path = run_dir / "best_model.pt"
    for epoch in range(1, args.epochs + 1):
        train_metrics = train_epoch(model, train_loader, optimizer, device, args, ce_weights)
        val_metrics = evaluate(model, val_loader, device, len(label_to_index), collect=False)
        improved = val_metrics["macro_f1"] > best_macro + float(args.early_stopping_min_delta)
        if improved:
            best_macro = val_metrics["macro_f1"]
            best_epoch = epoch
            no_improve = 0
            train_base.torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "label_to_index": label_to_index,
                    "structure_mean": train_base.torch.as_tensor(structure_mean),
                    "structure_std": train_base.torch.as_tensor(structure_std),
                    "phenology_mean": None if phenology_mean is None else train_base.torch.as_tensor(phenology_mean),
                    "phenology_std": None if phenology_std is None else train_base.torch.as_tensor(phenology_std),
                    "raw_sentinel_mean": None if raw_sentinel_mean is None else train_base.torch.as_tensor(raw_sentinel_mean),
                    "raw_sentinel_std": None if raw_sentinel_std is None else train_base.torch.as_tensor(raw_sentinel_std),
                    "satellite_embedding_mean": None if satellite_embedding_mean is None else train_base.torch.as_tensor(satellite_embedding_mean),
                    "satellite_embedding_std": None if satellite_embedding_std is None else train_base.torch.as_tensor(satellite_embedding_std),
                    "prism_mean": None if prism_mean is None else train_base.torch.as_tensor(prism_mean),
                    "prism_std": None if prism_std is None else train_base.torch.as_tensor(prism_std),
                    "args": safe_json_args(args),
                },
                best_path,
            )
        else:
            no_improve += 1
        row = {"epoch": epoch, **{f"train_{k}": v for k, v in train_metrics.items()}, **{f"val_{k}": v for k, v in val_metrics.items()}}
        history.append(row)
        gate_text = ",".join(f"{name}:{val_metrics.get(f'gate_{name}', 0.0):.3f}" for name in model.modality_names)
        print(
            f"epoch {epoch:03d}: train_loss={train_metrics['loss']:.4f}; train_supcon={train_metrics['supcon']:.4f}; "
            f"train_ce={train_metrics['ce']:.4f}; val_macro_f1={val_metrics['macro_f1']:.3f}; "
            f"val_weighted_f1={val_metrics['weighted_f1']:.3f}; gates={gate_text}; "
            f"best_val_macro_f1={best_macro:.3f}; improved={'yes' if improved else 'no'}; "
            f"no_improve={no_improve}/{args.early_stopping_patience}",
            flush=True,
        )
        pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)
        if no_improve >= args.early_stopping_patience:
            break

    if best_path.exists():
        checkpoint = train_base.torch.load(best_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
    test_metrics = evaluate(model, test_loader, device, len(label_to_index), collect=args.export_embeddings)
    summary = {
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_macro,
        "test_accuracy": test_metrics["accuracy"],
        "test_macro_f1": test_metrics["macro_f1"],
        "test_weighted_f1": test_metrics["weighted_f1"],
        "label_to_index": label_to_index,
        "args": safe_json_args(args),
    }
    (run_dir / "run_config.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if args.export_embeddings:
        np.savez_compressed(
            run_dir / "test_embeddings.npz",
            embeddings=test_metrics["embeddings"],
            labels=test_metrics["labels"],
            predictions=test_metrics["predictions"],
            city_token=test_metrics["city_token"],
            tree_id=test_metrics["tree_id"],
            crop_index=test_metrics["crop_index"],
            label_columns=np.asarray([label for label, _idx in sorted(label_to_index.items(), key=lambda kv: kv[1])], dtype=object),
        )
    print(
        f"Finished clean discriminator: best_epoch={best_epoch}; test_acc={test_metrics['accuracy']:.3f}; "
        f"test_macro_f1={test_metrics['macro_f1']:.3f}; test_weighted_f1={test_metrics['weighted_f1']:.3f}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
