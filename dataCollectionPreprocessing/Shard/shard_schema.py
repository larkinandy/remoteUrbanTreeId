"""Canonical validation contract for final model-input shards."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd


REQUIRED_ARRAY_DTYPES: dict[str, np.dtype[Any]] = {
    "tree_centered_naip": np.dtype(np.uint8),
    "tree_centered_chm": np.dtype(np.float32),
    "tree_centered_chm_valid_mask": np.dtype(bool),
    "tree_centered_vegetation_chm": np.dtype(np.float32),
    "tree_centered_vegetation_chm_weight": np.dtype(np.float32),
    "tree_centered_naip_chm_structure": np.dtype(np.float32),
    "sentinel_phenology": np.dtype(np.float32),
    "sentinel_phenology_source_code": np.dtype(np.int8),
    "sentinel_timeseries_match_distance_m": np.dtype(np.float32),
    "satellite_embedding": np.dtype(np.float32),
    "satellite_embedding_mask": np.dtype(np.float32),
    "satellite_embedding_quality": np.dtype(np.float32),
    "embedding_source_code": np.dtype(np.int8),
    "used_original_satellite_embedding": np.dtype(bool),
    "used_additional_satellite_embedding": np.dtype(bool),
    "sentinel_sequence": np.dtype(np.float32),
    "sentinel_sequence_mask": np.dtype(bool),
    "sentinel_sequence_dates_yyyymmdd": np.dtype(np.int32),
    "sentinel_sequence_length": np.dtype(np.int16),
    "sentinel_quality": np.dtype(np.float32),
    "raw_sentinel_source_code": np.dtype(np.int8),
    "missing_raw_sentinel": np.dtype(bool),
    "tree_id": np.dtype(np.int64),
    "tree_centered_index": np.dtype(np.int64),
    "crop_index": np.dtype(np.int64),
    "row_index": np.dtype(np.int64),
}

ROW_ALIGNED_OPTIONAL = {
    "sentinel_prism_daily",
    "sentinel_prism_daily_mask",
    "missing_prism_daily",
    "prism_normals",
    "missing_prism_normals",
}


def validate_shard_arrays(arrays: Mapping[str, Any], metadata: pd.DataFrame) -> None:
    """Fail before writing when the canonical row/dtype contract is violated."""
    missing = sorted(set(REQUIRED_ARRAY_DTYPES).difference(arrays))
    if missing:
        raise RuntimeError(f"Shard is missing required array(s): {missing}")
    rows = len(metadata)
    if "tree_id" not in metadata.columns:
        raise RuntimeError("Shard metadata is missing tree_id")
    for name, expected_dtype in REQUIRED_ARRAY_DTYPES.items():
        value = np.asarray(arrays[name])
        if value.ndim == 0 or value.shape[0] != rows:
            raise RuntimeError(f"{name} has {value.shape[0] if value.ndim else 0} rows; expected {rows}")
        if value.dtype != expected_dtype:
            raise RuntimeError(f"{name} has dtype {value.dtype}; expected {expected_dtype}")
    for name in sorted(ROW_ALIGNED_OPTIONAL.intersection(arrays)):
        value = np.asarray(arrays[name])
        if value.ndim == 0 or value.shape[0] != rows:
            raise RuntimeError(f"{name} is not row-aligned with metadata ({rows} rows)")
    array_ids = np.asarray(arrays["tree_id"], dtype=np.int64)
    metadata_ids = pd.to_numeric(metadata["tree_id"], errors="raise").to_numpy(dtype=np.int64)
    if not np.array_equal(array_ids, metadata_ids):
        raise RuntimeError("tree_id order differs between the NPZ arrays and metadata CSV")
    if np.unique(array_ids).size != rows:
        raise RuntimeError("tree_id values are not unique within the shard")
