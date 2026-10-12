"""Completeness screening applied before final shard assembly."""

from __future__ import annotations

from argparse import Namespace

import numpy as np
import pandas as pd

from dataCollectionPreprocessing.Shard.shard_common import boolean_array, finite_rows


def screen_complete_rows(
    metadata: pd.DataFrame,
    crops: np.ndarray,
    structure: dict[str, np.ndarray],
    structure_pos: np.ndarray,
    sentinel: dict[str, np.ndarray],
    sentinel_pos: np.ndarray,
    embedding: dict[str, np.ndarray],
    embedding_pos: np.ndarray,
    raw_sentinel: dict[str, np.ndarray],
    raw_sentinel_valid: np.ndarray,
    prism: dict[str, np.ndarray] | None,
    prism_pos: np.ndarray | None,
    args: Namespace,
) -> tuple[np.ndarray, dict[str, int]]:
    """Return complete-row mask and per-reason exclusion counts."""
    count = len(metadata)
    reasons: dict[str, np.ndarray] = {}
    crop_index = pd.to_numeric(metadata["crop_index"], errors="coerce").fillna(-1).astype(np.int64).to_numpy()
    reasons["missing_crop"] = (crop_index < 0) | (crop_index >= int(crops.shape[0]))
    reasons["qa_exclude_from_model"] = (
        boolean_array(metadata["qa_exclude_from_model"])
        if args.exclude_qa_from_model and "qa_exclude_from_model" in metadata.columns
        else np.zeros(count, dtype=bool)
    )
    reasons["missing_structure_row"] = structure_pos < 0
    reasons["missing_sentinel_row"] = sentinel_pos < 0
    reasons["missing_gee_row"] = embedding_pos < 0
    valid_structure = structure_pos.clip(min=0)
    valid_sentinel = sentinel_pos.clip(min=0)
    valid_embedding = embedding_pos.clip(min=0)

    reasons["missing_chm"] = (
        np.asarray(structure.get("missing_chm", np.zeros(len(valid_structure), dtype=bool))[valid_structure], dtype=bool)
        | reasons["missing_structure_row"]
    )
    if "tree_centered_chm_valid_mask" in structure:
        chm_valid = np.asarray(structure["tree_centered_chm_valid_mask"][valid_structure], dtype=bool)
        reasons["missing_chm"] |= ~chm_valid.reshape(chm_valid.shape[0], -1).any(axis=1)
    if "qa_flag_missing_lidar_chm" in metadata.columns:
        reasons["missing_chm"] |= boolean_array(metadata["qa_flag_missing_lidar_chm"])
    reasons["nonfinite_structure"] = (
        ~finite_rows(np.asarray(structure["tree_centered_naip_chm_structure"])[valid_structure])
        | reasons["missing_structure_row"]
    )
    reasons["missing_sentinel_phenology"] = (
        np.asarray(sentinel["missing_sentinel_phenology"][valid_sentinel], dtype=bool)
        | reasons["missing_sentinel_row"]
    )
    reasons["nonfinite_sentinel_phenology"] = (
        ~finite_rows(np.asarray(sentinel["sentinel_phenology"])[valid_sentinel])
        | reasons["missing_sentinel_row"]
    )
    reasons["missing_satellite_embedding"] = (
        np.asarray(embedding["missing_satellite_embedding"][valid_embedding], dtype=bool)
        | reasons["missing_gee_row"]
    )
    reasons["nonfinite_satellite_embedding"] = (
        ~finite_rows(np.asarray(embedding["satellite_embedding"])[valid_embedding])
        | reasons["missing_gee_row"]
    )
    if args.require_raw_sentinel:
        reasons["missing_raw_sentinel"] = ~raw_sentinel_valid
        reasons["nonfinite_raw_sentinel"] = ~finite_rows(
            np.asarray(raw_sentinel["sentinel_sequence"], dtype=np.float32)
        )
    if args.use_prism_daily:
        reasons["missing_prism_daily"] = np.asarray(
            raw_sentinel.get("missing_prism_daily", np.ones(count, dtype=bool)), dtype=bool
        )
    if args.require_prism_normals:
        if prism is None or prism_pos is None:
            reasons["missing_prism_normals"] = np.ones(count, dtype=bool)
        else:
            valid_prism = prism_pos.clip(min=0)
            reasons["missing_prism_normals"] = (
                np.asarray(prism["missing_prism_normals"][valid_prism], dtype=bool) | (prism_pos < 0)
            )
            reasons["nonfinite_prism_normals"] = (
                ~finite_rows(np.asarray(prism["prism_normals"])[valid_prism]) | (prism_pos < 0)
            )

    keep = np.ones(count, dtype=bool)
    for mask in reasons.values():
        keep &= ~mask
    return keep, {key: int(value.sum()) for key, value in reasons.items()}
