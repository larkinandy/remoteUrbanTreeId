#!/usr/bin/env python3
"""Audit satellite-embedding sidecars for schema and semantic consistency.

The command validates trusted local NPZ products and emits a compact city-level
CSV suitable for pipeline QA before multimodal shard assembly.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from satellite_embedding_utils import (
    EMBEDDING_COLUMNS,
    EMBEDDING_YEARS,
    QUALITY_COLUMNS,
    SOURCE_DIRECT_CROWN_CELL,
    SOURCE_MISSING,
    SOURCE_ORIGINAL_MATCH,
    normalize_city,
)


ROW_ARRAYS = [
    "tree_id",
    "crop_index",
    "row_index",
    "crown_cell_id",
    "crown_cell_epsg",
    "crown_cell_col",
    "crown_cell_row",
    "crown_cell_lat",
    "crown_cell_lon",
    "satellite_embedding",
    "satellite_embedding_mask",
    "satellite_embedding_quality",
    "embedding_source_code",
    "missing_satellite_embedding",
    "used_original_satellite_embedding",
    "used_additional_satellite_embedding",
]

EXPECTED_DTYPES = {
    "tree_id": np.dtype(np.int64),
    "crop_index": np.dtype(np.int64),
    "row_index": np.dtype(np.int64),
    "crown_cell_epsg": np.dtype(np.int32),
    "crown_cell_col": np.dtype(np.int64),
    "crown_cell_row": np.dtype(np.int64),
    "crown_cell_lat": np.dtype(np.float64),
    "crown_cell_lon": np.dtype(np.float64),
    "satellite_embedding": np.dtype(np.float32),
    "satellite_embedding_mask": np.dtype(np.float32),
    "satellite_embedding_quality": np.dtype(np.float32),
    "embedding_source_code": np.dtype(np.int8),
    "missing_satellite_embedding": np.dtype(bool),
    "used_original_satellite_embedding": np.dtype(bool),
    "used_additional_satellite_embedding": np.dtype(bool),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--sidecar-root", type=Path, required=True)
    parser.add_argument("--pattern", default="*/*_tree_id_centered_gee_inputs.npz")
    parser.add_argument("--city-token", action="append", default=[])
    parser.add_argument("--expected-years", nargs="+", type=int, default=EMBEDDING_YEARS)
    parser.add_argument("--summary", type=Path, help="Optional CSV validation summary.")
    return parser.parse_args()


def add_error(errors: list[str], condition: bool, message: str) -> None:
    if condition:
        errors.append(message)


def check_array_contract(data: Any, rows: int, errors: list[str], warnings: list[str]) -> None:
    """Check row alignment and declared storage dtypes."""
    for name in ROW_ARRAYS:
        add_error(errors, np.asarray(data[name]).shape[0] != rows, f"{name} row count differs")
    for name, expected_dtype in EXPECTED_DTYPES.items():
        actual = np.asarray(data[name]).dtype
        add_error(errors, actual != expected_dtype, f"{name} dtype is {actual}, expected {expected_dtype}")
    if np.asarray(data["crown_cell_id"]).dtype.kind == "O":
        warnings.append("crown_cell_id uses legacy object encoding")


def check_semantics(data: Any, rows: int, expected_years: list[int], errors: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Check dimensions, masks, quality values, identifiers, and source flags."""
    embedding = np.asarray(data["satellite_embedding"])
    mask = np.asarray(data["satellite_embedding_mask"])
    quality = np.asarray(data["satellite_embedding_quality"])
    years = np.asarray(data["satellite_embedding_years"], dtype=np.int32)
    missing = np.asarray(data["missing_satellite_embedding"], dtype=bool)
    source = np.asarray(data["embedding_source_code"])
    original = np.asarray(data["used_original_satellite_embedding"], dtype=bool)
    direct = np.asarray(data["used_additional_satellite_embedding"], dtype=bool)

    add_error(errors, embedding.shape != (rows, len(years), 64), f"satellite_embedding shape is {embedding.shape}")
    add_error(errors, mask.shape != (rows, len(years)), f"satellite_embedding_mask shape is {mask.shape}")
    add_error(errors, quality.shape != (rows, len(QUALITY_COLUMNS)), f"satellite_embedding_quality shape is {quality.shape}")
    add_error(errors, years.tolist() != expected_years, f"years are {years.tolist()}, expected {expected_years}")
    add_error(errors, np.asarray(data["satellite_embedding_columns"]).astype(str).tolist() != EMBEDDING_COLUMNS, "embedding columns are not A00-A63")
    add_error(errors, np.asarray(data["satellite_embedding_quality_columns"]).astype(str).tolist() != QUALITY_COLUMNS, "quality column names differ from schema")
    add_error(errors, not np.isfinite(embedding).all() or not np.isfinite(quality).all(), "embedding or quality contains non-finite values")
    add_error(errors, not np.isin(mask, [0.0, 1.0]).all(), "year mask is not binary")
    add_error(errors, not np.array_equal(missing, mask.sum(axis=1) == 0), "missing flag disagrees with year mask")
    add_error(errors, not np.array_equal(quality[:, 0], mask.sum(axis=1)), "year-count quality metric disagrees with mask")
    add_error(errors, not np.isin(source, [SOURCE_MISSING, SOURCE_ORIGINAL_MATCH, SOURCE_DIRECT_CROWN_CELL]).all(), "source code is outside 0, 1, 2")
    add_error(errors, np.any(original & direct), "rows are marked as both original and direct")
    add_error(errors, np.any((source == SOURCE_MISSING) != missing), "source code 0 disagrees with missing flag")
    add_error(errors, np.any((source == SOURCE_ORIGINAL_MATCH) != original), "source code 1 disagrees with original flag")
    add_error(errors, np.any((source == SOURCE_DIRECT_CROWN_CELL) != direct), "source code 2 disagrees with direct flag")
    add_error(errors, len(np.unique(np.asarray(data["tree_id"]))) != rows, "tree_id is not unique")
    add_error(errors, len(np.unique(np.asarray(data["crop_index"]))) != rows, "crop_index is not unique")
    return mask, missing


def check_config(data: Any, errors: list[str], warnings: list[str]) -> None:
    """Validate embedded JSON and warn about workstation-specific paths."""
    try:
        config = json.loads(str(np.asarray(data["config_json"]).item()))
    except json.JSONDecodeError:
        errors.append("config_json is invalid JSON")
        return
    if any(":\\" in str(value) or str(value).startswith("/") for value in config.values()):
        warnings.append("config_json contains an absolute path")


def validate_sidecar(path: Path, expected_years: list[int]) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    # Frozen study sidecars contain an object-encoded crown_cell_id array.
    # Sidecars are trusted local pipeline products; new outputs use Unicode.
    with np.load(path, allow_pickle=True) as data:
        missing_arrays = [name for name in [*ROW_ARRAYS, "satellite_embedding_columns", "satellite_embedding_years", "satellite_embedding_quality_columns", "config_json"] if name not in data.files]
        if missing_arrays:
            return {"path": str(path), "status": "failed", "rows": 0, "errors": "; ".join(f"missing {name}" for name in missing_arrays), "warnings": ""}

        rows = int(np.asarray(data["tree_id"]).shape[0])
        check_array_contract(data, rows, errors, warnings)
        mask, missing = check_semantics(data, rows, expected_years, errors)
        check_config(data, errors, warnings)

    return {
        "city_token": normalize_city(path.parent.name),
        "path": str(path),
        "status": "passed" if not errors else "failed",
        "rows": rows,
        "missing_rows": int(missing.sum()),
        "complete_three_year_rows": int((mask.sum(axis=1) == len(expected_years)).sum()),
        "errors": "; ".join(errors),
        "warnings": "; ".join(warnings),
    }


def main() -> int:
    args = parse_args()
    if not args.sidecar_root.is_dir():
        raise FileNotFoundError(args.sidecar_root)
    selected = {normalize_city(value) for value in args.city_token}
    paths = sorted(args.sidecar_root.glob(args.pattern))
    if selected:
        paths = [path for path in paths if normalize_city(path.parent.name) in selected]
    if not paths:
        raise FileNotFoundError(f"No sidecars matching {args.pattern!r} under {args.sidecar_root}")

    results = [validate_sidecar(path, list(args.expected_years)) for path in paths]
    summary = pd.DataFrame(results)
    if args.summary:
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(args.summary, index=False)
    print(summary[["city_token", "status", "rows", "missing_rows", "errors", "warnings"]].to_string(index=False), flush=True)
    failures = int((summary["status"] == "failed").sum())
    print(f"Validated {len(summary):,} sidecar(s); failures={failures:,}", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
