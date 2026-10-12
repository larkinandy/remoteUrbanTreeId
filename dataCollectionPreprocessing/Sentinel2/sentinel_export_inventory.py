"""Raw Sentinel export discovery and completeness checks."""

from __future__ import annotations

import argparse
from datetime import date, timedelta
from pathlib import Path
import re
from typing import Any

import numpy as np
import pandas as pd

from sentinel_common import date_windows, normalize_token, parse_iso_date


def missing_raw_cell_path(args: argparse.Namespace, city: str) -> Path | None:
    root = Path(args.missing_raw_cell_root)
    candidates = [
        root / city / f"{city}_tree_centered_sentinel_cells_missing_raw15day.csv",
        root / f"{city}_tree_centered_sentinel_cells_missing_raw15day.csv",
    ]
    for path in candidates:
        if path.exists():
            return path
    matches = [
        path for path in sorted(root.glob("*/*_tree_centered_sentinel_cells_missing_raw15day.csv"))
        if normalize_token(path.parent.name) == city or normalize_token(path.name).startswith(city)
    ]
    matches.extend(
        path for path in sorted(root.glob("*_tree_centered_sentinel_cells_missing_raw15day.csv"))
        if normalize_token(path.name).startswith(city)
    )
    return matches[0] if matches else None


def raw_export_file_stems(city: str, start: date, end: date, interval_days: int, batch_count: int) -> set[str]:
    stems: set[str] = set()
    for window_start, window_end in date_windows(start, end, interval_days):
        inclusive_end = window_end - timedelta(days=1)
        for batch_index in range(batch_count):
            stems.add(
                f"tree_centered_s2_raw15day_{city}_{window_start:%Y%m%d}_"
                f"{inclusive_end:%Y%m%d}_batch_{batch_index:05d}"
            )
    return stems


def parse_raw_batch_index(path: Path) -> int | None:
    match = re.search(r"_batch_(\d{5})$", path.stem)
    return int(match.group(1)) if match else None


def raw_batches_for_row_ids(row_ids: set[int], batch_size: int) -> set[int]:
    if batch_size <= 0:
        raise ValueError("raw export batch size must be positive")
    return {max(0, (int(row_id) - 1) // int(batch_size)) for row_id in row_ids if row_id >= 0}


def complete_raw_paths_by_present_batch(
    city: str,
    paths: list[Path],
    prefix: str,
    start_date_value: str,
    end_date_value: str,
    interval_days: int,
) -> tuple[list[Path], dict[str, Any]]:
    if not paths:
        return [], {"checked": True, "expected": 0, "present": 0, "present_matching": 0,
                    "missing": 0, "complete_batches": 0, "incomplete_batches": 0}
    start = parse_iso_date(str(start_date_value), "raw export start date")
    end = parse_iso_date(str(end_date_value), "raw export end date")
    windows = date_windows(start, end, int(interval_days))
    present = {path.stem for path in paths}
    batch_indices = sorted({batch for batch in (parse_raw_batch_index(path) for path in paths) if batch is not None})
    complete_batches: set[int] = set()
    incomplete_batches: set[int] = set()
    expected_count = missing_count = 0
    for batch_index in batch_indices:
        batch_complete = True
        for window_start, window_end in windows:
            inclusive_end = window_end - timedelta(days=1)
            expected_count += 1
            stem = f"{prefix}_{city}_{window_start:%Y%m%d}_{inclusive_end:%Y%m%d}_batch_{batch_index:05d}"
            if stem not in present:
                batch_complete = False
                missing_count += 1
        (complete_batches if batch_complete else incomplete_batches).add(batch_index)
    filtered = [path for path in paths if parse_raw_batch_index(path) in complete_batches]
    return filtered, {
        "checked": True, "expected": int(expected_count), "present": int(len(present)),
        "present_matching": int(expected_count - missing_count), "missing": int(missing_count),
        "complete_batches": int(len(complete_batches)), "incomplete_batches": int(len(incomplete_batches)),
    }


def read_missing_raw_cell_batches(args: argparse.Namespace, city: str) -> tuple[Path, dict[int, int], int]:
    points_path = missing_raw_cell_path(args, city)
    if points_path is None:
        raise RuntimeError(f"{city}: no missing-cell CSV was found under {args.missing_raw_cell_root}")
    header = pd.read_csv(points_path, nrows=0).columns
    row_col = next((matches[0] for candidate in ("reduced_id", "row_index", "crown_reduced_id")
                    if (matches := [column for column in header if str(column).lower() == candidate])), None)
    if row_col is None:
        raise RuntimeError(f"{city}: {points_path} has no reduced_id/row_index column")
    cell_ids = pd.to_numeric(pd.read_csv(points_path, usecols=[row_col], low_memory=False)[row_col], errors="coerce").dropna()
    batch_size = int(args.raw_export_batch_size)
    batch_by_cell = {int(cell_id): int(position // batch_size)
                     for position, cell_id in enumerate(cell_ids.astype(np.int64).tolist())}
    batch_count = int(np.ceil(len(cell_ids) / batch_size)) if len(cell_ids) else 0
    return points_path, batch_by_cell, batch_count


def plan_complete_additional_raw_batches(
    args: argparse.Namespace,
    city: str,
    needed_additional: set[int],
    additional_paths: list[Path],
) -> tuple[set[int], set[int], list[Path], dict[str, Any]]:
    if args.skip_raw_file_completeness_check or not needed_additional:
        return needed_additional, set(), additional_paths, {
            "checked": False, "expected": 0, "present": len(additional_paths), "missing": 0,
            "complete_batches": 0, "incomplete_batches": 0, "excluded_additional_cells": 0,
        }
    points_path, batch_by_cell, batch_count = read_missing_raw_cell_batches(args, city)
    if batch_count <= 0:
        raise RuntimeError(f"{city}: missing-cell CSV is empty: {points_path}")
    start = parse_iso_date(str(args.raw_export_start_date), "--raw-export-start-date")
    end = parse_iso_date(str(args.raw_export_end_date), "--raw-export-end-date")
    windows = date_windows(start, end, int(args.raw_export_interval_days))
    present = {path.stem for path in additional_paths}
    complete_batches: set[int] = set()
    incomplete_batches: set[int] = set()
    missing_file_count = expected_count = 0
    for batch_index in range(batch_count):
        batch_complete = True
        for window_start, window_end in windows:
            inclusive_end = window_end - timedelta(days=1)
            expected_count += 1
            stem = (f"tree_centered_s2_raw15day_{city}_{window_start:%Y%m%d}_"
                    f"{inclusive_end:%Y%m%d}_batch_{batch_index:05d}")
            if stem not in present:
                batch_complete = False
                missing_file_count += 1
        (complete_batches if batch_complete else incomplete_batches).add(batch_index)
    needed_batches = {batch_by_cell[cell_id] for cell_id in needed_additional if cell_id in batch_by_cell}
    complete_needed_batches = complete_batches.intersection(needed_batches)
    complete_cells = {cell_id for cell_id in needed_additional if batch_by_cell.get(cell_id) in complete_needed_batches}
    excluded_cells = needed_additional.difference(complete_cells)
    filtered_paths = [path for path in additional_paths if parse_raw_batch_index(path) in complete_needed_batches]
    if incomplete_batches:
        print(f"{city}: supplemental raw Sentinel complete batches={len(complete_batches):,}/{batch_count:,}; "
              f"excluding {len(excluded_cells):,} crown cell(s) due to incomplete exports.", flush=True)
    status = {
        "checked": True, "expected": int(expected_count), "present": int(len(present)),
        "present_matching": int(expected_count - missing_file_count), "missing": int(missing_file_count),
        "complete_batches": int(len(complete_batches)), "incomplete_batches": int(len(incomplete_batches)),
        "needed_batches": int(len(needed_batches)), "complete_needed_batches": int(len(complete_needed_batches)),
        "excluded_additional_cells": int(len(excluded_cells)), "points_file": points_path.name,
    }
    return complete_cells, excluded_cells, filtered_paths, status
