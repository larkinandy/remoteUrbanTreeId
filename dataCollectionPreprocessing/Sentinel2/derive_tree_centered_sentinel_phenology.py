#!/usr/bin/env python3
"""Derive standalone tree-centered Sentinel phenology sidecars.

Rows are keyed by the tree-centered record index. Original cell-centered
interpolated time series can be used for crown cells already covered by the old
Sentinel products, while additional tree-centered raw 15-day downloads are used
for crown cells outside the original download. The output sidecar is standalone:
downstream shard assembly only needs this NPZ and the tree-centered record
index, not the old cell-centered model inputs.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta
import json
from pathlib import Path
import re
import sys
import time
from typing import Any

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
SHARD_SCRIPTS = HERE.parent / "Shard"
if str(SHARD_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SHARD_SCRIPTS))

import assemble_clean_tree_id_centered_model_input_shards as shard_schema
import interpolate_sentinel_cell_time_series as interpolated_sentinel
import sentinel_cell_features as sentinel_base
import sentinel_phenology_metrics as derived_metrics


DEFAULT_RECORD_INDEX_ROOT = Path(r"E:\TreeCenteredModelInputs\tree_centered_record_index")
DEFAULT_ORIGINAL_TIMESERIES_DIR = Path(r"E:\cell\sentinel2_timeseries")
DEFAULT_SUPPLEMENTAL_TIMESERIES_DIR = Path(r"E:\TreeCenteredModelInputs\tree_centered_sentinel_timeseries_supplemental")
DEFAULT_ORIGINAL_RAW_SENTINEL_DIR = Path(r"E:\TreeId\Sentinel2")
DEFAULT_ADDITIONAL_RAW_SENTINEL_DIR = Path(r"E:\TreeCenterSentinel")
DEFAULT_OUTPUT_ROOT = Path(r"E:\TreeCenteredModelInputs\tree_centered_sentinel_phenology")
DEFAULT_TIMESERIES_OUTPUT_ROOT = Path(r"E:\TreeCenteredModelInputs\tree_centered_sentinel_timeseries")
DEFAULT_MISSING_RAW_CELL_ROOT = Path(r"E:\TreeCenteredModelInputs\tree_centered_sentinel_missing_raw15day")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--record-index-root", type=Path, default=DEFAULT_RECORD_INDEX_ROOT)
    parser.add_argument("--original-timeseries-dir", type=Path, default=DEFAULT_ORIGINAL_TIMESERIES_DIR)
    parser.add_argument("--supplemental-timeseries-dir", type=Path, default=DEFAULT_SUPPLEMENTAL_TIMESERIES_DIR)
    parser.add_argument("--original-raw-sentinel-dir", type=Path, default=DEFAULT_ORIGINAL_RAW_SENTINEL_DIR)
    parser.add_argument("--additional-raw-sentinel-dir", type=Path, default=DEFAULT_ADDITIONAL_RAW_SENTINEL_DIR)
    parser.add_argument("--missing-raw-cell-root", type=Path, default=DEFAULT_MISSING_RAW_CELL_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--timeseries-output-root", type=Path, default=DEFAULT_TIMESERIES_OUTPUT_ROOT)
    parser.add_argument("--stage", choices=["interpolate", "compute", "all"], default="all")
    parser.add_argument(
        "--prefer-separate-timeseries",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "In compute mode, prefer original/supplemental interpolated time-series folders even if a "
            "combined tree-centered time-series CSV exists."
        ),
    )
    parser.add_argument("--city-token", action="append", default=None)
    parser.add_argument("--exclude-city-token", action="append", default=[])
    parser.add_argument("--record-index-pattern", default="*_tree_centered_record_index.csv")
    parser.add_argument("--original-timeseries-pattern", default="**/*_sentinel2_*_time_series_batch_*.csv")
    parser.add_argument("--supplemental-timeseries-pattern", default="**/*_sentinel2_*_time_series_batch_*.csv")
    parser.add_argument("--original-raw-pattern", default="s2_reduced_cells_*_batch_*.csv")
    parser.add_argument("--additional-raw-pattern", default="tree_centered_s2_raw15day_*_batch_*.csv")
    parser.add_argument("--original-raw-export-start-date", default="2021-01-01")
    parser.add_argument("--original-raw-export-end-date", default="2024-01-01")
    parser.add_argument("--original-raw-export-interval-days", type=int, default=30)
    parser.add_argument("--original-raw-export-batch-size", type=int, default=5000)
    parser.add_argument("--raw-export-start-date", default="2021-01-01")
    parser.add_argument("--raw-export-end-date", default="2024-01-01")
    parser.add_argument("--raw-export-interval-days", type=int, default=15)
    parser.add_argument("--raw-export-batch-size", type=int, default=5000)
    parser.add_argument("--gap-threshold-days", type=int, default=10)
    parser.add_argument("--interpolation-interval-days", type=int, default=15)
    parser.add_argument("--sentinel-outlier-abs", type=float, default=10000.0)
    parser.add_argument("--chunksize", type=int, default=250000)
    parser.add_argument(
        "--raw-load-mode",
        choices=["batch", "bulk", "chunked"],
        default="batch",
        help=(
            "How to load raw Sentinel CSVs. batch processes one export batch at a "
            "time, matching the original interpolation workflow. bulk reads all "
            "selected files into memory before joining. chunked filters during "
            "read_csv and uses less RAM but is usually slower."
        ),
    )
    parser.add_argument(
        "--daily-aggregation-mode",
        choices=["fast", "exact"],
        default="fast",
        help=(
            "Daily raw Sentinel aggregation. fast groups by row_index/date only, "
            "uses source image count rather than nunique, and keeps first SCL. "
            "exact calls the original shared aggregator."
        ),
    )
    parser.add_argument("--parallel-workers", type=int, default=2)
    parser.add_argument("--progress-interval", type=int, default=10000)
    parser.add_argument(
        "--allow-missing-sentinel-phenology",
        action="store_true",
        help=(
            "Allow existing sidecars with missing_sentinel_phenology rows to be considered complete. "
            "By default, existing sidecars are only skipped when every row has phenology."
        ),
    )
    parser.add_argument(
        "--skip-raw-file-completeness-check",
        action="store_true",
        help=(
            "Do not require the supplemental tree-centered raw Sentinel CSV export set "
            "to contain every expected date-window/batch file before deriving phenology."
        ),
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def normalize_token(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value or "").lower())


def finite_int(value: object) -> int | None:
    if pd.isna(value):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(number):
        return None
    return int(number)


def discover_record_indexes(args: argparse.Namespace) -> dict[str, Path]:
    selected = None if args.city_token is None else {normalize_token(v) for v in args.city_token if str(v).strip()}
    excluded = {normalize_token(v) for v in args.exclude_city_token if str(v).strip()}
    paths = sorted(Path(args.record_index_root).glob(f"*/{args.record_index_pattern}"))
    paths.extend(sorted(Path(args.record_index_root).glob(args.record_index_pattern)))
    out: dict[str, Path] = {}
    for path in paths:
        city = normalize_token(path.parent.name)
        match = re.match(r"(.+?)_tree_centered_record_index$", path.stem)
        if match:
            city = normalize_token(match.group(1))
        if selected is not None and city not in selected:
            continue
        if city in excluded:
            continue
        out[city] = path
    if not out:
        raise FileNotFoundError(f"No record indexes found under {args.record_index_root}")
    return out


def city_paths(root: Path, pattern: str, city: str) -> list[Path]:
    if not Path(root).exists():
        return []
    return [path for path in sorted(Path(root).glob(pattern)) if city in normalize_token(path.name) or city == normalize_token(path.parent.name)]


def parse_date_arg(value: str, label: str) -> date:
    try:
        return datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError as exc:
        raise ValueError(f"{label} must be YYYY-MM-DD: {value!r}") from exc


def date_windows(start: date, end: date, interval_days: int) -> list[tuple[date, date]]:
    if end < start:
        raise ValueError("--raw-export-end-date must not precede --raw-export-start-date")
    if interval_days <= 0:
        raise ValueError("--raw-export-interval-days must be positive")
    current = start
    exclusive_end = end + timedelta(days=1)
    out: list[tuple[date, date]] = []
    while current < exclusive_end:
        window_end = min(current + timedelta(days=interval_days), exclusive_end)
        out.append((current, window_end))
        current = window_end
    return out


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
        path
        for path in sorted(root.glob("*/*_tree_centered_sentinel_cells_missing_raw15day.csv"))
        if normalize_token(path.parent.name) == city or normalize_token(path.name).startswith(city)
    ]
    matches.extend(
        path
        for path in sorted(root.glob("*_tree_centered_sentinel_cells_missing_raw15day.csv"))
        if normalize_token(path.name).startswith(city)
    )
    return matches[0] if matches else None


def raw_export_file_stems(city: str, start: date, end: date, interval_days: int, batch_count: int) -> set[str]:
    stems: set[str] = set()
    for window_start, window_end in date_windows(start, end, interval_days):
        inclusive_end = window_end - timedelta(days=1)
        for batch_index in range(batch_count):
            stems.add(
                f"tree_centered_s2_raw15day_{city}_"
                f"{window_start:%Y%m%d}_{inclusive_end:%Y%m%d}_"
                f"batch_{batch_index:05d}"
            )
    return stems


def parse_raw_batch_index(path: Path) -> int | None:
    match = re.search(r"_batch_(\d{5})$", path.stem)
    return int(match.group(1)) if match else None


def raw_batches_for_row_ids(row_ids: set[int], batch_size: int) -> set[int]:
    if batch_size <= 0:
        raise ValueError("raw export batch size must be positive")
    batches: set[int] = set()
    for row_id in row_ids:
        if row_id < 0:
            continue
        # Reduced-cell ids are 1-based in the cell map, while export batches are 0-based.
        batches.add(max(0, (int(row_id) - 1) // int(batch_size)))
    return batches


def complete_raw_paths_by_present_batch(
    city: str,
    paths: list[Path],
    prefix: str,
    start_date_value: str,
    end_date_value: str,
    interval_days: int,
) -> tuple[list[Path], dict[str, Any]]:
    if not paths:
        return [], {
            "checked": True,
            "expected": 0,
            "present": 0,
            "present_matching": 0,
            "missing": 0,
            "complete_batches": 0,
            "incomplete_batches": 0,
        }
    start = parse_date_arg(str(start_date_value), "raw export start date")
    end = parse_date_arg(str(end_date_value), "raw export end date")
    windows = date_windows(start, end, int(interval_days))
    present = {path.stem for path in paths}
    batch_indices = sorted({batch for batch in (parse_raw_batch_index(path) for path in paths) if batch is not None})
    complete_batches: set[int] = set()
    incomplete_batches: set[int] = set()
    expected_count = 0
    missing_count = 0
    for batch_index in batch_indices:
        batch_complete = True
        for window_start, window_end in windows:
            inclusive_end = window_end - timedelta(days=1)
            expected_count += 1
            stem = (
                f"{prefix}_{city}_"
                f"{window_start:%Y%m%d}_{inclusive_end:%Y%m%d}_"
                f"batch_{batch_index:05d}"
            )
            if stem not in present:
                batch_complete = False
                missing_count += 1
        if batch_complete:
            complete_batches.add(batch_index)
        else:
            incomplete_batches.add(batch_index)
    filtered = [path for path in paths if parse_raw_batch_index(path) in complete_batches]
    return filtered, {
        "checked": True,
        "expected": int(expected_count),
        "present": int(len(present)),
        "present_matching": int(expected_count - missing_count),
        "missing": int(missing_count),
        "complete_batches": int(len(complete_batches)),
        "incomplete_batches": int(len(incomplete_batches)),
    }


def read_missing_raw_cell_batches(args: argparse.Namespace, city: str) -> tuple[Path, dict[int, int], int]:
    points_path = missing_raw_cell_path(args, city)
    if points_path is None:
        raise RuntimeError(f"{city}: no missing-cell CSV was found under {args.missing_raw_cell_root}")
    header = pd.read_csv(points_path, nrows=0).columns
    row_col = None
    for candidate in ("reduced_id", "row_index", "crown_reduced_id"):
        matches = [column for column in header if str(column).lower() == candidate]
        if matches:
            row_col = matches[0]
            break
    if row_col is None:
        raise RuntimeError(f"{city}: {points_path} has no reduced_id/row_index column")
    table = pd.read_csv(points_path, usecols=[row_col], low_memory=False)
    cell_ids = pd.to_numeric(table[row_col], errors="coerce")
    valid = cell_ids.notna()
    if not bool(valid.all()):
        table = table.loc[valid].copy()
        cell_ids = cell_ids.loc[valid]
    batch_size = int(args.raw_export_batch_size)
    batch_by_cell: dict[int, int] = {}
    for position, cell_id in enumerate(cell_ids.astype(np.int64).tolist()):
        batch_by_cell[int(cell_id)] = int(position // batch_size)
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
            "checked": False,
            "expected": 0,
            "present": len(additional_paths),
            "missing": 0,
            "complete_batches": 0,
            "incomplete_batches": 0,
            "excluded_additional_cells": 0,
        }
    points_path, batch_by_cell, batch_count = read_missing_raw_cell_batches(args, city)
    if batch_count <= 0:
        raise RuntimeError(f"{city}: missing-cell CSV is empty: {points_path}")
    start = parse_date_arg(str(args.raw_export_start_date), "--raw-export-start-date")
    end = parse_date_arg(str(args.raw_export_end_date), "--raw-export-end-date")
    windows = date_windows(start, end, int(args.raw_export_interval_days))
    present = {path.stem for path in additional_paths}
    complete_batches: set[int] = set()
    incomplete_batches: set[int] = set()
    missing_file_count = 0
    expected_count = 0
    for batch_index in range(batch_count):
        batch_complete = True
        for window_start, window_end in windows:
            inclusive_end = window_end - timedelta(days=1)
            expected_count += 1
            stem = (
                f"tree_centered_s2_raw15day_{city}_"
                f"{window_start:%Y%m%d}_{inclusive_end:%Y%m%d}_"
                f"batch_{batch_index:05d}"
            )
            if stem not in present:
                batch_complete = False
                missing_file_count += 1
        if batch_complete:
            complete_batches.add(batch_index)
        else:
            incomplete_batches.add(batch_index)
    needed_batches = {batch_by_cell[cell_id] for cell_id in needed_additional if cell_id in batch_by_cell}
    complete_needed_batches = complete_batches.intersection(needed_batches)
    complete_cells = {cell_id for cell_id in needed_additional if batch_by_cell.get(cell_id) in complete_needed_batches}
    excluded_cells = needed_additional.difference(complete_cells)
    filtered_paths = [path for path in additional_paths if parse_raw_batch_index(path) in complete_needed_batches]
    if incomplete_batches:
        print(
            f"{city}: supplemental raw Sentinel complete batches={len(complete_batches):,}/{batch_count:,}; "
            f"excluding {len(excluded_cells):,} crown cell(s) from supplemental phenology due to incomplete batch exports.",
            flush=True,
        )
    status = {
        "checked": True,
        "expected": int(expected_count),
        "present": int(len(present)),
        "present_matching": int(expected_count - missing_file_count),
        "missing": int(missing_file_count),
        "complete_batches": int(len(complete_batches)),
        "incomplete_batches": int(len(incomplete_batches)),
        "needed_batches": int(len(needed_batches)),
        "complete_needed_batches": int(len(complete_needed_batches)),
        "excluded_additional_cells": int(len(excluded_cells)),
        "points_file": str(points_path),
    }
    return complete_cells, excluded_cells, filtered_paths, status


def read_interpolated_timeseries(paths: list[Path], needed_rows: set[int], chunksize: int) -> dict[int, pd.DataFrame]:
    if not paths or not needed_rows:
        return {}
    wanted = ["row_index", "date"] + list(shard_schema.SENTINEL_SEQUENCE_COLUMNS)
    pieces: list[pd.DataFrame] = []
    for path in paths:
        header = set(pd.read_csv(path, nrows=0).columns)
        missing = set(wanted).difference(header)
        if missing:
            raise ValueError(f"{path} is missing Sentinel time-series columns: {sorted(missing)}")
        for chunk in pd.read_csv(path, usecols=wanted, chunksize=chunksize, low_memory=False):
            chunk["row_index"] = pd.to_numeric(chunk["row_index"], errors="coerce")
            chunk = chunk.loc[chunk["row_index"].isin(needed_rows)].copy()
            if chunk.empty:
                continue
            chunk["row_index"] = chunk["row_index"].astype(np.int64)
            chunk["date"] = pd.to_datetime(chunk["date"], errors="coerce")
            chunk = chunk.dropna(subset=["date"])
            pieces.append(chunk)
    if not pieces:
        return {}
    frame = pd.concat(pieces, ignore_index=True)
    frame = frame.sort_values(["row_index", "date"], kind="stable").drop_duplicates(["row_index", "date"], keep="first")
    for column in shard_schema.SENTINEL_SEQUENCE_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return {int(row_index): group for row_index, group in frame.groupby("row_index", sort=False)}


def load_separate_interpolated_groups(
    args: argparse.Namespace,
    city: str,
    needed_original: set[int],
    needed_additional: set[int],
) -> tuple[dict[int, pd.DataFrame], dict[int, pd.DataFrame], list[Path], list[Path]]:
    original_paths = city_paths(args.original_timeseries_dir, args.original_timeseries_pattern, city)
    supplemental_paths = city_paths(args.supplemental_timeseries_dir, args.supplemental_timeseries_pattern, city)
    started = time.perf_counter()
    original_groups = read_interpolated_timeseries(original_paths, needed_original, int(args.chunksize))
    print(
        f"{city}: loaded original interpolated Sentinel groups={len(original_groups):,}/"
        f"{len(needed_original):,} from {len(original_paths):,} file(s); "
        f"elapsed={time.perf_counter() - started:.1f}s",
        flush=True,
    )
    started = time.perf_counter()
    additional_groups = read_interpolated_timeseries(supplemental_paths, needed_additional, int(args.chunksize))
    print(
        f"{city}: loaded supplemental interpolated Sentinel groups={len(additional_groups):,}/"
        f"{len(needed_additional):,} from {len(supplemental_paths):,} file(s); "
        f"elapsed={time.perf_counter() - started:.1f}s",
        flush=True,
    )
    return original_groups, additional_groups, original_paths, supplemental_paths


def parse_tree_centered_raw_export(path: Path) -> tuple[int, int] | None:
    match = re.search(r"_(?P<start>\d{8})_(?P<end>\d{8})_batch_(?P<batch>\d{5})", path.stem)
    if not match:
        return None
    return int(match.group("batch")), int(match.group("start"))


def raw_wanted_columns(path: Path) -> list[str]:
    header = pd.read_csv(path, nrows=0).columns.tolist()
    wanted = {
        "row_index",
        "reduced_id",
        "date",
        "datetime",
        "acquisition_time",
        "source_image_id",
        "source_item_id",
        "image_id",
        "system:index",
        "mgrs_tile",
        "valid_pixel",
        "latitude",
        "lat",
        "longitude",
        "lon",
        "scl",
        *{column.lower() for column in sentinel_base.S2_RAW_COLS},
    }
    return [column for column in header if column.lower() in wanted]


def raw_column_plan(path: Path) -> tuple[list[str], str]:
    usecols = raw_wanted_columns(path)
    lower_lookup = {column.lower(): column for column in usecols}
    row_col = lower_lookup.get("row_index") or lower_lookup.get("reduced_id")
    if row_col is None:
        raise ValueError(f"{path} is missing row_index/reduced_id")
    return usecols, row_col


def raw_paths_by_batch(paths: list[Path]) -> dict[int, list[Path]]:
    out: dict[int, list[Path]] = {}
    for path in paths:
        parsed = parse_tree_centered_raw_export(path)
        batch_index = parsed[0] if parsed else 0
        out.setdefault(int(batch_index), []).append(path)
    return {batch: sorted(batch_paths) for batch, batch_paths in sorted(out.items())}


def fast_aggregate_daily(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    x = frame.copy()
    if "source_image_id" not in x.columns:
        x["source_image_id"] = "unknown"
    aggregations: dict[str, object] = {column: "mean" for column in sentinel_base.S2_COLS}
    aggregations.update({"acquisition_time": "min", "source_image_id": "count"})
    if "SCL" in x.columns:
        aggregations["SCL"] = "first"
    if "latitude" in x.columns:
        aggregations["latitude"] = "first"
    if "longitude" in x.columns:
        aggregations["longitude"] = "first"

    grouped = x.groupby(["row_index", "date"], sort=False, as_index=False).agg(aggregations)
    grouped = grouped.rename(columns={"source_image_id": "source_image_count"})
    grouped.insert(0, "city_token", str(x["city_token"].iloc[0]) if "city_token" in x.columns and len(x) else "")
    grouped["acquisition_time"] = pd.to_datetime(grouped["acquisition_time"], utc=True, errors="coerce")
    grouped = grouped.sort_values(["row_index", "acquisition_time"], kind="stable")
    doy = grouped["acquisition_time"].dt.dayofyear.astype("float32")
    grouped["doy_sin"] = np.sin(2.0 * np.pi * doy / 365.25).astype("float32")
    grouped["doy_cos"] = np.cos(2.0 * np.pi * doy / 365.25).astype("float32")
    grouped["delta_days"] = (
        grouped.groupby("row_index", sort=False)["acquisition_time"]
        .diff()
        .dt.days
        .fillna(0)
        .astype("float32")
    )
    return grouped


def aggregate_daily_for_args(raw: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    if str(getattr(args, "daily_aggregation_mode", "fast")) == "exact":
        return sentinel_base.aggregate_daily(raw)
    return fast_aggregate_daily(raw)


def normalize_raw_chunk_for_needed(
    chunk: pd.DataFrame,
    city: str,
    source_path: Path,
    needed_rows: set[int] | None,
) -> pd.DataFrame:
    lower_lookup = {str(column).lower(): column for column in chunk.columns}

    def find_column(candidates: tuple[str, ...]) -> str | None:
        for candidate in candidates:
            found = lower_lookup.get(candidate.lower())
            if found is not None:
                return found
        return None

    row_col = find_column(("row_index", "reduced_id"))
    time_col = find_column(("datetime", "acquisition_time", "date"))
    if row_col is None or time_col is None:
        raise ValueError(f"{source_path} is missing row_index/reduced_id or acquisition time")

    row_values = pd.to_numeric(chunk[row_col], errors="coerce")
    keep = row_values.notna()
    if needed_rows is not None:
        keep &= row_values.isin(needed_rows)
    if not bool(keep.any()):
        return pd.DataFrame()

    frame = chunk.loc[keep].copy()
    frame["city_token"] = city
    frame["row_index"] = row_values.loc[keep].astype("int64").to_numpy()
    frame["acquisition_time"] = pd.to_datetime(frame[time_col], utc=True, errors="coerce")
    frame = frame.dropna(subset=["acquisition_time"])
    if frame.empty:
        return frame
    frame["date"] = frame["acquisition_time"].dt.date.astype(str)

    for band in sentinel_base.S2_RAW_COLS:
        source = lower_lookup.get(band.lower())
        if source is None:
            raise ValueError(f"{source_path} is missing required Sentinel band {band}")
        frame[band] = pd.to_numeric(frame[source], errors="coerce")

    scl_col = find_column(("SCL", "scl"))
    frame["SCL"] = pd.to_numeric(frame[scl_col], errors="coerce") if scl_col is not None else np.nan

    valid_col = find_column(("valid_pixel",))
    if valid_col is not None:
        frame = frame.loc[sentinel_base.truthy_series(frame[valid_col])].copy()

    source_image_col = find_column(("source_image_id", "source_item_id", "image_id", "system:index"))
    frame["source_image_id"] = frame[source_image_col] if source_image_col is not None else source_path.stem
    lat_col = find_column(("latitude", "lat"))
    lon_col = find_column(("longitude", "lon"))
    if lat_col is not None:
        frame["latitude"] = pd.to_numeric(frame[lat_col], errors="coerce")
    if lon_col is not None:
        frame["longitude"] = pd.to_numeric(frame[lon_col], errors="coerce")

    keep_cols = [
        "city_token",
        "row_index",
        "date",
        "acquisition_time",
        "source_image_id",
        "SCL",
        "latitude",
        "longitude",
        *sentinel_base.S2_RAW_COLS,
    ]
    keep_cols = [column for column in keep_cols if column in frame.columns]
    return frame.loc[frame[sentinel_base.S2_RAW_COLS].notna().any(axis=1), keep_cols].copy()
def load_raw_daily_observations(
    city: str,
    paths: list[Path],
    needed_rows: set[int],
    args: argparse.Namespace,
    label: str,
) -> dict[int, pd.DataFrame]:
    if not paths or not needed_rows:
        return {}
    pieces: list[pd.DataFrame] = []
    progress_step = max(1, min(10, len(paths)))
    started = time.perf_counter()
    raw_load_mode = str(getattr(args, "raw_load_mode", "bulk"))
    if raw_load_mode == "batch":
        groups: dict[int, pd.DataFrame] = {}
        batch_paths = raw_paths_by_batch(paths)
        batch_progress_step = max(1, int(getattr(args, "progress_interval", 1)))
        for batch_pos, (batch_index, batch_files) in enumerate(batch_paths.items(), start=1):
            batch_started = time.perf_counter()
            bulk_usecols, bulk_row_col = raw_column_plan(batch_files[0])
            batch_pieces: list[pd.DataFrame] = []
            for path in batch_files:
                raw_file = pd.read_csv(path, usecols=bulk_usecols, low_memory=False)
                raw_file["batch_index"] = int(batch_index)
                if not raw_file.empty:
                    batch_pieces.append(raw_file)
            if not batch_pieces:
                continue
            raw = pd.concat(batch_pieces, ignore_index=True)
            raw["row_index"] = pd.to_numeric(raw[bulk_row_col], errors="coerce")
            raw = raw.loc[raw["row_index"].notna()].copy()
            raw["row_index"] = raw["row_index"].astype(np.int64)
            needed = pd.DataFrame({"row_index": np.fromiter(needed_rows, dtype=np.int64)})
            before_rows = int(len(raw))
            raw = raw.merge(needed, on="row_index", how="inner", sort=False)
            if raw.empty:
                if batch_pos == 1 or batch_pos == len(batch_paths) or batch_pos % batch_progress_step == 0:
                    print(
                        f"{city}: {label} batch {batch_pos:,}/{len(batch_paths):,} "
                        f"(batch_{batch_index:05d}) had no matched rows; "
                        f"raw_rows={before_rows:,}; elapsed={time.perf_counter() - started:.1f}s",
                        flush=True,
                    )
                continue
            raw = normalize_raw_chunk_for_needed(raw, city, batch_files[0], needed_rows)
            if raw.empty:
                continue
            phase_started = time.perf_counter()
            raw = sentinel_base.add_s2_indices(raw)
            indices_elapsed = time.perf_counter() - phase_started
            phase_started = time.perf_counter()
            daily = aggregate_daily_for_args(raw, args)
            aggregate_elapsed = time.perf_counter() - phase_started
            if daily.empty:
                continue
            interp_args = argparse.Namespace(
                gap_threshold_days=int(args.gap_threshold_days),
                interpolation_interval_days=int(args.interpolation_interval_days),
                progress_every_cells=0,
            )
            phase_started = time.perf_counter()
            frame = interpolated_sentinel.interpolate_time_series(daily, int(batch_index), interp_args)
            interpolate_elapsed = time.perf_counter() - phase_started
            if frame.empty:
                continue
            keep = ["row_index", "date"] + list(shard_schema.SENTINEL_SEQUENCE_COLUMNS)
            frame = frame[[column for column in keep if column in frame.columns]].copy()
            for column in shard_schema.SENTINEL_SEQUENCE_COLUMNS:
                if column not in frame:
                    frame[column] = 0.0
                frame[column] = pd.to_numeric(frame[column], errors="coerce")
            frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
            for row_index, group in frame.groupby("row_index", sort=False):
                cell_id = int(row_index)
                if cell_id in groups:
                    groups[cell_id] = pd.concat([groups[cell_id], group], ignore_index=True)
                else:
                    groups[cell_id] = group
            if batch_pos == 1 or batch_pos == len(batch_paths) or batch_pos % batch_progress_step == 0:
                print(
                    f"{city}: processed {label} raw batch {batch_pos:,}/{len(batch_paths):,} "
                    f"(batch_{batch_index:05d}); files={len(batch_files):,}; "
                    f"raw_rows={before_rows:,}->{len(raw):,}; daily={len(daily):,}; "
                    f"cells={frame['row_index'].nunique():,}; "
                    f"indices={indices_elapsed:.1f}s; aggregate={aggregate_elapsed:.1f}s; "
                    f"interpolate={interpolate_elapsed:.1f}s; batch_elapsed={time.perf_counter() - batch_started:.1f}s; "
                    f"total_elapsed={time.perf_counter() - started:.1f}s",
                    flush=True,
                )
        return groups
    if raw_load_mode == "bulk":
        bulk_usecols, bulk_row_col = raw_column_plan(paths[0])
        for path_index, path in enumerate(paths, start=1):
            parsed = parse_tree_centered_raw_export(path)
            batch_index = parsed[0] if parsed else 0
            raw_file = pd.read_csv(path, usecols=bulk_usecols, low_memory=False)
            raw_file["batch_index"] = int(batch_index)
            loaded_rows = int(len(raw_file))
            if not raw_file.empty:
                pieces.append(raw_file)
            if path_index == 1 or path_index == len(paths) or path_index % progress_step == 0:
                elapsed = time.perf_counter() - started
                print(
                    f"{city}: bulk-loaded {label} raw file {path_index:,}/{len(paths):,}; "
                    f"valid_rows_in_file={loaded_rows:,}; elapsed={elapsed:.1f}s",
                    flush=True,
                )
    else:
        for path_index, path in enumerate(paths, start=1):
            parsed = parse_tree_centered_raw_export(path)
            batch_index = parsed[0] if parsed else 0
            matched_rows = 0
            for chunk in pd.read_csv(path, usecols=raw_wanted_columns(path), chunksize=args.chunksize, low_memory=False):
                normalized = normalize_raw_chunk_for_needed(chunk, city, path, needed_rows)
                if normalized.empty:
                    continue
                matched_rows += int(len(normalized))
                normalized["batch_index"] = batch_index
                pieces.append(normalized)
            if path_index == 1 or path_index == len(paths) or path_index % progress_step == 0:
                elapsed = time.perf_counter() - started
                print(
                    f"{city}: loaded {label} raw file {path_index:,}/{len(paths):,}; "
                    f"matched_rows_in_file={matched_rows:,}; elapsed={elapsed:.1f}s",
                    flush=True,
                )
    if not pieces:
        return {}
    phase_started = time.perf_counter()
    raw = pd.concat(pieces, ignore_index=True)
    print(
        f"{city}: {label} concat raw rows={len(raw):,}; elapsed={time.perf_counter() - phase_started:.1f}s",
        flush=True,
    )
    if raw_load_mode == "bulk":
        phase_started = time.perf_counter()
        raw["row_index"] = pd.to_numeric(raw[bulk_row_col], errors="coerce")
        raw = raw.loc[raw["row_index"].notna()].copy()
        raw["row_index"] = raw["row_index"].astype(np.int64)
        print(
            f"{city}: {label} prepared row_index for bulk raw rows={len(raw):,}; "
            f"elapsed={time.perf_counter() - phase_started:.1f}s",
            flush=True,
        )
        phase_started = time.perf_counter()
        needed = pd.DataFrame({"row_index": np.fromiter(needed_rows, dtype=np.int64)})
        before_rows = int(len(raw))
        raw = raw.merge(needed, on="row_index", how="inner", sort=False)
        print(
            f"{city}: {label} inner-joined raw rows {before_rows:,}->{len(raw):,}; "
            f"needed_ids={len(needed_rows):,}; elapsed={time.perf_counter() - phase_started:.1f}s",
            flush=True,
        )
        if raw.empty:
            return {}
        phase_started = time.perf_counter()
        raw = normalize_raw_chunk_for_needed(raw, city, paths[0], needed_rows)
        print(
            f"{city}: {label} normalized joined raw rows={len(raw):,}; "
            f"elapsed={time.perf_counter() - phase_started:.1f}s",
            flush=True,
        )
        if raw.empty:
            return {}
    phase_started = time.perf_counter()
    raw = sentinel_base.add_s2_indices(raw)
    print(
        f"{city}: {label} derived Sentinel indices; elapsed={time.perf_counter() - phase_started:.1f}s",
        flush=True,
    )
    phase_started = time.perf_counter()
    daily = aggregate_daily_for_args(raw, args)
    print(
        f"{city}: {label} aggregated daily rows={len(daily):,}; elapsed={time.perf_counter() - phase_started:.1f}s",
        flush=True,
    )
    if daily.empty:
        return {}
    interp_args = argparse.Namespace(
        gap_threshold_days=int(args.gap_threshold_days),
        interpolation_interval_days=int(args.interpolation_interval_days),
        progress_every_cells=int(args.progress_interval),
    )
    phase_started = time.perf_counter()
    frame = interpolated_sentinel.interpolate_time_series(daily, 0, interp_args)
    print(
        f"{city}: {label} interpolated time-series rows={len(frame):,}; elapsed={time.perf_counter() - phase_started:.1f}s",
        flush=True,
    )
    if frame.empty:
        return {}
    keep = ["row_index", "date"] + list(shard_schema.SENTINEL_SEQUENCE_COLUMNS)
    frame = frame[[column for column in keep if column in frame.columns]].copy()
    for column in shard_schema.SENTINEL_SEQUENCE_COLUMNS:
        if column not in frame:
            frame[column] = 0.0
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    phase_started = time.perf_counter()
    groups = {int(row_index): group for row_index, group in frame.groupby("row_index", sort=False)}
    print(
        f"{city}: {label} grouped interpolated cells={len(groups):,}; elapsed={time.perf_counter() - phase_started:.1f}s",
        flush=True,
    )
    return groups


def compute_phenology(group: pd.DataFrame, outlier_abs: float) -> np.ndarray:
    group = group.sort_values("date", kind="stable")
    values = group[list(shard_schema.SENTINEL_SEQUENCE_COLUMNS)].to_numpy(dtype=np.float32, copy=True)
    mask = np.ones(values.shape[0], dtype=bool)
    return derived_metrics.compute_sentinel_phenology(
        values,
        mask,
        list(shard_schema.SENTINEL_SEQUENCE_COLUMNS),
        float(outlier_abs),
    ).astype(np.float32)


def compute_group_phenology_cache(
    city: str,
    label: str,
    groups: dict[int, pd.DataFrame],
    outlier_abs: float,
    progress_interval: int,
) -> dict[int, np.ndarray]:
    started = time.perf_counter()
    cache: dict[int, np.ndarray] = {}
    total = len(groups)
    for pos, (cell_id, group) in enumerate(groups.items(), start=1):
        cache[int(cell_id)] = compute_phenology(group, outlier_abs)
        if progress_interval and (pos % progress_interval == 0 or pos == total):
            elapsed = time.perf_counter() - started
            rate = pos / max(elapsed, 1e-6)
            print(
                f"{city}: computed {label} phenology groups {pos:,}/{total:,}; "
                f"rate={rate:.1f}/s",
                flush=True,
            )
    return cache


def time_series_output_path(args: argparse.Namespace, city: str) -> Path:
    return Path(args.timeseries_output_root) / city / f"{city}_tree_centered_sentinel_time_series.csv"


def groups_to_frame(groups: dict[int, pd.DataFrame], source_code: int) -> pd.DataFrame:
    pieces: list[pd.DataFrame] = []
    for cell_id, group in groups.items():
        if group.empty:
            continue
        frame = group.copy()
        frame.insert(0, "sentinel_cell_id", int(cell_id))
        frame.insert(0, "source_code", int(source_code))
        pieces.append(frame)
    if not pieces:
        return pd.DataFrame(columns=["source_code", "sentinel_cell_id", "row_index", "date", *shard_schema.SENTINEL_SEQUENCE_COLUMNS])
    return pd.concat(pieces, ignore_index=True)


def write_interpolated_time_series(
    args: argparse.Namespace,
    city: str,
    original_groups: dict[int, pd.DataFrame],
    additional_groups: dict[int, pd.DataFrame],
    config: dict[str, Any],
) -> tuple[Path, int, int]:
    path = time_series_output_path(args, city)
    original = groups_to_frame(original_groups, 1)
    additional = groups_to_frame(additional_groups, 2)
    frame = pd.concat([original, additional], ignore_index=True)
    if args.dry_run:
        return path, int(len(frame)), int(frame["sentinel_cell_id"].nunique()) if not frame.empty else 0
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    config_path = path.with_name(f"{city}_tree_centered_sentinel_time_series_config.json")
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(
        f"{city}: wrote interpolated Sentinel time series {path}; "
        f"rows={len(frame):,}; cells={int(frame['sentinel_cell_id'].nunique()) if not frame.empty else 0:,}",
        flush=True,
    )
    return path, int(len(frame)), int(frame["sentinel_cell_id"].nunique()) if not frame.empty else 0


def read_tree_centered_time_series(path: Path, chunksize: int) -> tuple[dict[int, pd.DataFrame], dict[int, pd.DataFrame]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing interpolated Sentinel time-series file: {path}")
    wanted = ["source_code", "sentinel_cell_id", "date"] + list(shard_schema.SENTINEL_SEQUENCE_COLUMNS)
    header = set(pd.read_csv(path, nrows=0).columns)
    missing = set(wanted).difference(header)
    if missing:
        raise ValueError(f"{path} is missing Sentinel time-series columns: {sorted(missing)}")
    pieces: list[pd.DataFrame] = []
    for chunk in pd.read_csv(path, usecols=wanted, chunksize=chunksize, low_memory=False):
        chunk["source_code"] = pd.to_numeric(chunk["source_code"], errors="coerce").astype("Int64")
        chunk["sentinel_cell_id"] = pd.to_numeric(chunk["sentinel_cell_id"], errors="coerce").astype("Int64")
        chunk["date"] = pd.to_datetime(chunk["date"], errors="coerce")
        chunk = chunk.dropna(subset=["source_code", "sentinel_cell_id", "date"])
        if chunk.empty:
            continue
        for column in shard_schema.SENTINEL_SEQUENCE_COLUMNS:
            chunk[column] = pd.to_numeric(chunk[column], errors="coerce")
        pieces.append(chunk)
    if not pieces:
        return {}, {}
    frame = pd.concat(pieces, ignore_index=True)
    original: dict[int, pd.DataFrame] = {}
    additional: dict[int, pd.DataFrame] = {}
    for (source_code, cell_id), group in frame.groupby(["source_code", "sentinel_cell_id"], sort=False):
        target = original if int(source_code) == 1 else additional
        target[int(cell_id)] = group.drop(columns=["source_code", "sentinel_cell_id"]).copy()
    return original, additional


def load_expected_tree_index(index_path: Path) -> np.ndarray:
    index = pd.read_csv(index_path, usecols=["tree_centered_index"], low_memory=False)
    return pd.to_numeric(index["tree_centered_index"], errors="raise").to_numpy(dtype=np.int64)


def existing_sidecar_status(args: argparse.Namespace, out_path: Path, index_path: Path) -> dict[str, Any]:
    expected_index = load_expected_tree_index(index_path)
    with np.load(out_path, allow_pickle=False) as data:
        actual_index = np.asarray(data["tree_centered_index"], dtype=np.int64)
        missing = np.asarray(data["missing_sentinel_phenology"], dtype=bool)
    rows = int(actual_index.shape[0])
    missing_count = int(missing.sum())
    index_aligned = rows == int(expected_index.shape[0]) and np.array_equal(actual_index, expected_index)
    phenology_complete = missing_count == 0 or bool(args.allow_missing_sentinel_phenology)
    complete = bool(index_aligned and phenology_complete)
    reason = "complete"
    if not index_aligned:
        reason = f"stale_or_misaligned_rows expected={int(expected_index.shape[0]):,} actual={rows:,}"
    elif not phenology_complete:
        reason = f"missing_sentinel_phenology={missing_count:,}"
    return {
        "complete": complete,
        "reason": reason,
        "rows": rows,
        "valid": rows - missing_count,
        "missing": missing_count,
    }


def process_city(args: argparse.Namespace, city: str, index_path: Path) -> dict[str, Any]:
    out_dir = Path(args.output_root) / city
    out_path = out_dir / f"{city}_tree_centered_sentinel_phenology.npz"
    print(f"{city}: starting Sentinel stage={args.stage} from {index_path}", flush=True)
    if args.stage in {"compute", "all"} and out_path.exists() and not args.force:
        status = existing_sidecar_status(args, out_path, index_path)
        if status["complete"]:
            return {
                "city_token": city,
                "status": "skipped_complete",
                "stage": str(args.stage),
                "rows": int(status["rows"]),
                "valid": int(status["valid"]),
                "missing": int(status["missing"]),
            }
        print(f"{city}: existing sidecar is not complete ({status['reason']}); rebuilding.", flush=True)
    index = pd.read_csv(index_path, low_memory=False)
    required = {"tree_centered_index", "crown_reduced_id"}
    missing = required.difference(index.columns)
    if missing:
        raise RuntimeError(f"{index_path} is missing required columns: {sorted(missing)}")
    tree_index = pd.to_numeric(index["tree_centered_index"], errors="coerce").fillna(-1).to_numpy(dtype=np.int64)
    crown_ids = pd.to_numeric(index["crown_reduced_id"], errors="coerce")
    existing_ids = pd.to_numeric(index["existing_reduced_id"], errors="coerce") if "existing_reduced_id" in index.columns else pd.Series(np.nan, index=index.index)
    needed_original = {value for value in (finite_int(v) for v in existing_ids) if value is not None}
    crown_by_existing: dict[int, set[int]] = {}
    crown_without_existing: set[int] = set()
    for existing_value, crown_value in zip(existing_ids, crown_ids):
        crown_id = finite_int(crown_value)
        if crown_id is None:
            continue
        existing_id = finite_int(existing_value)
        if existing_id is None:
            crown_without_existing.add(crown_id)
        else:
            crown_by_existing.setdefault(existing_id, set()).add(crown_id)
    needed_additional = set(crown_without_existing)

    def count_records_with_time_series(
        original_groups: dict[int, pd.DataFrame],
        additional_groups: dict[int, pd.DataFrame],
    ) -> int:
        count = 0
        for existing_value, crown_value in zip(existing_ids, crown_ids):
            existing_id = finite_int(existing_value)
            crown_id = finite_int(crown_value)
            if existing_id is not None and existing_id in original_groups:
                count += 1
            elif crown_id is not None and crown_id in additional_groups:
                count += 1
        return count

    cached_original_groups: dict[int, pd.DataFrame] = {}
    cached_additional_groups: dict[int, pd.DataFrame] = {}
    original_paths: list[Path] = []
    original_raw_paths: list[Path] = []
    complete_original_raw_paths: list[Path] = []
    additional_paths: list[Path] = []
    complete_additional_paths: list[Path] = []
    empty_raw_status = {
        "checked": False,
        "expected": 0,
        "present": 0,
        "present_matching": 0,
        "missing": 0,
        "complete_batches": 0,
        "incomplete_batches": 0,
        "needed_batches": 0,
        "complete_needed_batches": 0,
        "excluded_additional_cells": 0,
    }
    original_raw_status = dict(empty_raw_status)
    raw_file_status = dict(empty_raw_status)
    if args.stage in {"interpolate", "all"} and not args.force:
        ts_path = time_series_output_path(args, city)
        if ts_path.exists() and not bool(args.prefer_separate_timeseries):
            load_started = time.perf_counter()
            cached_original_groups, cached_additional_groups = read_tree_centered_time_series(ts_path, int(args.chunksize))
            needed_original = needed_original.difference(cached_original_groups.keys())
            for existing_id in cached_original_groups:
                needed_additional.difference_update(crown_by_existing.get(existing_id, set()))
            needed_additional = needed_additional.difference(cached_additional_groups.keys())
            print(
                f"{city}: reusing existing tree-centered interpolated time series; "
                f"cached_original={len(cached_original_groups):,}; "
                f"cached_supplemental={len(cached_additional_groups):,}; "
                f"remaining_original={len(needed_original):,}; "
                f"remaining_supplemental={len(needed_additional):,}; "
                f"elapsed={time.perf_counter() - load_started:.1f}s",
                flush=True,
            )
            if args.stage == "interpolate" and not needed_original and not needed_additional:
                rows = sum(1 for _ in open(ts_path, "rb")) - 1
                valid_tree_records = count_records_with_time_series(cached_original_groups, cached_additional_groups)
                sentinel_keys = len(cached_original_groups) + len(cached_additional_groups)
                return {
                    "city_token": city,
                    "status": "skipped_interpolated_complete",
                    "stage": "interpolate",
                    "rows": int(len(index)),
                    "valid": int(valid_tree_records),
                    "missing": int(len(index) - valid_tree_records),
                    "sentinel_keys": int(sentinel_keys),
                    "time_series_rows": int(rows),
                    "time_series_path": str(ts_path),
                }

    original_source = "interpolated"
    if args.stage == "compute":
        ts_path = time_series_output_path(args, city)
        if ts_path.exists():
            print(f"{city}: reading combined interpolated Sentinel time series {ts_path}", flush=True)
            load_started = time.perf_counter()
            original_groups, additional_groups = read_tree_centered_time_series(ts_path, int(args.chunksize))
            original_source = "combined_time_series"
            print(
                f"{city}: loaded combined time-series groups; original={len(original_groups):,}; "
                f"supplemental={len(additional_groups):,}; elapsed={time.perf_counter() - load_started:.1f}s",
                flush=True,
            )
        else:
            print(
                f"{city}: combined time series not found; reading separate original/supplemental interpolated folders",
                flush=True,
            )
            original_groups, additional_groups, original_paths, additional_paths = load_separate_interpolated_groups(
                args,
                city,
                needed_original,
                needed_additional,
            )
            unresolved_original = needed_original.difference(original_groups.keys())
            fallback_additional = set(needed_additional)
            for existing_id in unresolved_original:
                fallback_additional.update(crown_by_existing.get(existing_id, set()))
            if fallback_additional != needed_additional:
                _unused_original, loaded_fallback, _orig_paths, _supp_paths = load_separate_interpolated_groups(
                    args,
                    city,
                    set(),
                    fallback_additional,
                )
                additional_groups.update(loaded_fallback)
            original_source = "separate_time_series"
            print(
                f"{city}: loaded separate time-series groups; original={len(original_groups):,}; "
                f"supplemental={len(additional_groups):,}; unresolved_original={len(unresolved_original):,}",
                flush=True,
            )
    else:
        original_paths = city_paths(args.original_timeseries_dir, args.original_timeseries_pattern, city)
        original_raw_paths = city_paths(args.original_raw_sentinel_dir, args.original_raw_pattern, city)
        complete_original_raw_paths, original_raw_status = complete_raw_paths_by_present_batch(
            city,
            original_raw_paths,
            "s2_reduced_cells",
            str(args.original_raw_export_start_date),
            str(args.original_raw_export_end_date),
            int(args.original_raw_export_interval_days),
        )
        needed_original_raw_batches = raw_batches_for_row_ids(needed_original, int(args.original_raw_export_batch_size))
        if needed_original_raw_batches:
            before_count = len(complete_original_raw_paths)
            complete_original_raw_paths = [
                path for path in complete_original_raw_paths if parse_raw_batch_index(path) in needed_original_raw_batches
            ]
            if before_count != len(complete_original_raw_paths):
                print(
                    f"{city}: pruned original raw Sentinel files by needed batch; "
                    f"needed_batches={len(needed_original_raw_batches):,}; "
                    f"files={len(complete_original_raw_paths):,}/{before_count:,}",
                    flush=True,
                )
        else:
            complete_original_raw_paths = []
        additional_paths = city_paths(args.additional_raw_sentinel_dir, args.additional_raw_pattern, city)
        complete_additional, incomplete_additional, complete_additional_paths, raw_file_status = plan_complete_additional_raw_batches(
            args,
            city,
            needed_additional,
            additional_paths,
        )
        print(
            f"{city}: loading Sentinel sources; original_interpolated_files={len(original_paths):,}; "
            f"original_raw_files={len(complete_original_raw_paths):,}/{len(original_raw_paths):,}; "
            f"supplemental_files={len(complete_additional_paths):,}/{len(additional_paths):,}; "
            f"original_raw_complete_batches={int(original_raw_status['complete_batches']):,}; "
            f"original_raw_incomplete_batches={int(original_raw_status['incomplete_batches']):,}; "
            f"complete_batches={int(raw_file_status['complete_batches']):,}; "
            f"incomplete_batches={int(raw_file_status['incomplete_batches']):,}",
            flush=True,
        )
        load_started = time.perf_counter()
        original_groups = dict(cached_original_groups)
        additional_groups = dict(cached_additional_groups)
        loaded_original_groups = read_interpolated_timeseries(original_paths, needed_original, int(args.chunksize))
        if loaded_original_groups:
            original_groups.update(loaded_original_groups)
            original_source = "interpolated"
            for existing_id in loaded_original_groups:
                needed_additional.difference_update(crown_by_existing.get(existing_id, set()))
        remaining_original = needed_original.difference(loaded_original_groups.keys())
        if remaining_original and complete_original_raw_paths:
            raw_original_groups = load_raw_daily_observations(city, complete_original_raw_paths, remaining_original, args, "original")
            if raw_original_groups:
                original_groups.update(raw_original_groups)
                for existing_id in raw_original_groups:
                    needed_additional.difference_update(crown_by_existing.get(existing_id, set()))
                if loaded_original_groups:
                    original_source = "interpolated+raw"
                elif cached_original_groups:
                    original_source = "cached_time_series+raw"
                else:
                    original_source = "raw"
        elif len(original_groups) == len(cached_original_groups) and cached_original_groups:
            original_source = "cached_time_series"
        print(
            f"{city}: loaded original Sentinel groups={len(original_groups):,} from {original_source}; "
            f"elapsed={time.perf_counter() - load_started:.1f}s",
            flush=True,
        )
        load_started = time.perf_counter()
        loaded_additional_groups = load_raw_daily_observations(city, complete_additional_paths, complete_additional, args, "supplemental")
        if loaded_additional_groups:
            additional_groups.update(loaded_additional_groups)
        print(
            f"{city}: loaded supplemental Sentinel groups={len(additional_groups):,}; "
            f"elapsed={time.perf_counter() - load_started:.1f}s",
            flush=True,
        )
        ts_config = {
            "record_index": str(index_path),
            "original_source": original_source,
            "original_timeseries_dir": str(args.original_timeseries_dir),
            "original_raw_sentinel_dir": str(args.original_raw_sentinel_dir),
            "additional_raw_sentinel_dir": str(args.additional_raw_sentinel_dir),
            "original_files": [str(path) for path in original_paths],
            "complete_original_raw_files": [str(path) for path in complete_original_raw_paths],
            "complete_additional_files": [str(path) for path in complete_additional_paths],
            "original_raw_file_completeness": original_raw_status,
            "raw_file_completeness": raw_file_status,
            "gap_threshold_days": int(args.gap_threshold_days),
            "interpolation_interval_days": int(args.interpolation_interval_days),
            "original_raw_export_batch_size": int(args.original_raw_export_batch_size),
        }
        ts_path, ts_rows, ts_cells = write_interpolated_time_series(args, city, original_groups, additional_groups, ts_config)
        if args.stage == "interpolate":
            valid_tree_records = count_records_with_time_series(original_groups, additional_groups)
            return {
                "city_token": city,
                "status": "dry_run" if args.dry_run else "interpolated",
                "stage": "interpolate",
                "rows": int(len(index)),
                "valid": int(valid_tree_records),
                "missing": int(len(index) - valid_tree_records),
                "sentinel_keys": int(ts_cells),
                "time_series_rows": int(ts_rows),
                "time_series_path": str(ts_path),
            }

    cache_progress = max(0, int(args.progress_interval))
    original_pheno = compute_group_phenology_cache(
        city,
        f"original-{original_source}",
        original_groups,
        args.sentinel_outlier_abs,
        cache_progress,
    )
    additional_pheno = compute_group_phenology_cache(
        city,
        "supplemental",
        additional_groups,
        args.sentinel_outlier_abs,
        cache_progress,
    )

    phenology = np.zeros((len(index), len(derived_metrics.SENTINEL_PHENOLOGY_COLUMNS)), dtype=np.float32)
    missing_pheno = np.ones(len(index), dtype=bool)
    used_original = np.zeros(len(index), dtype=bool)
    used_additional = np.zeros(len(index), dtype=bool)
    source_code = np.zeros(len(index), dtype=np.int8)  # 0 missing, 1 original, 2 additional
    progress_interval = max(0, int(args.progress_interval))
    for out_index, (existing_value, crown_value) in enumerate(zip(existing_ids, crown_ids)):
        existing_id = finite_int(existing_value)
        crown_id = finite_int(crown_value)
        vector = original_pheno.get(existing_id) if existing_id is not None else None
        if vector is not None:
            phenology[out_index] = vector
            missing_pheno[out_index] = False
            used_original[out_index] = True
            source_code[out_index] = 1
            continue
        vector = additional_pheno.get(crown_id) if crown_id is not None else None
        if vector is not None:
            phenology[out_index] = vector
            missing_pheno[out_index] = False
            used_additional[out_index] = True
            source_code[out_index] = 2
        if progress_interval and (out_index + 1) % progress_interval == 0:
            print(
                f"{city}: phenology rows {out_index + 1:,}/{len(index):,}; "
                f"valid={int((~missing_pheno[: out_index + 1]).sum()):,}; "
                f"missing={int(missing_pheno[: out_index + 1].sum()):,}",
                flush=True,
            )
    if args.dry_run:
        return {
            "city_token": city,
            "status": "dry_run",
            "rows": int(len(index)),
            "valid": int((~missing_pheno).sum()),
            "missing": int(missing_pheno.sum()),
            "original": int(used_original.sum()),
            "additional": int(used_additional.sum()),
            "original_raw_expected_files": int(original_raw_status["expected"]),
            "original_raw_present_files": int(original_raw_status["present"]),
            "original_raw_complete_batches": int(original_raw_status["complete_batches"]),
            "original_raw_incomplete_batches": int(original_raw_status["incomplete_batches"]),
            "raw_expected_files": int(raw_file_status["expected"]),
            "raw_present_files": int(raw_file_status["present"]),
            "raw_complete_batches": int(raw_file_status["complete_batches"]),
            "raw_incomplete_batches": int(raw_file_status["incomplete_batches"]),
            "raw_excluded_additional_cells": int(raw_file_status["excluded_additional_cells"]),
        }
    out_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "record_index": str(index_path),
        "original_timeseries_dir": str(args.original_timeseries_dir),
        "supplemental_timeseries_dir": str(args.supplemental_timeseries_dir),
        "original_raw_sentinel_dir": str(args.original_raw_sentinel_dir),
        "additional_raw_sentinel_dir": str(args.additional_raw_sentinel_dir),
        "original_files": [str(path) for path in original_paths],
        "supplemental_timeseries_files": [str(path) for path in additional_paths],
        "original_raw_files": [str(path) for path in original_raw_paths],
        "complete_original_raw_files": [str(path) for path in complete_original_raw_paths],
        "additional_files": [str(path) for path in additional_paths],
        "complete_additional_files": [str(path) for path in complete_additional_paths],
        "missing_raw_cell_root": str(args.missing_raw_cell_root),
        "raw_export_start_date": str(args.raw_export_start_date),
        "raw_export_end_date": str(args.raw_export_end_date),
        "raw_export_interval_days": int(args.raw_export_interval_days),
        "raw_export_batch_size": int(args.raw_export_batch_size),
        "original_raw_export_batch_size": int(args.original_raw_export_batch_size),
        "original_raw_file_completeness": original_raw_status,
        "raw_file_completeness": raw_file_status,
        "gap_threshold_days": int(args.gap_threshold_days),
        "interpolation_interval_days": int(args.interpolation_interval_days),
    }
    np.savez_compressed(
        out_path,
        sentinel_phenology=phenology,
        sentinel_phenology_columns=np.asarray(derived_metrics.SENTINEL_PHENOLOGY_COLUMNS),
        tree_centered_index=tree_index,
        crown_reduced_id=crown_ids.fillna(-1).to_numpy(dtype=np.int64),
        existing_reduced_id=existing_ids.fillna(-1).to_numpy(dtype=np.int64),
        missing_sentinel_phenology=missing_pheno,
        used_original_sentinel=used_original,
        used_additional_sentinel=used_additional,
        sentinel_phenology_source_code=source_code,
        config_json=np.asarray(json.dumps(config, indent=2)),
    )
    print(
        f"{city}: wrote Sentinel phenology sidecar {out_path}; "
        f"valid={int((~missing_pheno).sum()):,}/{len(index):,}; missing={int(missing_pheno.sum()):,}",
        flush=True,
    )
    return {
        "city_token": city,
        "status": "completed",
        "rows": int(len(index)),
        "valid": int((~missing_pheno).sum()),
        "missing": int(missing_pheno.sum()),
        "original": int(used_original.sum()),
        "additional": int(used_additional.sum()),
        "original_raw_expected_files": int(original_raw_status["expected"]),
        "original_raw_present_files": int(original_raw_status["present"]),
        "original_raw_complete_batches": int(original_raw_status["complete_batches"]),
        "original_raw_incomplete_batches": int(original_raw_status["incomplete_batches"]),
        "raw_expected_files": int(raw_file_status["expected"]),
        "raw_present_files": int(raw_file_status["present"]),
        "raw_complete_batches": int(raw_file_status["complete_batches"]),
        "raw_incomplete_batches": int(raw_file_status["incomplete_batches"]),
        "raw_excluded_additional_cells": int(raw_file_status["excluded_additional_cells"]),
        "output_path": str(out_path),
    }


def main() -> int:
    args = parse_args()
    indexes = discover_record_indexes(args)
    print(f"Deriving tree-centered Sentinel phenology for {len(indexes):,} city/cities.", flush=True)
    results: list[dict[str, Any]] = []
    failures: list[tuple[str, str]] = []
    workers = max(1, min(int(args.parallel_workers), len(indexes)))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(process_city, args, city, path): city for city, path in sorted(indexes.items())}
        for future in as_completed(futures):
            city = futures[future]
            try:
                result = future.result()
                results.append(result)
                if result.get("stage") == "interpolate":
                    print(
                        f"{city}: {result['status']}; "
                        f"valid_tree_records={int(result['valid']):,}/{int(result['rows']):,}; "
                        f"sentinel_keys={int(result.get('sentinel_keys', 0)):,}; "
                        f"missing_tree_records={int(result['missing']):,}",
                        flush=True,
                    )
                else:
                    print(
                        f"{city}: {result['status']}; valid={int(result['valid']):,}/{int(result['rows']):,}; "
                        f"missing={int(result['missing']):,}",
                        flush=True,
                    )
            except Exception as exc:
                failures.append((city, str(exc)))
                print(f"{city}: FAILED: {exc}", flush=True)
    if not args.dry_run:
        Path(args.output_root).mkdir(parents=True, exist_ok=True)
        summary_path = Path(args.output_root) / "tree_centered_sentinel_phenology_summary.csv"
        pd.DataFrame(results).sort_values("city_token").to_csv(summary_path, index=False)
        print(f"Wrote summary: {summary_path}", flush=True)
    else:
        print("Dry run complete; no files written.", flush=True)
    if failures:
        print("Failed cities:", flush=True)
        for city, message in failures:
            print(f"  {city}: {message}", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
