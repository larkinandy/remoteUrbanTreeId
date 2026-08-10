#!/usr/bin/env python3
"""Derive daily PRISM weather-context features for Sentinel-2 cell sequences.

The output is a per-city sidecar keyed by ``row_index`` and Sentinel observation
date. It is meant to pair daily weather context with the Sentinel-2 time-series
rows already used by the multimodal model.

Recommended first feature set:

* GDD base-5C cumulative year-to-date
* GDD base-5C over the previous 30 days
* precipitation over the previous 30 days
* precipitation over the previous 90 days
* dry days over the previous 30 days
* mean maximum VPD over the previous 30 days
* maximum maximum VPD over the previous 30 days
* 30-day precipitation anomaly relative to PRISM monthly normals
* 30-day maximum VPD anomaly relative to PRISM monthly normals
"""

from __future__ import annotations

import argparse
import calendar
import csv
import json
import math
import re
import shutil
import sys
import tempfile
import time
import zipfile
from collections import deque
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
LIDAR_HELPER_DIR = HERE.parent / "LiDAR"
DEFAULT_CELL_MAP_DIR = HERE.parent / "Sentinel2" / "mccoy_sentinel_10m_cells_utm"
DEFAULT_SENTINEL2_DIR = Path(r"E:\cell\sentinel2_timeseries")
DEFAULT_PRISM_DAILY_ROOT = Path(r"E:\PRISM\sentinel_cells\raw\daily")
DEFAULT_PRISM_NORMALS_DIR = Path(r"E:\PRISM\sentinel_cells\extracted_normals")
DEFAULT_OUTPUT_DIR = Path(r"E:\PRISM\sentinel_cells\daily_context")

DEFAULT_SENTINEL_PATTERN = "**/*_sentinel2_*_time_series_batch_*.csv"
DEFAULT_DAILY_VARIABLES = ("ppt", "tmean", "vpdmax")
FEATURE_NAMES = (
    "prism_gdd5_ytd",
    "prism_gdd5_30d",
    "prism_ppt_30d_mm",
    "prism_ppt_90d_mm",
    "prism_dry_days_30d",
    "prism_vpdmax_30d_mean",
    "prism_vpdmax_30d_max",
    "prism_ppt_30d_anomaly_mm",
    "prism_vpdmax_30d_anomaly",
)


@dataclass(frozen=True)
class CityCells:
    city: str
    cells: list[dict]
    row_index: np.ndarray
    lon: np.ndarray
    lat: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--city-token", action="append", default=[], help="Study city token/name. Repeatable.")
    source.add_argument("--all-cities", action="store_true", help="Use every city in --cell-map-dir. Default when omitted.")

    parser.add_argument("--exclude-city-token", action="append", default=["honolulu"], help="City token/name to skip.")
    parser.add_argument("--include-honolulu", action="store_true", help="Do not exclude Honolulu.")
    parser.add_argument("--cell-map-dir", type=Path, default=DEFAULT_CELL_MAP_DIR)
    parser.add_argument("--sentinel2-dir", type=Path, default=DEFAULT_SENTINEL2_DIR)
    parser.add_argument("--sentinel2-pattern", default=DEFAULT_SENTINEL_PATTERN)
    parser.add_argument("--prism-daily-root", type=Path, default=DEFAULT_PRISM_DAILY_ROOT)
    parser.add_argument("--prism-normals-dir", type=Path, default=DEFAULT_PRISM_NORMALS_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--variables", default=",".join(DEFAULT_DAILY_VARIABLES))
    parser.add_argument(
        "--temperature-source",
        choices=("auto", "tmean", "tmin_tmax"),
        default="auto",
        help="Temperature input used for GDD. auto prefers tmean and falls back to averaging tmin/tmax.",
    )
    parser.add_argument("--sentinel-cell-size", type=float, default=10.0)
    parser.add_argument("--sentinel-origin-x", type=float, default=0.0)
    parser.add_argument("--sentinel-origin-y", type=float, default=0.0)
    parser.add_argument("--gdd-base-c", type=float, default=5.0)
    parser.add_argument("--dry-day-threshold-mm", type=float, default=1.0)
    parser.add_argument("--observed-only", action="store_true", help="Use only Sentinel rows with sentinel_observed > 0.")
    parser.add_argument("--start-date", help="Optional lower bound for Sentinel target dates, YYYY-MM-DD.")
    parser.add_argument("--end-date", help="Optional upper bound for Sentinel target dates, YYYY-MM-DD.")
    parser.add_argument("--max-cells-per-city", type=int, default=0, help="Debug/sample cap. 0 means all cells.")
    parser.add_argument("--max-target-dates", type=int, default=0, help="Debug cap on unique Sentinel dates per city.")
    parser.add_argument("--sample-chunk-size", type=int, default=100000)
    parser.add_argument("--missing-daily-policy", choices=("error", "nan", "skip-date"), default="error")
    parser.add_argument(
        "--reproject-points",
        action="store_true",
        help="Reproject lon/lat cell centers to raster CRS before sampling. Usually unnecessary for PRISM grids.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--write-csv", action="store_true", help="Also write a wide CSV. This can be large.")
    parser.add_argument("--keep-temp", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def normalize_token(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value or "").lower().replace("alberquerque", "albuquerque"))


def parse_date(value: str | None) -> date | None:
    if not value:
        return None
    return datetime.strptime(value, "%Y-%m-%d").date()


def date_range(start: date, end: date) -> Iterable[date]:
    current = start
    while current <= end:
        yield current
        current += timedelta(days=1)


def load_city_helpers():
    sys.path.insert(0, str(LIDAR_HELPER_DIR))
    try:
        from identify_tnm_lidar_city_coverage import cell_center_lonlat, city_cell_tables, read_unique_cells
    finally:
        try:
            sys.path.remove(str(LIDAR_HELPER_DIR))
        except ValueError:
            pass
    return cell_center_lonlat, city_cell_tables, read_unique_cells


def iter_city_cells(args: argparse.Namespace) -> Iterable[CityCells]:
    cell_center_lonlat, city_cell_tables, read_unique_cells = load_city_helpers()
    wanted = {normalize_token(token) for token in args.city_token}
    excluded = set() if args.include_honolulu else {normalize_token(token) for token in args.exclude_city_token}
    found: set[str] = set()
    for city_name, path in city_cell_tables(args.cell_map_dir):
        city = normalize_token(city_name)
        if wanted and city not in wanted:
            continue
        if city in excluded:
            continue
        cells = read_unique_cells(path)
        if args.max_cells_per_city:
            cells = cells[: args.max_cells_per_city]
        for cell in cells:
            lon, lat = cell_center_lonlat(cell, args)
            cell["lon"] = lon
            cell["lat"] = lat
        found.add(city)
        yield CityCells(
            city=city,
            cells=cells,
            row_index=np.asarray([int(cell["reduced_id"]) for cell in cells], dtype=np.int64),
            lon=np.asarray([float(cell["lon"]) for cell in cells], dtype=np.float64),
            lat=np.asarray([float(cell["lat"]) for cell in cells], dtype=np.float64),
        )
    missing = sorted(wanted - found)
    if missing:
        raise FileNotFoundError(f"No Sentinel cell map found for city token(s): {', '.join(missing)}")


def discover_sentinel_paths(city: str, args: argparse.Namespace) -> list[Path]:
    city_dir = args.sentinel2_dir / city
    paths = sorted(city_dir.glob(args.sentinel2_pattern)) if city_dir.exists() else []
    if paths:
        return paths
    return [
        path
        for path in sorted(args.sentinel2_dir.glob(args.sentinel2_pattern))
        if city in normalize_token(path.name) or city in normalize_token(path.parent.name)
    ]


def read_target_rows(city_cells: CityCells, args: argparse.Namespace) -> pd.DataFrame:
    paths = discover_sentinel_paths(city_cells.city, args)
    if not paths:
        raise FileNotFoundError(f"{city_cells.city}: no Sentinel time-series CSVs found under {args.sentinel2_dir}")

    needed = set(int(value) for value in city_cells.row_index)
    start_date = parse_date(args.start_date)
    end_date = parse_date(args.end_date)
    usecols_base = ["row_index", "date"]
    pieces = []
    for path_index, path in enumerate(paths, start=1):
        header = set(pd.read_csv(path, nrows=0).columns)
        usecols = list(usecols_base)
        if args.observed_only and "sentinel_observed" in header:
            usecols.append("sentinel_observed")
        missing = set(usecols_base) - header
        if missing:
            raise ValueError(f"{path} is missing required Sentinel columns: {sorted(missing)}")
        print(f"{city_cells.city}: reading Sentinel dates {path_index:,}/{len(paths):,}: {path.name}", flush=True)
        for chunk in pd.read_csv(path, usecols=usecols, chunksize=250_000, low_memory=False):
            chunk["row_index"] = pd.to_numeric(chunk["row_index"], errors="coerce")
            chunk = chunk.loc[chunk["row_index"].isin(needed)].copy()
            if chunk.empty:
                continue
            chunk["row_index"] = chunk["row_index"].astype(np.int64)
            chunk["date"] = pd.to_datetime(chunk["date"], errors="coerce").dt.date
            chunk = chunk.dropna(subset=["date"])
            if args.observed_only and "sentinel_observed" in chunk:
                chunk["sentinel_observed"] = pd.to_numeric(chunk["sentinel_observed"], errors="coerce").fillna(0.0)
                chunk = chunk.loc[chunk["sentinel_observed"].gt(0)]
            if start_date:
                chunk = chunk.loc[chunk["date"] >= start_date]
            if end_date:
                chunk = chunk.loc[chunk["date"] <= end_date]
            if not chunk.empty:
                pieces.append(chunk[["row_index", "date"]])
    if not pieces:
        return pd.DataFrame(columns=["row_index", "date"])
    frame = pd.concat(pieces, ignore_index=True)
    frame = frame.sort_values(["date", "row_index"], kind="stable").drop_duplicates(["row_index", "date"], keep="first")
    if args.max_target_dates:
        keep_dates = sorted(frame["date"].unique())[: args.max_target_dates]
        frame = frame.loc[frame["date"].isin(keep_dates)].copy()
    return frame.reset_index(drop=True)


def find_daily_zip(variable: str, day: date, daily_root: Path) -> Path | None:
    token = day.strftime("%Y%m%d")
    variable_dir = daily_root / variable
    patterns = [
        f"prism_{variable}_*_{token}.zip",
        f"*{variable}*{token}*.zip",
    ]
    roots = [variable_dir] if variable_dir.exists() else []
    roots.append(daily_root)
    seen: set[Path] = set()
    for root in roots:
        if root in seen or not root.exists():
            continue
        seen.add(root)
        for pattern in patterns:
            matches = sorted(root.rglob(pattern))
            if matches:
                return matches[0]
    return None


def extract_raster(zip_path: Path, temp_root: Path) -> Path:
    if not zipfile.is_zipfile(zip_path):
        raise ValueError(f"Not a valid zip file: {zip_path}")
    target_dir = temp_root / zip_path.stem
    if target_dir.exists():
        shutil.rmtree(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(target_dir)
    candidates = []
    for pattern in ("*.bil", "*.tif", "*.tiff"):
        candidates.extend(target_dir.rglob(pattern))
    if not candidates:
        raise FileNotFoundError(f"No raster file found after extracting {zip_path}")
    return sorted(candidates)[0]


def raster_grid_key(dataset) -> tuple:
    transform_values = tuple(round(float(value), 12) for value in dataset.transform[:6])
    return (int(dataset.width), int(dataset.height), str(dataset.crs or ""), transform_values)


def grid_indices_for_dataset(dataset, city_cells: CityCells, args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    try:
        from rasterio.transform import rowcol
    except ImportError as exc:
        raise RuntimeError("Could not import rasterio.transform.rowcol.") from exc

    xs = city_cells.lon.astype(float).tolist()
    ys = city_cells.lat.astype(float).tolist()
    crs_text = str(dataset.crs or "").upper()
    if args.reproject_points and dataset.crs and crs_text not in {"EPSG:4326", "EPSG:4269"}:
        from rasterio.warp import transform as rasterio_transform

        xs, ys = rasterio_transform("EPSG:4326", dataset.crs, xs, ys)
    rows, cols = rowcol(dataset.transform, xs, ys)
    rows = np.asarray(rows, dtype=np.int64)
    cols = np.asarray(cols, dtype=np.int64)
    valid = (rows >= 0) & (rows < int(dataset.height)) & (cols >= 0) & (cols < int(dataset.width))
    return rows, cols, valid


def sample_raster(
    raster_path: Path,
    city_cells: CityCells,
    args: argparse.Namespace,
    grid_cache: dict[tuple, tuple[np.ndarray, np.ndarray, np.ndarray]],
) -> np.ndarray:
    try:
        import rasterio
    except ImportError as exc:
        raise RuntimeError("This script requires rasterio in the active Python environment.") from exc

    values = np.full(city_cells.row_index.shape[0], np.nan, dtype=np.float32)
    with rasterio.open(raster_path) as dataset:
        nodata = dataset.nodata
        key = raster_grid_key(dataset)
        if key not in grid_cache:
            grid_cache[key] = grid_indices_for_dataset(dataset, city_cells, args)
        rows, cols, valid = grid_cache[key]
        band = np.asarray(dataset.read(1), dtype=np.float32)
        sampled = band[rows[valid], cols[valid]]
        if nodata is not None:
            sampled = np.where(sampled == float(nodata), np.nan, sampled)
        sampled = np.where(np.isfinite(sampled), sampled, np.nan).astype(np.float32, copy=False)
        values[valid] = sampled
    return values


def sample_daily_variable(
    variable: str,
    day: date,
    city_cells: CityCells,
    temp_root: Path,
    args: argparse.Namespace,
    grid_cache: dict[tuple, tuple[np.ndarray, np.ndarray, np.ndarray]],
) -> np.ndarray | None:
    zip_path = find_daily_zip(variable, day, args.prism_daily_root)
    if zip_path is None:
        if args.missing_daily_policy == "error":
            raise FileNotFoundError(f"Missing PRISM daily ZIP for {variable} {day:%Y%m%d} under {args.prism_daily_root}")
        if args.missing_daily_policy == "skip-date":
            return None
        return np.full(city_cells.row_index.shape[0], np.nan, dtype=np.float32)
    raster_path = extract_raster(zip_path, temp_root)
    try:
        return sample_raster(raster_path, city_cells, args, grid_cache)
    finally:
        if not args.keep_temp:
            shutil.rmtree(raster_path.parent, ignore_errors=True)


def resolve_temperature_source(variables: set[str], args: argparse.Namespace) -> str:
    if args.temperature_source == "tmean":
        if "tmean" not in variables:
            raise ValueError("--temperature-source tmean requires --variables to include tmean")
        return "tmean"
    if args.temperature_source == "tmin_tmax":
        missing = sorted({"tmin", "tmax"} - variables)
        if missing:
            raise ValueError(f"--temperature-source tmin_tmax requires --variables to include tmin,tmax; missing {missing}")
        return "tmin_tmax"
    if "tmean" in variables:
        return "tmean"
    missing = sorted({"tmin", "tmax"} - variables)
    if not missing:
        return "tmin_tmax"
    raise ValueError(
        "--temperature-source auto needs either tmean or both tmin,tmax in --variables; "
        f"got {sorted(variables)}"
    )


def load_monthly_normals(city_cells: CityCells, args: argparse.Namespace) -> dict[str, np.ndarray]:
    path = args.prism_normals_dir / city_cells.city / f"{city_cells.city}_prism_normals.npz"
    if not path.exists():
        print(f"{city_cells.city}: no PRISM normals sidecar found; anomaly features will be NaN: {path}", flush=True)
        return {}
    with np.load(path, allow_pickle=True) as data:
        normal_rows = np.asarray(data["reduced_id"], dtype=np.int64)
        feature_names = [str(value) for value in data["feature_names"].tolist()]
        normal_values = np.asarray(data["values"], dtype=np.float32)
    lookup = {int(row): index for index, row in enumerate(normal_rows)}
    order = np.asarray([lookup.get(int(row), -1) for row in city_cells.row_index], dtype=np.int64)
    out: dict[str, np.ndarray] = {}
    for variable in ("ppt", "vpdmax"):
        matrix = np.full((len(city_cells.row_index), 12), np.nan, dtype=np.float32)
        for month in range(1, 13):
            name = f"prism_normal_{variable}_m{month:02d}"
            if name not in feature_names:
                continue
            source = normal_values[:, feature_names.index(name)]
            valid = order >= 0
            matrix[valid, month - 1] = source[order[valid]]
        out[variable] = matrix
    return out


def daily_normal_values(normals: dict[str, np.ndarray], variable: str, day: date, count: int) -> np.ndarray:
    if variable not in normals:
        return np.full(count, np.nan, dtype=np.float32)
    monthly = normals[variable][:, day.month - 1]
    if variable == "ppt":
        return (monthly / float(calendar.monthrange(day.year, day.month)[1])).astype(np.float32)
    return monthly.astype(np.float32, copy=False)


def finite_or_zero(values: np.ndarray) -> np.ndarray:
    return np.where(np.isfinite(values), values, 0.0).astype(np.float32)


def append_and_trim(queue: deque[np.ndarray], value: np.ndarray, max_len: int) -> None:
    queue.append(value)
    while len(queue) > max_len:
        queue.popleft()


def stack_sum(queue: deque[np.ndarray], count: int) -> np.ndarray:
    if not queue:
        return np.full(count, np.nan, dtype=np.float32)
    return np.sum(np.stack(list(queue), axis=0), axis=0, dtype=np.float32)


def stack_mean(queue: deque[np.ndarray], count: int) -> np.ndarray:
    if not queue:
        return np.full(count, np.nan, dtype=np.float32)
    stack = np.stack(list(queue), axis=0)
    finite = np.isfinite(stack)
    sums = np.where(finite, stack, 0.0).sum(axis=0, dtype=np.float32)
    counts = finite.sum(axis=0)
    return np.divide(sums, counts, out=np.full(count, np.nan, dtype=np.float32), where=counts > 0)


def stack_max(queue: deque[np.ndarray], count: int) -> np.ndarray:
    if not queue:
        return np.full(count, np.nan, dtype=np.float32)
    stack = np.stack(list(queue), axis=0)
    return np.nanmax(stack, axis=0).astype(np.float32)


def build_feature_table(
    city_cells: CityCells,
    targets: pd.DataFrame,
    temp_root: Path,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if targets.empty:
        return (
            np.zeros(0, dtype=np.int64),
            np.zeros(0, dtype=np.int32),
            np.zeros((0, len(FEATURE_NAMES)), dtype=np.float32),
        )

    variables = {part.strip().lower() for part in args.variables.split(",") if part.strip()}
    temperature_source = resolve_temperature_source(variables, args)
    required_variables = {"ppt", "vpdmax", "tmean"} if temperature_source == "tmean" else {"ppt", "vpdmax", "tmin", "tmax"}
    missing_vars = sorted(required_variables - variables)
    if missing_vars:
        raise ValueError(
            f"--variables must include {sorted(required_variables)} for temperature_source={temperature_source}; "
            f"missing {missing_vars}"
        )

    target_dates = sorted(targets["date"].unique())
    min_day = min(target_dates) - timedelta(days=89)
    max_day = max(target_dates)
    target_by_date = {day: group.index.to_numpy(dtype=np.int64) for day, group in targets.groupby("date", sort=False)}
    cell_position = {int(row): index for index, row in enumerate(city_cells.row_index)}
    target_cell_pos = targets["row_index"].map(cell_position).to_numpy(dtype=np.int64)

    row_out = np.full(len(targets), -1, dtype=np.int64)
    date_out = np.full(len(targets), -1, dtype=np.int32)
    values_out = np.full((len(targets), len(FEATURE_NAMES)), np.nan, dtype=np.float32)

    normals = load_monthly_normals(city_cells, args)
    cell_count = len(city_cells.row_index)
    gdd_ytd = np.zeros(cell_count, dtype=np.float32)
    current_year: int | None = None
    gdd30: deque[np.ndarray] = deque()
    ppt30: deque[np.ndarray] = deque()
    ppt90: deque[np.ndarray] = deque()
    dry30: deque[np.ndarray] = deque()
    vpd30: deque[np.ndarray] = deque()
    normal_ppt30: deque[np.ndarray] = deque()
    normal_vpd30: deque[np.ndarray] = deque()
    grid_cache: dict[tuple, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    started = time.perf_counter()

    for day_index, day in enumerate(date_range(min_day, max_day), start=1):
        if current_year != day.year:
            gdd_ytd = np.zeros(cell_count, dtype=np.float32)
            current_year = day.year

        if day_index == 1 or day.day == 1 or day in target_by_date:
            print(
                f"{city_cells.city}: PRISM daily context {day:%Y-%m-%d}; "
                f"elapsed={time.perf_counter() - started:.1f}s",
                flush=True,
            )

        ppt = sample_daily_variable("ppt", day, city_cells, temp_root, args, grid_cache)
        vpdmax = sample_daily_variable("vpdmax", day, city_cells, temp_root, args, grid_cache)
        if temperature_source == "tmean":
            tmean = sample_daily_variable("tmean", day, city_cells, temp_root, args, grid_cache)
            temp_for_gdd = tmean
        else:
            tmin = sample_daily_variable("tmin", day, city_cells, temp_root, args, grid_cache)
            tmax = sample_daily_variable("tmax", day, city_cells, temp_root, args, grid_cache)
            temp_for_gdd = None if tmin is None or tmax is None else (tmin + tmax) * 0.5
        if ppt is None or temp_for_gdd is None or vpdmax is None:
            print(f"{city_cells.city}: skipping {day:%Y-%m-%d}; at least one daily PRISM raster is missing", flush=True)
            continue

        gdd = np.maximum(temp_for_gdd - float(args.gdd_base_c), 0.0).astype(np.float32)
        gdd = np.where(np.isfinite(gdd), gdd, 0.0).astype(np.float32)
        gdd_ytd += gdd

        dry = np.where(np.isfinite(ppt) & (ppt < float(args.dry_day_threshold_mm)), 1.0, 0.0).astype(np.float32)
        append_and_trim(gdd30, gdd, 30)
        append_and_trim(ppt30, finite_or_zero(ppt), 30)
        append_and_trim(ppt90, finite_or_zero(ppt), 90)
        append_and_trim(dry30, dry, 30)
        append_and_trim(vpd30, vpdmax.astype(np.float32, copy=False), 30)
        append_and_trim(normal_ppt30, daily_normal_values(normals, "ppt", day, cell_count), 30)
        append_and_trim(normal_vpd30, daily_normal_values(normals, "vpdmax", day, cell_count), 30)

        target_indices = target_by_date.get(day)
        if target_indices is None:
            continue

        ppt30_sum = stack_sum(ppt30, cell_count)
        vpd30_mean = stack_mean(vpd30, cell_count)
        normal_ppt30_sum = stack_sum(normal_ppt30, cell_count)
        normal_vpd30_mean = stack_mean(normal_vpd30, cell_count)
        daily_features = np.column_stack(
            [
                gdd_ytd,
                stack_sum(gdd30, cell_count),
                ppt30_sum,
                stack_sum(ppt90, cell_count),
                stack_sum(dry30, cell_count),
                vpd30_mean,
                stack_max(vpd30, cell_count),
                ppt30_sum - normal_ppt30_sum,
                vpd30_mean - normal_vpd30_mean,
            ]
        ).astype(np.float32)
        cell_indices = target_cell_pos[target_indices]
        row_out[target_indices] = targets["row_index"].to_numpy(dtype=np.int64)[target_indices]
        date_out[target_indices] = int(day.strftime("%Y%m%d"))
        values_out[target_indices] = daily_features[cell_indices]

    keep = row_out >= 0
    return row_out[keep], date_out[keep], values_out[keep]


def write_feature_manifest(path: Path) -> None:
    fields = ["feature_index", "feature_name", "description"]
    descriptions = {
        "prism_gdd5_ytd": "Cumulative growing degree days since Jan 1 using daily mean temperature and base 5C.",
        "prism_gdd5_30d": "Growing degree days over the current and previous 29 days, base 5C.",
        "prism_ppt_30d_mm": "Precipitation total over the current and previous 29 days.",
        "prism_ppt_90d_mm": "Precipitation total over the current and previous 89 days.",
        "prism_dry_days_30d": "Count of days in the current and previous 29 days with precipitation below the dry-day threshold.",
        "prism_vpdmax_30d_mean": "Mean daily maximum vapor pressure deficit over the current and previous 29 days.",
        "prism_vpdmax_30d_max": "Maximum daily maximum vapor pressure deficit over the current and previous 29 days.",
        "prism_ppt_30d_anomaly_mm": "30-day precipitation minus monthly-normal expected precipitation over the same dates.",
        "prism_vpdmax_30d_anomaly": "30-day mean daily maximum VPD minus monthly-normal expected daily maximum VPD.",
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for index, name in enumerate(FEATURE_NAMES):
            writer.writerow({"feature_index": index, "feature_name": name, "description": descriptions[name]})


def write_city_outputs(
    city_cells: CityCells,
    row_index: np.ndarray,
    date_int: np.ndarray,
    values: np.ndarray,
    args: argparse.Namespace,
) -> Path:
    city_dir = args.output_dir / city_cells.city
    city_dir.mkdir(parents=True, exist_ok=True)
    path = city_dir / f"{city_cells.city}_prism_daily_context.npz"
    np.savez_compressed(
        path,
        row_index=row_index.astype(np.int64, copy=False),
        date=date_int.astype(np.int32, copy=False),
        feature_names=np.asarray(FEATURE_NAMES, dtype=object),
        values=values.astype(np.float32, copy=False),
        source="PRISM daily raster context aligned to Sentinel dates",
        gdd_base_c=np.asarray([float(args.gdd_base_c)], dtype=np.float32),
        dry_day_threshold_mm=np.asarray([float(args.dry_day_threshold_mm)], dtype=np.float32),
    )
    metadata = {
        "city": city_cells.city,
        "rows": int(values.shape[0]),
        "features": list(FEATURE_NAMES),
        "gdd_base_c": float(args.gdd_base_c),
        "dry_day_threshold_mm": float(args.dry_day_threshold_mm),
        "prism_daily_root": str(args.prism_daily_root),
        "prism_normals_dir": str(args.prism_normals_dir),
    }
    (city_dir / f"{city_cells.city}_prism_daily_context_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    if args.write_csv:
        csv_path = city_dir / f"{city_cells.city}_prism_daily_context.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(["row_index", "date", *FEATURE_NAMES])
            for i in range(values.shape[0]):
                writer.writerow([int(row_index[i]), int(date_int[i]), *[f"{v:.7g}" for v in values[i]]])
        print(f"Wrote CSV: {csv_path}", flush=True)
    return path


def main() -> int:
    args = parse_args()
    city_rows = list(iter_city_cells(args))
    if not city_rows:
        raise SystemExit("No cities matched the requested filters.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_feature_manifest(args.output_dir / "prism_daily_context_feature_manifest.csv")

    print(
        f"PRISM daily context derivation: cities={len(city_rows):,}; "
        f"features={len(FEATURE_NAMES):,}; daily_root={args.prism_daily_root}",
        flush=True,
    )

    if args.dry_run:
        for city_cells in city_rows:
            targets = read_target_rows(city_cells, args)
            if targets.empty:
                print(f"{city_cells.city}: cells={len(city_cells.row_index):,}; target_rows=0", flush=True)
                continue
            print(
                f"{city_cells.city}: cells={len(city_cells.row_index):,}; "
                f"target_rows={len(targets):,}; target_dates={targets['date'].nunique():,}; "
                f"date_range={targets['date'].min()}..{targets['date'].max()}",
                flush=True,
            )
        return 0

    with tempfile.TemporaryDirectory(prefix="prism_daily_context_") as temp_name:
        temp_root = Path(temp_name)
        if args.keep_temp:
            temp_root = args.output_dir / "_temp_prism_daily_context"
            temp_root.mkdir(parents=True, exist_ok=True)
        for city_cells in city_rows:
            output_path = args.output_dir / city_cells.city / f"{city_cells.city}_prism_daily_context.npz"
            if output_path.exists() and not args.overwrite:
                print(f"Skipping existing {output_path}", flush=True)
                continue
            targets = read_target_rows(city_cells, args)
            if targets.empty:
                print(f"{city_cells.city}: no Sentinel target dates; skipping", flush=True)
                continue
            print(
                f"{city_cells.city}: deriving {len(targets):,} target row-date(s), "
                f"{targets['date'].nunique():,} date(s), {len(city_cells.row_index):,} cell(s)",
                flush=True,
            )
            row_index, date_int, values = build_feature_table(city_cells, targets, temp_root, args)
            written = write_city_outputs(city_cells, row_index, date_int, values, args)
            print(f"Wrote {city_cells.city}: {written} ({values.shape[0]:,} rows x {values.shape[1]:,} features)", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
