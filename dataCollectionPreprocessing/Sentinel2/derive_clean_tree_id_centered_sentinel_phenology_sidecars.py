#!/usr/bin/env python3
"""Derive clean tree-id-centered Sentinel phenology sidecars from time series.

This script is for the rebuilt clean tree-centered dataset. It intentionally
does not rely on old tree-centered row ids or cell-centered row ids. Clean tree
records are linked to Sentinel time-series cells by Sentinel center latitude and
longitude, then phenology metrics are computed with the same helper used by the
cell-centered pipeline.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import math
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import pandas as pd
from pyproj import Transformer

HERE = Path(__file__).resolve().parent
SHARD_SCRIPTS = HERE.parents[1] / "dataCollectionPreprocessing" / "Shard"
if str(SHARD_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SHARD_SCRIPTS))

import assemble_clean_tree_id_centered_model_input_shards as shard_schema
import sentinel_phenology_metrics as derived_metrics


DEFAULT_CROP_ROOT = Path(r"H:\TreeCenteredModelInputs\tree_centered_naip_crops_clean")
DEFAULT_CLEAN_SENTINEL_LINK_ROOT = Path(r"H:\TreeCenteredModelInputs\tree_record_metadata_clean")
DEFAULT_ORIGINAL_TIMESERIES_ROOT = Path(r"E:\cell\sentinel2_timeseries")
DEFAULT_SUPPLEMENTAL_TIMESERIES_ROOT = Path(r"E:\TreeCenteredModelInputs\tree_centered_sentinel_timeseries_supplemental")
DEFAULT_OUTPUT_DIR = Path(r"H:\TreeCenteredModelInputs\tree_centered_sentinel_phenology_clean")

COORD_SCALE = 10_000_000
EARTH_RADIUS_M = 6_371_008.8
CITY_ALIASES = {
    "abq": "albuquerque",
    "ana": "anaheim",
    "arl": "arlington",
    "atl": "atlanta",
    "bal": "baltimore",
    "buf": "buffalo",
    "cc": "capecoral",
    "cos": "coloradosprings",
    "csp": "coloradosprings",
    "dca": "washingtondc",
    "dc": "washingtondc",
    "den": "denver",
    "ggv": "gardengrove",
    "hnb": "huntingtonbeach",
    "la": "losangeles",
    "lax": "losangeles",
    "nyc": "newyork",
    "sf": "sanfrancisco",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--crop-root", type=Path, default=DEFAULT_CROP_ROOT)
    parser.add_argument(
        "--clean-sentinel-link-root",
        type=Path,
        default=DEFAULT_CLEAN_SENTINEL_LINK_ROOT,
        help="Deprecated; retained for command compatibility. Clean linkage is derived from tree_id crop metadata rows.",
    )
    parser.add_argument("--original-timeseries-root", type=Path, default=DEFAULT_ORIGINAL_TIMESERIES_ROOT)
    parser.add_argument("--supplemental-timeseries-root", type=Path, default=DEFAULT_SUPPLEMENTAL_TIMESERIES_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--city-token", action="append", default=[])
    parser.add_argument("--exclude-city-token", action="append", default=[])
    parser.add_argument("--parallel-workers", type=int, default=1)
    parser.add_argument("--coordinate-decimals", type=int, default=7)
    parser.add_argument("--max-coordinate-distance-m", type=float, default=1.0)
    parser.add_argument("--outlier-abs", type=float, default=0.0)
    parser.add_argument("--progress-interval", type=int, default=1000)
    parser.add_argument(
        "--sentinel-anchor",
        choices=("crown", "tree"),
        default="crown",
        help=(
            "Coordinate used to choose the Sentinel cell. The clean tree-centered "
            "dataset should use crown, because the model inputs are centered on "
            "the detected crown rather than the inventory coordinate."
        ),
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def norm_city(value: str) -> str:
    token = value.strip().lower().replace("-", "").replace("_", "")
    return CITY_ALIASES.get(token, token)


def discover_cities(crop_root: Path) -> list[str]:
    if not crop_root.exists():
        raise FileNotFoundError(crop_root)
    cities: list[str] = []
    for path in crop_root.iterdir():
        if path.is_dir() and find_crop_metadata(crop_root, path.name, missing_ok=True) is not None:
            cities.append(path.name)
    return sorted(cities)


def find_crop_metadata(crop_root: Path, city: str, missing_ok: bool = False) -> Path | None:
    city_dir = crop_root / city
    candidates = sorted(city_dir.glob(f"{city}_tree_id_centered_nearest_64px_metadata.csv"))
    if not candidates:
        candidates = sorted(city_dir.glob("*tree_id_centered*metadata.csv"))
    if candidates:
        return candidates[0]
    if missing_ok:
        return None
    raise FileNotFoundError(f"No clean crop metadata found under {city_dir}")


def time_series_files(root: Path, city: str) -> list[Path]:
    city_dir = root / city
    if not city_dir.exists():
        return []
    files = sorted(city_dir.glob("*sentinel2_15day_time_series_batch_*.csv"))
    return [path for path in files if "_summary_" not in path.name]


def coord_key(lat: pd.Series | np.ndarray, lon: pd.Series | np.ndarray, decimals: int) -> np.ndarray:
    scale = 10**decimals
    lat_i = np.rint(np.asarray(lat, dtype=np.float64) * scale).astype(np.int64)
    lon_i = np.rint(np.asarray(lon, dtype=np.float64) * scale).astype(np.int64)
    return lat_i.astype(str) + "_" + lon_i.astype(str)


def haversine_m(lat1: np.ndarray, lon1: np.ndarray, lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
    lat1r = np.radians(lat1.astype(np.float64))
    lon1r = np.radians(lon1.astype(np.float64))
    lat2r = np.radians(lat2.astype(np.float64))
    lon2r = np.radians(lon2.astype(np.float64))
    dlat = lat2r - lat1r
    dlon = lon2r - lon1r
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1r) * np.cos(lat2r) * np.sin(dlon / 2.0) ** 2
    return 2.0 * EARTH_RADIUS_M * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def read_required_columns(path: Path) -> list[str]:
    return list(pd.read_csv(path, nrows=0).columns)


def load_timeseries_source(
    city: str,
    label: str,
    files: list[Path],
    needed_keys: set[str],
    decimals: int,
) -> pd.DataFrame:
    if not files:
        return pd.DataFrame()
    base_columns = {"latitude", "longitude", "date", *shard_schema.SENTINEL_SEQUENCE_COLUMNS}
    frames: list[pd.DataFrame] = []
    started = time.perf_counter()
    for index, path in enumerate(files, start=1):
        header = read_required_columns(path)
        usecols = [column for column in header if column in base_columns]
        if "latitude" not in usecols or "longitude" not in usecols or "date" not in usecols:
            print(f"{city}: WARNING {label} file missing coordinate/date columns: {path}", flush=True)
            continue
        frame = pd.read_csv(path, usecols=usecols)
        frame["sentinel_coord_key"] = coord_key(frame["latitude"], frame["longitude"], decimals)
        frame = frame[frame["sentinel_coord_key"].isin(needed_keys)]
        if not frame.empty:
            frame["sentinel_source_label"] = label
            frames.append(frame)
        if index == 1 or index == len(files) or index % 25 == 0:
            print(
                f"{city}: loaded {label} time-series file {index:,}/{len(files):,}; "
                f"kept_rows={len(frame):,}; elapsed={time.perf_counter() - started:.1f}s",
                flush=True,
            )
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True, copy=False)
    print(
        f"{city}: {label} matched time-series rows={len(out):,}; "
        f"cells={out['sentinel_coord_key'].nunique():,}",
        flush=True,
    )
    return out


def ensure_sequence_columns(frame: pd.DataFrame) -> pd.DataFrame:
    for column in shard_schema.SENTINEL_SEQUENCE_COLUMNS:
        if column not in frame.columns:
            frame[column] = 0.0
        frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(0.0)
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    return frame


def compute_phenology_cache(
    city: str,
    frame: pd.DataFrame,
    outlier_abs: float,
    progress_interval: int,
) -> dict[str, np.ndarray]:
    cache: dict[str, np.ndarray] = {}
    if frame.empty:
        return cache
    frame = ensure_sequence_columns(frame)
    groups = frame.groupby("sentinel_coord_key", sort=False)
    total = groups.ngroups
    started = time.perf_counter()
    for pos, (key, group) in enumerate(groups, start=1):
        group = group.sort_values("date", kind="stable")
        values = group[list(shard_schema.SENTINEL_SEQUENCE_COLUMNS)].to_numpy(dtype=np.float32, copy=True)
        mask = np.ones(values.shape[0], dtype=bool)
        cache[str(key)] = derived_metrics.compute_sentinel_phenology(
            values,
            mask,
            list(shard_schema.SENTINEL_SEQUENCE_COLUMNS),
            float(outlier_abs),
        ).astype(np.float32)
        if progress_interval and (pos == 1 or pos == total or pos % progress_interval == 0):
            elapsed = time.perf_counter() - started
            print(
                f"{city}: computed phenology {pos:,}/{total:,} Sentinel cells; "
                f"rate={pos / max(elapsed, 1e-6):.1f}/s",
                flush=True,
            )
    return cache


def sentinel_centers_from_utm(
    x: pd.Series,
    y: pd.Series,
    epsg: pd.Series,
) -> pd.DataFrame:
    x_num = pd.to_numeric(x, errors="coerce")
    y_num = pd.to_numeric(y, errors="coerce")
    epsg_num = pd.to_numeric(epsg, errors="coerce")
    col = np.floor(x_num.to_numpy(dtype=np.float64) / 10.0).astype(np.float64)
    row = np.floor(y_num.to_numpy(dtype=np.float64) / 10.0).astype(np.float64)
    center_x = col * 10.0 + 5.0
    center_y = row * 10.0 + 5.0
    out_lat = np.full(len(x_num), np.nan, dtype=np.float64)
    out_lon = np.full(len(x_num), np.nan, dtype=np.float64)
    cell_ids = np.full(len(x_num), "", dtype=object)
    epsg_int = np.full(len(x_num), -1, dtype=np.int64)

    valid = np.isfinite(center_x) & np.isfinite(center_y) & np.isfinite(epsg_num.to_numpy(dtype=np.float64))
    for epsg_value in sorted(set(epsg_num[valid].astype(int))):
        mask = valid & (epsg_num.to_numpy(dtype=np.float64).astype(np.int64) == int(epsg_value))
        transformer = Transformer.from_crs(f"EPSG:{int(epsg_value)}", "EPSG:4326", always_xy=True)
        lon, lat = transformer.transform(center_x[mask], center_y[mask])
        out_lat[mask] = np.asarray(lat, dtype=np.float64)
        out_lon[mask] = np.asarray(lon, dtype=np.float64)
        epsg_int[mask] = int(epsg_value)
        cols = np.floor(center_x[mask] / 10.0).astype(np.int64)
        rows = np.floor(center_y[mask] / 10.0).astype(np.int64)
        cell_ids[mask] = [f"epsg{int(epsg_value)}_c{c}_r{r}" for c, r in zip(cols, rows)]

    return pd.DataFrame(
        {
            "sentinel_lat": out_lat,
            "sentinel_lon": out_lon,
            "sentinel_cell_id": cell_ids,
            "sentinel_epsg": np.where(epsg_int >= 0, np.char.add("EPSG:", epsg_int.astype(str)), ""),
            "sentinel_center_x": center_x,
            "sentinel_center_y": center_y,
        }
    )


def prepare_index(crop_path: Path, sentinel_anchor: str) -> pd.DataFrame:
    crop = pd.read_csv(crop_path)
    if "tree_id" not in crop.columns:
        raise RuntimeError(f"{crop_path} is missing tree_id")
    keep_crop = [
        column
        for column in (
            "tree_id",
            "crop_index",
            "row_index",
            "source_file",
            "source_row",
            "tree_lat",
            "tree_lon",
            "tree_x",
            "tree_y",
            "crown_lat",
            "crown_lon",
            "crown_x_utm",
            "crown_y_utm",
            "crown_epsg",
        )
        if column in crop.columns
    ]
    index = crop[keep_crop].copy()
    index["tree_sentinel_lat"] = np.nan
    index["tree_sentinel_lon"] = np.nan
    index["tree_sentinel_cell_id"] = ""

    if sentinel_anchor == "crown":
        required = {"crown_x_utm", "crown_y_utm", "crown_epsg"}
        missing_crop = sorted(required.difference(index.columns))
        if missing_crop:
            raise RuntimeError(f"{crop_path} is missing crown Sentinel anchor column(s): {missing_crop}")
        crown_centers = sentinel_centers_from_utm(index["crown_x_utm"], index["crown_y_utm"], index["crown_epsg"])
        for column in ("sentinel_lat", "sentinel_lon", "sentinel_cell_id"):
            index[column] = crown_centers[column]
        index["sentinel_anchor"] = "crown"
    else:
        raise RuntimeError("Clean Sentinel phenology derivation no longer trusts old tree-to-Sentinel link metadata; use --sentinel-anchor crown.")

    index["missing_clean_sentinel_link"] = index["sentinel_lat"].isna() | index["sentinel_lon"].isna()
    return index


def as_bool_array(series: pd.Series) -> np.ndarray:
    if series.dtype == bool:
        return series.to_numpy(dtype=bool)
    return series.astype(str).str.lower().isin({"true", "1", "yes"}).to_numpy(dtype=bool)


def process_city(city: str, args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    out_dir = Path(args.output_dir) / city
    out_path = out_dir / f"{city}_tree_id_centered_sentinel_phenology.npz"
    if out_path.exists() and not args.force:
        with np.load(out_path, allow_pickle=True) as data:
            missing = int(np.asarray(data["missing_sentinel_phenology"], dtype=bool).sum())
            rows = int(data["sentinel_phenology"].shape[0])
        return {"city_token": city, "status": "exists", "rows": rows, "missing": missing, "elapsed_sec": 0.0}

    crop_path = find_crop_metadata(Path(args.crop_root), city)
    index = prepare_index(crop_path, str(args.sentinel_anchor))
    index["sentinel_coord_key"] = coord_key(index["sentinel_lat"], index["sentinel_lon"], int(args.coordinate_decimals))
    needed_keys = set(index.loc[~index["missing_clean_sentinel_link"], "sentinel_coord_key"].astype(str))
    print(f"{city}: clean rows={len(index):,}; unique Sentinel cells needed={len(needed_keys):,}", flush=True)

    original_files = time_series_files(Path(args.original_timeseries_root), city)
    supplemental_files = time_series_files(Path(args.supplemental_timeseries_root), city)
    print(
        f"{city}: time-series files original={len(original_files):,}; supplemental={len(supplemental_files):,}",
        flush=True,
    )

    original = load_timeseries_source(city, "old", original_files, needed_keys, int(args.coordinate_decimals))
    supplemental = load_timeseries_source(city, "supplemental", supplemental_files, needed_keys, int(args.coordinate_decimals))
    old_cache = compute_phenology_cache(city, original, float(args.outlier_abs), int(args.progress_interval))
    supplemental_cache = compute_phenology_cache(city, supplemental, float(args.outlier_abs), int(args.progress_interval))

    pheno = np.zeros((len(index), len(derived_metrics.SENTINEL_PHENOLOGY_COLUMNS)), dtype=np.float32)
    missing = np.ones(len(index), dtype=bool)
    source_code = np.zeros(len(index), dtype=np.int8)
    match_distance = np.full(len(index), np.nan, dtype=np.float32)
    source_label = np.full(len(index), "", dtype=object)

    keys = index["sentinel_coord_key"].astype(str).to_numpy()
    sentinel_lat = pd.to_numeric(index["sentinel_lat"], errors="coerce").to_numpy(dtype=np.float64)
    sentinel_lon = pd.to_numeric(index["sentinel_lon"], errors="coerce").to_numpy(dtype=np.float64)

    for row, key in enumerate(keys):
        value = None
        code = 0
        label = ""
        value = old_cache.get(key)
        code = 1 if value is not None else 0
        label = "old" if value is not None else ""
        if value is None:
            value = supplemental_cache.get(key)
            code = 2 if value is not None else 0
            label = "supplemental" if value is not None else ""
        if value is not None:
            pheno[row] = value
            missing[row] = False
            source_code[row] = code
            source_label[row] = label
            match_distance[row] = 0.0 if math.isfinite(sentinel_lat[row]) and math.isfinite(sentinel_lon[row]) else np.nan

    out_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "city_token": city,
        "crop_metadata": str(crop_path),
        "clean_sentinel_link_csv": "",
        "linkage_note": "Sentinel cells derived from crop metadata tree_id rows and crown UTM coordinates; old link metadata ignored.",
        "original_timeseries_root": str(args.original_timeseries_root),
        "supplemental_timeseries_root": str(args.supplemental_timeseries_root),
        "coordinate_decimals": int(args.coordinate_decimals),
        "max_coordinate_distance_m": float(args.max_coordinate_distance_m),
        "outlier_abs": float(args.outlier_abs),
        "sentinel_anchor": str(args.sentinel_anchor),
        "join_policy": "sentinel center latitude/longitude rounded coordinate key; ids retained only as metadata",
    }
    np.savez_compressed(
        out_path,
        tree_id=index["tree_id"].to_numpy(),
        crop_index=index["crop_index"].to_numpy() if "crop_index" in index else np.arange(len(index), dtype=np.int64),
        row_index=index["row_index"].to_numpy() if "row_index" in index else np.arange(len(index), dtype=np.int64),
        sentinel_cell_id=index["sentinel_cell_id"].astype(str).to_numpy(),
        sentinel_lat=sentinel_lat.astype(np.float64),
        sentinel_lon=sentinel_lon.astype(np.float64),
        sentinel_source=np.full(len(index), "", dtype=object),
        sentinel_source_used=source_label,
        sentinel_anchor=index["sentinel_anchor"].astype(str).to_numpy(),
        tree_sentinel_cell_id=index["tree_sentinel_cell_id"].astype(str).to_numpy(),
        tree_sentinel_lat=pd.to_numeric(index["tree_sentinel_lat"], errors="coerce").to_numpy(dtype=np.float64),
        tree_sentinel_lon=pd.to_numeric(index["tree_sentinel_lon"], errors="coerce").to_numpy(dtype=np.float64),
        sentinel_phenology_source_code=source_code,
        sentinel_timeseries_match_distance_m=match_distance,
        sentinel_phenology=pheno,
        sentinel_phenology_columns=np.asarray(derived_metrics.SENTINEL_PHENOLOGY_COLUMNS),
        missing_clean_sentinel_link=index["missing_clean_sentinel_link"].to_numpy(dtype=bool),
        missing_sentinel_phenology=missing,
        config_json=np.asarray(json.dumps(config, indent=2)),
    )

    missing_count = int(missing.sum())
    elapsed = time.perf_counter() - started
    print(
        f"{city}: wrote {out_path}; rows={len(index):,}; "
        f"missing_phenology={missing_count:,} ({missing_count / max(len(index), 1):.2%}); "
        f"elapsed={elapsed:.1f}s",
        flush=True,
    )
    return {
        "city_token": city,
        "status": "completed",
        "rows": int(len(index)),
        "unique_sentinel_cells_needed": int(len(needed_keys)),
        "old_cells_loaded": int(len(old_cache)),
        "supplemental_cells_loaded": int(len(supplemental_cache)),
        "missing": missing_count,
        "missing_pct": float(missing_count / max(len(index), 1) * 100.0),
        "elapsed_sec": float(elapsed),
        "output": str(out_path),
    }


def main() -> int:
    args = parse_args()
    cities = [norm_city(city) for city in args.city_token] if args.city_token else discover_cities(Path(args.crop_root))
    exclude = {norm_city(city) for city in args.exclude_city_token}
    cities = [city for city in cities if city not in exclude]
    if not cities:
        raise SystemExit("No city jobs selected.")
    print(f"Deriving clean Sentinel phenology for {len(cities):,} city/cities.", flush=True)

    results: list[dict[str, Any]] = []
    workers = max(1, int(args.parallel_workers))
    if workers == 1:
        for city in cities:
            try:
                results.append(process_city(city, args))
            except Exception as exc:  # noqa: BLE001
                print(f"{city}: FAILED {exc}", flush=True)
                results.append({"city_token": city, "status": "failed", "error": str(exc)})
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            future_map = {pool.submit(process_city, city, args): city for city in cities}
            for future in as_completed(future_map):
                city = future_map[future]
                try:
                    results.append(future.result())
                except Exception as exc:  # noqa: BLE001
                    print(f"{city}: FAILED {exc}", flush=True)
                    results.append({"city_token": city, "status": "failed", "error": str(exc)})

    summary = pd.DataFrame(results).sort_values("city_token")
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    summary_path = Path(args.output_dir) / "clean_tree_id_centered_sentinel_phenology_summary.csv"
    summary.to_csv(summary_path, index=False)
    print("\nSummary:", flush=True)
    print(summary["status"].value_counts(dropna=False).to_string(), flush=True)
    print(f"Wrote summary: {summary_path}", flush=True)
    return 1 if (summary["status"] == "failed").any() else 0


if __name__ == "__main__":
    raise SystemExit(main())
