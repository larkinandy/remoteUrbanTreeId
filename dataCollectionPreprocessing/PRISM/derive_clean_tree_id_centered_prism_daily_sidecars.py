#!/usr/bin/env python3
"""Extract daily PRISM sequences and tree-to-PRISM-pixel lookup sidecars.

The daily PRISM rasters are much coarser than the 10 m Sentinel cells. Storing
one daily sequence per tree would therefore duplicate the same climate values
many times. This script writes:

* one global NPZ containing daily values for every unique PRISM pixel required
  by the selected clean tree inventory; and
* one city NPZ mapping each tree_id and Sentinel cell to a row in that global
  daily-value array.

The default 2021-01-01 through 2023-12-31 interval matches the Sentinel sequence
period used by the clean multimodal model.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import date, datetime, timedelta
import json
from pathlib import Path
import re
from typing import Any

import numpy as np
import pandas as pd

try:
    import rasterio
except Exception as exc:  # pragma: no cover - reported clearly at runtime
    rasterio = None
    RASTERIO_IMPORT_ERROR = exc
else:
    RASTERIO_IMPORT_ERROR = None


CLEAN_ROOT = Path(r"H:\TreeCenteredModelInputs")
DEFAULT_LINK_ROOT = CLEAN_ROOT / "tree_record_metadata_clean"
DEFAULT_RAW_ROOT = Path(r"E:\PRISM\sentinel_cells\raw\daily")
DEFAULT_OUTPUT_DIR = CLEAN_ROOT / "tree_centered_prism_daily_clean"
DEFAULT_VARIABLES = ("ppt", "tmean", "vpdmax")

CITY_ALIASES = {
    "abq": "albuquerque",
    "ana": "anaheim",
    "arl": "arlington",
    "atl": "atlanta",
    "bal": "baltimore",
    "buf": "buffalo",
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
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--link-root", type=Path, default=DEFAULT_LINK_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--city-token", action="append", default=[])
    parser.add_argument("--exclude-city-token", action="append", default=[])
    parser.add_argument("--variable", action="append", default=[])
    parser.add_argument("--start-date", default="2021-01-01")
    parser.add_argument("--end-date", default="2023-12-31")
    parser.add_argument(
        "--link-pattern",
        default="{city}_tree_to_computed_sentinel_cells.csv",
        help="Filename pattern inside each city directory.",
    )
    parser.add_argument(
        "--max-dates",
        type=int,
        default=0,
        help="Optional smoke-test limit after date filtering; zero keeps all dates.",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def norm_city(value: object) -> str:
    token = re.sub(r"[^a-z0-9]+", "", str(value or "").lower())
    return CITY_ALIASES.get(token, token)


def parse_iso_date(value: str, label: str) -> date:
    try:
        return datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError as exc:
        raise ValueError(f"{label} must use YYYY-MM-DD; got {value!r}") from exc


def date_range(start: date, end: date) -> list[date]:
    if end < start:
        raise ValueError("--end-date must be on or after --start-date")
    return [start + timedelta(days=offset) for offset in range((end - start).days + 1)]


def discover_cities(root: Path, pattern: str) -> list[str]:
    if not root.exists():
        raise FileNotFoundError(root)
    cities = []
    for directory in sorted(path for path in root.iterdir() if path.is_dir()):
        city = norm_city(directory.name)
        if (directory / pattern.format(city=city, city_token=city)).exists():
            cities.append(city)
    return sorted(set(cities))


def link_path(root: Path, city: str, pattern: str) -> Path:
    path = root / city / pattern.format(city=city, city_token=city)
    if path.exists():
        return path
    matches = sorted((root / city).glob("*tree_to_computed_sentinel_cells.csv"))
    if matches:
        return matches[0]
    raise FileNotFoundError(f"No tree/Sentinel-cell link CSV for {city} under {root / city}")


def raster_archives(raw_root: Path, variable: str) -> dict[date, tuple[Path, str]]:
    directory = raw_root / variable
    if not directory.exists():
        raise FileNotFoundError(directory)
    pattern = re.compile(
        rf"^prism_{re.escape(variable)}_us_25m_(\d{{8}})\.zip$",
        flags=re.IGNORECASE,
    )
    out: dict[date, tuple[Path, str]] = {}
    for path in sorted(directory.glob("*.zip")):
        match = pattern.match(path.name)
        if not match:
            continue
        current = datetime.strptime(match.group(1), "%Y%m%d").date()
        tif_name = path.with_suffix(".tif").name
        out[current] = (path, tif_name)
    return out


def zip_raster_path(archive: Path, member: str) -> str:
    # rasterio/GDAL accepts this URI on Windows and reads the TIFF directly
    # from the archive without materializing thousands of extracted files.
    return f"zip://{archive.as_posix()}!{member}"


def require_columns(frame: pd.DataFrame, columns: tuple[str, ...], path: Path) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise RuntimeError(f"{path} is missing required column(s): {missing}")


def load_city_links(path: Path) -> pd.DataFrame:
    wanted = (
        "tree_id",
        "sentinel_cell_id",
        "sentinel_cell_col",
        "sentinel_cell_row",
        "sentinel_lat",
        "sentinel_lon",
    )
    frame = pd.read_csv(path, usecols=lambda column: column in wanted, low_memory=False)
    require_columns(frame, ("tree_id", "sentinel_lat", "sentinel_lon"), path)
    # A few clean city placeholders have no retained records and only carry the
    # minimal coordinate header. Keep them valid as empty lookup sidecars.
    if "sentinel_cell_id" not in frame:
        frame["sentinel_cell_id"] = ""
    if "sentinel_cell_col" not in frame:
        frame["sentinel_cell_col"] = -1
    if "sentinel_cell_row" not in frame:
        frame["sentinel_cell_row"] = -1
    if frame["tree_id"].duplicated().any():
        raise RuntimeError(f"{path} contains duplicated tree_id values")
    frame["sentinel_lat"] = pd.to_numeric(frame["sentinel_lat"], errors="coerce")
    frame["sentinel_lon"] = pd.to_numeric(frame["sentinel_lon"], errors="coerce")
    return frame


def raster_signature(dataset) -> dict[str, Any]:
    return {
        "crs": dataset.crs.to_string() if dataset.crs is not None else "",
        "width": int(dataset.width),
        "height": int(dataset.height),
        "transform": tuple(float(value) for value in dataset.transform),
        "nodata": None if dataset.nodata is None else float(dataset.nodata),
    }


def north_up_grid_rowcol(transform, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert lon/lat to raster indices without invoking GDAL/PROJ transforms."""
    if not np.isclose(float(transform.b), 0.0) or not np.isclose(float(transform.d), 0.0):
        raise RuntimeError(f"Expected a north-up PRISM transform; got {transform}")
    cols = np.floor((np.asarray(x, dtype=np.float64) - float(transform.c)) / float(transform.a))
    rows = np.floor((np.asarray(y, dtype=np.float64) - float(transform.f)) / float(transform.e))
    return rows.astype(np.int32), cols.astype(np.int32)


def same_grid(dataset, reference: dict[str, Any]) -> bool:
    candidate = raster_signature(dataset)
    return (
        candidate["crs"] == reference["crs"]
        and candidate["width"] == reference["width"]
        and candidate["height"] == reference["height"]
        and np.allclose(candidate["transform"], reference["transform"], atol=1.0e-12, rtol=0.0)
    )


def main() -> int:
    args = parse_args()
    if rasterio is None:
        raise SystemExit(f"rasterio is required. Original import error: {RASTERIO_IMPORT_ERROR}")

    variables = tuple(dict.fromkeys(str(value).strip().lower() for value in (args.variable or DEFAULT_VARIABLES)))
    if not variables:
        raise SystemExit("No PRISM variables selected.")
    selected = {norm_city(value) for value in args.city_token if str(value).strip()}
    excluded = {norm_city(value) for value in args.exclude_city_token if str(value).strip()}
    cities = sorted(selected or discover_cities(Path(args.link_root), str(args.link_pattern)))
    cities = [city for city in cities if city not in excluded]
    if not cities:
        raise SystemExit("No city jobs selected.")

    start = parse_iso_date(str(args.start_date), "--start-date")
    end = parse_iso_date(str(args.end_date), "--end-date")
    dates = date_range(start, end)
    if int(args.max_dates) > 0:
        dates = dates[: int(args.max_dates)]

    archives = {variable: raster_archives(Path(args.raw_root), variable) for variable in variables}
    missing_archives = [
        (variable, current)
        for variable in variables
        for current in dates
        if current not in archives[variable]
    ]
    if missing_archives:
        preview = ", ".join(f"{variable}:{current}" for variable, current in missing_archives[:10])
        raise FileNotFoundError(
            f"Missing {len(missing_archives):,} requested PRISM archive(s); first: {preview}"
        )

    first_archive, first_member = archives[variables[0]][dates[0]]
    with rasterio.open(zip_raster_path(first_archive, first_member)) as reference_dataset:
        reference = raster_signature(reference_dataset)
        transform = reference_dataset.transform
        raster_height = int(reference_dataset.height)
        raster_width = int(reference_dataset.width)

    city_frames: dict[str, pd.DataFrame] = {}
    required_pixels: set[tuple[int, int]] = set()
    for city in cities:
        path = link_path(Path(args.link_root), city, str(args.link_pattern))
        frame = load_city_links(path)
        valid_coord = frame["sentinel_lat"].notna() & frame["sentinel_lon"].notna()
        prism_rows = np.full(len(frame), -1, dtype=np.int32)
        prism_cols = np.full(len(frame), -1, dtype=np.int32)
        if valid_coord.any():
            valid_positions = np.flatnonzero(valid_coord.to_numpy())
            rows, cols = north_up_grid_rowcol(
                transform,
                frame.loc[valid_coord, "sentinel_lon"].to_numpy(dtype=np.float64),
                frame.loc[valid_coord, "sentinel_lat"].to_numpy(dtype=np.float64),
            )
            inside = (rows >= 0) & (rows < raster_height) & (cols >= 0) & (cols < raster_width)
            prism_rows[valid_positions[inside]] = rows[inside]
            prism_cols[valid_positions[inside]] = cols[inside]
            required_pixels.update((int(row), int(col)) for row, col in zip(rows[inside], cols[inside]))
        frame["prism_grid_row"] = prism_rows
        frame["prism_grid_col"] = prism_cols
        city_frames[city] = frame
        unique_city_pixels = {
            (int(row), int(col))
            for row, col in zip(prism_rows, prism_cols)
            if row >= 0 and col >= 0
        }
        print(
            f"{city}: trees={len(frame):,}; valid coordinates={valid_coord.sum():,}; "
            f"unique PRISM pixels={len(unique_city_pixels):,}",
            flush=True,
        )

    pixels = sorted(required_pixels)
    if not pixels:
        raise SystemExit("No selected tree coordinates fall inside the PRISM raster.")
    pixel_rows = np.asarray([value[0] for value in pixels], dtype=np.int32)
    pixel_cols = np.asarray([value[1] for value in pixels], dtype=np.int32)
    pixel_lookup = {value: index for index, value in enumerate(pixels)}

    output_dir = Path(args.output_dir)
    global_path = output_dir / "prism_daily_values.npz"
    if global_path.exists() and not bool(args.force):
        raise SystemExit(f"Output exists: {global_path}; pass --force to replace it.")

    values = np.zeros((len(pixels), len(dates), len(variables)), dtype=np.float32)
    valid_mask = np.zeros_like(values, dtype=bool)
    for variable_index, variable in enumerate(variables):
        for date_index, current in enumerate(dates):
            archive, member = archives[variable][current]
            with rasterio.open(zip_raster_path(archive, member)) as dataset:
                if not same_grid(dataset, reference):
                    raise RuntimeError(f"PRISM grid changed in {archive}")
                band = dataset.read(1)
                sampled = np.asarray(band[pixel_rows, pixel_cols], dtype=np.float32)
                valid = np.isfinite(sampled)
                if dataset.nodata is not None:
                    valid &= ~np.isclose(sampled, float(dataset.nodata))
                values[:, date_index, variable_index] = np.where(valid, sampled, 0.0)
                valid_mask[:, date_index, variable_index] = valid
            if date_index == 0 or (date_index + 1) % 100 == 0 or date_index + 1 == len(dates):
                print(
                    f"{variable}: extracted {date_index + 1:,}/{len(dates):,} dates",
                    flush=True,
                )

    output_dir.mkdir(parents=True, exist_ok=True)
    date_codes = np.asarray([int(current.strftime("%Y%m%d")) for current in dates], dtype=np.int32)
    config = {
        "raw_root": str(Path(args.raw_root)),
        "link_root": str(Path(args.link_root)),
        "start_date": str(dates[0]),
        "end_date": str(dates[-1]),
        "variables": list(variables),
        "cities": cities,
        "storage_contract": (
            "values[prism_pixel, date, variable]; city lookup files map tree_id and "
            "Sentinel cell to prism_daily_source_row"
        ),
        "reference_raster": reference,
    }
    np.savez_compressed(
        global_path,
        values=values,
        valid_mask=valid_mask,
        dates_yyyymmdd=date_codes,
        variable_names=np.asarray(variables),
        prism_grid_row=pixel_rows,
        prism_grid_col=pixel_cols,
        config_json=json.dumps(config, sort_keys=True),
    )

    summary_rows: list[dict[str, Any]] = []
    for city, frame in city_frames.items():
        source_row = np.asarray(
            [
                pixel_lookup.get((int(row), int(col)), -1)
                if row >= 0 and col >= 0
                else -1
                for row, col in zip(frame["prism_grid_row"], frame["prism_grid_col"])
            ],
            dtype=np.int32,
        )
        missing = source_row < 0
        city_dir = output_dir / city
        city_dir.mkdir(parents=True, exist_ok=True)
        city_path = city_dir / f"{city}_prism_daily_index.npz"
        if city_path.exists() and not bool(args.force):
            raise SystemExit(f"Output exists: {city_path}; pass --force to replace it.")
        np.savez_compressed(
            city_path,
            tree_id=frame["tree_id"].to_numpy(dtype=np.int64),
            sentinel_cell_id=frame["sentinel_cell_id"].fillna("").astype(str).to_numpy(),
            sentinel_cell_col=pd.to_numeric(frame["sentinel_cell_col"], errors="coerce")
            .fillna(-1)
            .to_numpy(dtype=np.int64),
            sentinel_cell_row=pd.to_numeric(frame["sentinel_cell_row"], errors="coerce")
            .fillna(-1)
            .to_numpy(dtype=np.int64),
            sentinel_lat=frame["sentinel_lat"].to_numpy(dtype=np.float64),
            sentinel_lon=frame["sentinel_lon"].to_numpy(dtype=np.float64),
            prism_grid_row=frame["prism_grid_row"].to_numpy(dtype=np.int32),
            prism_grid_col=frame["prism_grid_col"].to_numpy(dtype=np.int32),
            prism_daily_source_row=source_row,
            missing_prism_daily=missing,
            global_values_path=np.asarray(str(global_path)),
        )
        summary_rows.append(
            {
                "city_token": city,
                "trees": int(len(frame)),
                "valid_tree_mappings": int((~missing).sum()),
                "missing_tree_mappings": int(missing.sum()),
                "unique_prism_pixels": int(len(np.unique(source_row[source_row >= 0]))),
                "output": str(city_path),
            }
        )

    summary_path = output_dir / "prism_daily_sidecar_summary.csv"
    pd.DataFrame(summary_rows).sort_values("city_token").to_csv(summary_path, index=False)
    print(
        f"Wrote {global_path}; pixels={len(pixels):,}; dates={len(dates):,}; "
        f"variables={len(variables):,}; valid={valid_mask.mean() * 100:.2f}%",
        flush=True,
    )
    print(f"Wrote city lookup summary: {summary_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
