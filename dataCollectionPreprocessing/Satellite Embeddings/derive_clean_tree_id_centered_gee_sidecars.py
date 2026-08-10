#!/usr/bin/env python3
"""Derive clean tree-id-centered GEE/Satlas embedding sidecars.

The clean tree-centered dataset is anchored on matched crown coordinates, not
the original inventory coordinate. This script computes the crown-centered
10 m cell for each clean crop row, then fills satellite embeddings from:

1. Supplemental/tree-centered embedding CSVs keyed to crown_cell_id.
2. Original embedding CSVs only when their latitude/longitude matches the
   crown-centered cell coordinate after rounding.

The output row order follows the clean NAIP crop metadata exactly.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
import re
import time
from typing import Any

import numpy as np
import pandas as pd
from pyproj import Transformer


DEFAULT_CROP_ROOT = Path(r"H:\TreeCenteredModelInputs\tree_centered_naip_crops_clean")
DEFAULT_ORIGINAL_EMBEDDING_ROOT = Path(r"E:\TreeId\SatelliteEmbedding")
DEFAULT_SUPPLEMENTAL_EMBEDDING_ROOT = Path(r"E:\TreeCenterSatelliteEmbedding")
DEFAULT_OUTPUT_DIR = Path(r"H:\TreeCenteredModelInputs\tree_centered_gee_inputs_clean")

EMBEDDING_COLUMNS = [f"A{i:02d}" for i in range(64)]
EMBEDDING_YEARS = [2021, 2022, 2023]
QUALITY_COLUMNS = [
    "embedding_year_count",
    "embedding_has_2021",
    "embedding_has_2022",
    "embedding_has_2023",
    "embedding_mean_pairwise_dot",
    "embedding_min_pairwise_dot",
]

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
    parser.add_argument("--crop-root", type=Path, default=DEFAULT_CROP_ROOT)
    parser.add_argument("--original-embedding-root", type=Path, default=DEFAULT_ORIGINAL_EMBEDDING_ROOT)
    parser.add_argument("--supplemental-embedding-root", type=Path, default=DEFAULT_SUPPLEMENTAL_EMBEDDING_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--city-token", action="append", default=[])
    parser.add_argument("--exclude-city-token", action="append", default=[])
    parser.add_argument("--metadata-pattern", default="*_tree_id_centered_nearest_64px_metadata.csv")
    parser.add_argument("--original-pattern", default="mccoy_satellite_embedding_*_{year}_batch_*.csv")
    parser.add_argument("--supplemental-pattern", default="tree_centered_satellite_embedding_{city}_{year}_batch_*.csv")
    parser.add_argument("--embedding-years", default="2021,2022,2023")
    parser.add_argument("--coordinate-decimals", type=int, default=7)
    parser.add_argument("--parallel-workers", type=int, default=1)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def norm_city(value: str) -> str:
    token = re.sub(r"[^a-z0-9]+", "", str(value).lower())
    return CITY_ALIASES.get(token, token)


def display_city_variants(city: str) -> set[str]:
    compact = norm_city(city)
    variants = {compact, compact.lower(), compact.title()}
    special = {
        "albuquerque": "Albuquerque",
        "anaheim": "Anaheim",
        "arlington": "Arlington",
        "atlanta": "Atlanta",
        "auroraco": "AuroraCO",
        "austin": "Austin",
        "baltimore": "Baltimore",
        "buffalo": "Buffalo",
        "capecoral": "CapeCoral",
        "coloradosprings": "ColoradoSprings",
        "columbus": "Columbus",
        "denver": "Denver",
        "desmoines": "DesMoines",
        "detroit": "Detroit",
        "gardengrove": "GardenGrove",
        "huntingtonbeach": "HuntingtonBeach",
        "knoxville": "Knoxville",
        "lasvegas": "LasVegas",
        "losangeles": "LosAngeles",
        "louisville": "Louisville",
        "madison": "Madison",
        "milwaukee": "Milwaukee",
        "minneapolis": "Minneapolis",
        "newyork": "NewYork",
        "oklahomacity": "OklahomaCity",
        "overlandpark": "OverlandPark",
        "pittsburgh": "Pittsburgh",
        "portland": "Portland",
        "providence": "Providence",
        "ranchocucamonga": "RanchoCucamonga",
        "rochester": "Rochester",
        "sacramento": "Sacramento",
        "sandiego": "SanDiego",
        "sanfrancisco": "SanFrancisco",
        "sanjose": "SanJose",
        "santarosa": "SantaRosa",
        "seattle": "Seattle",
        "siouxfalls": "SiouxFalls",
        "stlouis": "StLouis",
        "washingtondc": "WashingtonDC",
    }
    if compact in special:
        variants.add(special[compact])
    return variants


def parse_years(value: str) -> list[int]:
    years: list[int] = []
    for token in str(value).split(","):
        token = token.strip()
        if token:
            years.append(int(token))
    return years or EMBEDDING_YEARS


def discover_cities(crop_root: Path, pattern: str) -> list[str]:
    if not crop_root.exists():
        raise FileNotFoundError(crop_root)
    cities = []
    for city_dir in crop_root.iterdir():
        if city_dir.is_dir() and sorted(city_dir.glob(pattern)):
            cities.append(city_dir.name)
    return sorted(cities)


def find_crop_metadata(args: argparse.Namespace, city: str) -> Path:
    matches = sorted((Path(args.crop_root) / city).glob(args.metadata_pattern))
    if not matches:
        raise FileNotFoundError(f"No clean crop metadata matching {args.metadata_pattern!r} for {city}")
    return matches[0]


def coord_key(lat: pd.Series | np.ndarray, lon: pd.Series | np.ndarray, decimals: int) -> np.ndarray:
    scale = 10**decimals
    lat_i = np.rint(np.asarray(lat, dtype=np.float64) * scale).astype(np.int64)
    lon_i = np.rint(np.asarray(lon, dtype=np.float64) * scale).astype(np.int64)
    return lat_i.astype(str) + "_" + lon_i.astype(str)


def crown_cell_table(metadata: pd.DataFrame, decimals: int) -> pd.DataFrame:
    required = {"tree_id", "crop_index", "crown_x_utm", "crown_y_utm", "crown_epsg"}
    missing = sorted(required.difference(metadata.columns))
    if missing:
        raise RuntimeError(f"Clean crop metadata is missing required column(s): {missing}")
    pieces: list[pd.DataFrame] = []
    for epsg_value, group in metadata.groupby("crown_epsg", sort=False):
        epsg_int = int(epsg_value)
        x = pd.to_numeric(group["crown_x_utm"], errors="coerce").to_numpy(dtype=np.float64)
        y = pd.to_numeric(group["crown_y_utm"], errors="coerce").to_numpy(dtype=np.float64)
        col = np.floor(x / 10.0).astype(np.int64)
        row = np.floor(y / 10.0).astype(np.int64)
        center_x = col * 10.0 + 5.0
        center_y = row * 10.0 + 5.0
        transformer = Transformer.from_crs(f"EPSG:{epsg_int}", "EPSG:4326", always_xy=True)
        lon, lat = transformer.transform(center_x, center_y)
        part = pd.DataFrame(
            {
                "tree_id": group["tree_id"].to_numpy(),
                "crop_index": group["crop_index"].to_numpy(),
                "crown_cell_id": [f"epsg{epsg_int}_c{c}_r{r}" for c, r in zip(col, row)],
                "crown_cell_epsg": epsg_int,
                "crown_cell_col": col,
                "crown_cell_row": row,
                "crown_cell_center_x": center_x,
                "crown_cell_center_y": center_y,
                "crown_cell_lat": np.asarray(lat, dtype=np.float64),
                "crown_cell_lon": np.asarray(lon, dtype=np.float64),
            },
            index=group.index,
        )
        pieces.append(part)
    out = pd.concat(pieces).sort_index()
    out["crown_coord_key"] = coord_key(out["crown_cell_lat"], out["crown_cell_lon"], decimals)
    return out


def embedding_quality_array(values: np.ndarray, mask: np.ndarray, years: list[int]) -> np.ndarray:
    present = mask.astype(bool)
    quality = np.zeros((values.shape[0], len(QUALITY_COLUMNS)), dtype=np.float32)
    quality[:, 0] = present.sum(axis=1).astype(np.float32)
    for year in EMBEDDING_YEARS:
        if year in years:
            quality[:, QUALITY_COLUMNS.index(f"embedding_has_{year}")] = present[:, years.index(year)].astype(np.float32)
    quality[:, QUALITY_COLUMNS.index("embedding_mean_pairwise_dot")] = 1.0
    quality[:, QUALITY_COLUMNS.index("embedding_min_pairwise_dot")] = 1.0
    if len(years) >= 2:
        normed = values / np.maximum(np.linalg.norm(values, axis=2, keepdims=True), 1e-6)
        pair_dots: list[np.ndarray] = []
        for i in range(len(years)):
            for j in range(i + 1, len(years)):
                valid_pair = present[:, i] & present[:, j]
                dot = np.sum(normed[:, i, :] * normed[:, j, :], axis=1).astype(np.float32)
                dot[~valid_pair] = np.nan
                pair_dots.append(dot)
        if pair_dots:
            dots = np.vstack(pair_dots).T
            any_pair = np.isfinite(dots).any(axis=1)
            mean_dot = np.ones(values.shape[0], dtype=np.float32)
            min_dot = np.ones(values.shape[0], dtype=np.float32)
            if any_pair.any():
                with np.errstate(invalid="ignore"):
                    mean_dot[any_pair] = np.nanmean(dots[any_pair], axis=1)
                    min_dot[any_pair] = np.nanmin(dots[any_pair], axis=1)
            quality[any_pair, QUALITY_COLUMNS.index("embedding_mean_pairwise_dot")] = mean_dot[any_pair]
            quality[any_pair, QUALITY_COLUMNS.index("embedding_min_pairwise_dot")] = min_dot[any_pair]
    return np.nan_to_num(quality, nan=1.0, posinf=1.0, neginf=0.0)


def eligible_supplemental_files(args: argparse.Namespace, city: str, years: list[int]) -> list[tuple[Path, int]]:
    files: list[tuple[Path, int]] = []
    root = Path(args.supplemental_embedding_root)
    for year in years:
        pattern = args.supplemental_pattern.format(city=city, year=year)
        for path in sorted(root.glob(pattern)):
            files.append((path, year))
    return files


def eligible_original_files(args: argparse.Namespace, city: str, years: list[int]) -> list[tuple[Path, int]]:
    files: list[tuple[Path, int]] = []
    root = Path(args.original_embedding_root)
    variants = display_city_variants(city)
    for year in years:
        for path in sorted(root.glob(args.original_pattern.format(year=year))):
            name_norm = norm_city(path.name)
            if any(norm_city(v) in name_norm for v in variants):
                files.append((path, year))
    return files


def load_embedding_lookup(
    city: str,
    label: str,
    files: list[tuple[Path, int]],
    needed_cells: pd.DataFrame,
    years: list[int],
    decimals: int,
) -> tuple[dict[str, dict[int, np.ndarray]], dict[str, int]]:
    lookup: dict[str, dict[int, np.ndarray]] = {}
    stats = {"files": len(files), "matched_rows": 0, "matched_cells": 0}
    if not files:
        return lookup, stats
    needed_ids = set(needed_cells["crown_cell_id"].astype(str))
    needed_epsgs = sorted(set(pd.to_numeric(needed_cells["crown_cell_epsg"], errors="coerce").dropna().astype(int)))
    started = time.perf_counter()
    for index, (path, year) in enumerate(files, start=1):
        header = set(pd.read_csv(path, nrows=0).columns)
        usecols = [column for column in ["crown_cell_id", "latitude", "longitude", "embedding_year", *EMBEDDING_COLUMNS] if column in header]
        if not {"latitude", "longitude", "embedding_year"}.issubset(usecols):
            print(f"{city}: WARNING {label} file missing coordinate/year columns: {path}", flush=True)
            continue
        frame = pd.read_csv(path, usecols=usecols)
        frame["embedding_year"] = pd.to_numeric(frame["embedding_year"], errors="coerce").astype("Int64")
        frame = frame.loc[frame["embedding_year"].eq(year)].copy()
        if frame.empty:
            continue
        if "crown_cell_id" in frame.columns:
            frame["match_key"] = frame["crown_cell_id"].astype(str)
            frame = frame.loc[frame["match_key"].isin(needed_ids)].copy()
        else:
            match_keys = np.full(len(frame), "", dtype=object)
            lon = pd.to_numeric(frame["longitude"], errors="coerce").to_numpy(dtype=np.float64)
            lat = pd.to_numeric(frame["latitude"], errors="coerce").to_numpy(dtype=np.float64)
            for epsg in needed_epsgs:
                transformer = Transformer.from_crs("EPSG:4326", f"EPSG:{int(epsg)}", always_xy=True)
                x, y = transformer.transform(lon, lat)
                col = np.floor(np.asarray(x, dtype=np.float64) / 10.0).astype(np.int64)
                row = np.floor(np.asarray(y, dtype=np.float64) / 10.0).astype(np.int64)
                candidate = np.asarray([f"epsg{int(epsg)}_c{c}_r{r}" for c, r in zip(col, row)], dtype=object)
                candidate_mask = np.isin(candidate, list(needed_ids))
                match_keys[candidate_mask] = candidate[candidate_mask]
            frame["match_key"] = match_keys
            frame = frame.loc[frame["match_key"].isin(needed_ids)].copy()
        if frame.empty:
            if index == 1 or index == len(files) or index % 25 == 0:
                print(
                    f"{city}: {label} embedding file {index:,}/{len(files):,}; matched_rows={stats['matched_rows']:,}; "
                    f"elapsed={time.perf_counter() - started:.1f}s",
                    flush=True,
                )
            continue
        for column in EMBEDDING_COLUMNS:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        grouped = frame.groupby(["match_key", "embedding_year"], as_index=False)[EMBEDDING_COLUMNS].mean()
        for record in grouped.itertuples(index=False):
            key = str(record.match_key)
            vector = np.asarray([getattr(record, column) for column in EMBEDDING_COLUMNS], dtype=np.float32)
            vector = np.nan_to_num(vector, nan=0.0, posinf=0.0, neginf=0.0)
            lookup.setdefault(key, {})[int(record.embedding_year)] = vector
        stats["matched_rows"] += int(len(frame))
        if index == 1 or index == len(files) or index % 25 == 0:
            print(
                f"{city}: {label} embedding file {index:,}/{len(files):,}; matched_rows={stats['matched_rows']:,}; "
                f"elapsed={time.perf_counter() - started:.1f}s",
                flush=True,
            )
    stats["matched_cells"] = len(lookup)
    return lookup, stats


def process_city(city: str, args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    out_dir = Path(args.output_dir) / city
    out_path = out_dir / f"{city}_tree_id_centered_gee_inputs.npz"
    if out_path.exists() and not args.force:
        with np.load(out_path, allow_pickle=False) as data:
            rows = int(data["tree_id"].shape[0])
            missing = int(np.asarray(data["missing_satellite_embedding"], dtype=bool).sum())
        return {"city_token": city, "status": "exists", "rows": rows, "missing_satellite_embedding": missing}

    years = parse_years(args.embedding_years)
    metadata_path = find_crop_metadata(args, city)
    metadata = pd.read_csv(metadata_path, low_memory=False)
    cells = crown_cell_table(metadata, int(args.coordinate_decimals))
    unique_cells = cells.drop_duplicates("crown_cell_id", keep="first")
    print(
        f"{city}: rows={len(metadata):,}; crown-centered cells={len(unique_cells):,}",
        flush=True,
    )

    supplemental_lookup, supplemental_stats = load_embedding_lookup(
        city,
        "supplemental",
        eligible_supplemental_files(args, city, years),
        unique_cells,
        years,
        int(args.coordinate_decimals),
    )
    original_lookup, original_stats = load_embedding_lookup(
        city,
        "original",
        eligible_original_files(args, city, years),
        unique_cells,
        years,
        int(args.coordinate_decimals),
    )

    year_to_pos = {year: pos for pos, year in enumerate(years)}
    embedding = np.zeros((len(metadata), len(years), len(EMBEDDING_COLUMNS)), dtype=np.float32)
    mask = np.zeros((len(metadata), len(years)), dtype=np.float32)
    source_code = np.zeros(len(metadata), dtype=np.int8)
    used_original = np.zeros(len(metadata), dtype=bool)
    used_supplemental = np.zeros(len(metadata), dtype=bool)

    print(
        f"{city}: assigning embeddings to {len(metadata):,} crop row(s); "
        f"original_cells={len(original_lookup):,}; supplemental_cells={len(supplemental_lookup):,}",
        flush=True,
    )
    crown_ids = cells["crown_cell_id"].astype(str).to_numpy()
    for row, cell_id in enumerate(crown_ids):
        # Prefer exact crown-cell supplemental exports. Original exports are
        # only used when their coordinate matches the crown-centered cell.
        vectors_by_year = supplemental_lookup.get(cell_id)
        if vectors_by_year:
            source_code[row] = 2
            used_supplemental[row] = True
        else:
            vectors_by_year = original_lookup.get(cell_id)
            if vectors_by_year:
                source_code[row] = 1
                used_original[row] = True
        if not vectors_by_year:
            continue
        for year, vector in vectors_by_year.items():
            if year not in year_to_pos:
                continue
            pos = year_to_pos[year]
            embedding[row, pos] = vector
            mask[row, pos] = 1.0

    print(f"{city}: computing embedding quality metrics", flush=True)
    quality = embedding_quality_array(embedding, mask, years)
    missing = mask.sum(axis=1) == 0

    out_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "city_token": city,
        "crop_metadata": str(metadata_path),
        "original_embedding_root": str(args.original_embedding_root),
        "supplemental_embedding_root": str(args.supplemental_embedding_root),
        "embedding_years": years,
        "coordinate_decimals": int(args.coordinate_decimals),
        "join_policy": "crown-centered 10m cell; supplemental by crown_cell_id, original fallback by rounded crown-cell center lat/lon",
    }
    np.savez_compressed(
        out_path,
        tree_id=metadata["tree_id"].to_numpy(dtype=np.int64),
        crop_index=metadata["crop_index"].to_numpy(dtype=np.int64),
        row_index=metadata["row_index"].to_numpy(dtype=np.int64) if "row_index" in metadata.columns else np.arange(len(metadata), dtype=np.int64),
        crown_cell_id=cells["crown_cell_id"].astype(str).to_numpy(),
        crown_cell_epsg=cells["crown_cell_epsg"].to_numpy(dtype=np.int32),
        crown_cell_col=cells["crown_cell_col"].to_numpy(dtype=np.int64),
        crown_cell_row=cells["crown_cell_row"].to_numpy(dtype=np.int64),
        crown_cell_lat=cells["crown_cell_lat"].to_numpy(dtype=np.float64),
        crown_cell_lon=cells["crown_cell_lon"].to_numpy(dtype=np.float64),
        satellite_embedding=embedding,
        satellite_embedding_mask=mask,
        satellite_embedding_quality=quality,
        satellite_embedding_columns=np.asarray(EMBEDDING_COLUMNS),
        satellite_embedding_years=np.asarray(years, dtype=np.int32),
        satellite_embedding_quality_columns=np.asarray(QUALITY_COLUMNS),
        embedding_source_code=source_code,
        missing_satellite_embedding=missing,
        used_original_satellite_embedding=used_original,
        used_additional_satellite_embedding=used_supplemental,
        config_json=np.asarray(json.dumps(config, indent=2)),
    )
    missing_count = int(missing.sum())
    elapsed = time.perf_counter() - started
    print(
        f"{city}: wrote {out_path}; rows={len(metadata):,}; "
        f"missing={missing_count:,} ({missing_count / max(len(metadata), 1):.2%}); "
        f"original_cells={original_stats['matched_cells']:,}; supplemental_cells={supplemental_stats['matched_cells']:,}; "
        f"elapsed={elapsed:.1f}s",
        flush=True,
    )
    return {
        "city_token": city,
        "status": "completed",
        "rows": int(len(metadata)),
        "unique_crown_cells": int(len(unique_cells)),
        "missing_satellite_embedding": missing_count,
        "missing_satellite_embedding_pct": float(missing_count / max(len(metadata), 1) * 100.0),
        "used_original_rows": int(used_original.sum()),
        "used_supplemental_rows": int(used_supplemental.sum()),
        "original_matched_cells": int(original_stats["matched_cells"]),
        "supplemental_matched_cells": int(supplemental_stats["matched_cells"]),
        "elapsed_sec": float(elapsed),
        "output": str(out_path),
    }


def main() -> int:
    args = parse_args()
    selected = [norm_city(city) for city in args.city_token] if args.city_token else discover_cities(Path(args.crop_root), args.metadata_pattern)
    excluded = {norm_city(city) for city in args.exclude_city_token}
    cities = [city for city in selected if city not in excluded]
    if not cities:
        raise SystemExit("No city jobs selected.")
    print(f"Deriving clean crown-centered GEE sidecars for {len(cities):,} city/cities.", flush=True)

    results: list[dict[str, Any]] = []
    workers = max(1, int(args.parallel_workers))
    if workers == 1:
        for city in cities:
            try:
                results.append(process_city(city, args))
            except Exception as error:  # noqa: BLE001
                print(f"{city}: FAILED {error}", flush=True)
                results.append({"city_token": city, "status": "failed", "error": str(error)})
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            future_map = {pool.submit(process_city, city, args): city for city in cities}
            for future in as_completed(future_map):
                city = future_map[future]
                try:
                    results.append(future.result())
                except Exception as error:  # noqa: BLE001
                    print(f"{city}: FAILED {error}", flush=True)
                    results.append({"city_token": city, "status": "failed", "error": str(error)})

    summary = pd.DataFrame(results).sort_values("city_token")
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    summary_path = Path(args.output_dir) / "clean_tree_id_centered_gee_sidecar_summary.csv"
    summary.to_csv(summary_path, index=False)
    print("\nSummary:", flush=True)
    print(summary["status"].value_counts(dropna=False).to_string(), flush=True)
    print(f"Wrote summary: {summary_path}", flush=True)
    return 1 if (summary["status"] == "failed").any() else 0


if __name__ == "__main__":
    raise SystemExit(main())
