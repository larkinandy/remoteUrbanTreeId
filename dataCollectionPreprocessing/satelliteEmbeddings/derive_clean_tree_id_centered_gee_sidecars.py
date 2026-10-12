#!/usr/bin/env python3
"""Build model-ready Google Satellite Embedding sidecars by city.

The clean tree-centered dataset is anchored on matched crown coordinates, not
the original inventory coordinate. This script computes the crown-centered
10 m cell for each clean crop row, then joins the annual crown-cell exports by
``crown_cell_id``. The output row order follows the clean NAIP crop metadata
exactly.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
import time
from typing import Any

import numpy as np
import pandas as pd

from satellite_embedding_utils import (
    EMBEDDING_COLUMNS,
    EMBEDDING_YEARS,
    QUALITY_COLUMNS,
    SOURCE_DIRECT_CROWN_CELL,
    crown_cell_table,
    embedding_quality_array,
    normalize_city,
)
from satellite_embedding_io import discover_embedding_files, load_embedding_lookup


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--crop-root", type=Path, required=True)
    parser.add_argument("--embedding-root", type=Path, required=True, help="Downloaded crown-centered annual export CSVs.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--city-token", action="append", default=[])
    parser.add_argument("--exclude-city-token", action="append", default=[])
    parser.add_argument("--metadata-pattern", default="*_tree_id_centered_nearest_64px_metadata.csv")
    parser.add_argument("--embedding-pattern", default="tree_centered_satellite_embedding_{city}_{year}_batch_*.csv")
    parser.add_argument("--embedding-years", nargs="+", type=int, default=EMBEDDING_YEARS)
    parser.add_argument("--parallel-workers", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def norm_city(value: str) -> str:
    return normalize_city(value)


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


def assign_embeddings(
    crown_ids: np.ndarray,
    lookup: dict[str, dict[int, np.ndarray]],
    years: list[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Expand cell-level annual embeddings to metadata-row arrays."""
    embedding = np.zeros((len(crown_ids), len(years), len(EMBEDDING_COLUMNS)), dtype=np.float32)
    mask = np.zeros((len(crown_ids), len(years)), dtype=np.float32)
    source = np.zeros(len(crown_ids), dtype=np.int8)
    original = np.zeros(len(crown_ids), dtype=bool)
    direct = np.zeros(len(crown_ids), dtype=bool)
    year_to_pos = {year: pos for pos, year in enumerate(years)}
    for row, cell_id in enumerate(crown_ids):
        vectors = lookup.get(str(cell_id))
        if not vectors:
            continue
        source[row] = SOURCE_DIRECT_CROWN_CELL
        direct[row] = True
        for year, vector in vectors.items():
            if year in year_to_pos:
                embedding[row, year_to_pos[year]] = vector
                mask[row, year_to_pos[year]] = 1.0
    return embedding, mask, source, original, direct


def write_sidecar(
    path: Path,
    metadata: pd.DataFrame,
    cells: pd.DataFrame,
    years: list[int],
    embedding: np.ndarray,
    mask: np.ndarray,
    source: np.ndarray,
    original: np.ndarray,
    direct: np.ndarray,
    metadata_name: str,
) -> np.ndarray:
    """Write one portable city sidecar and return its missing-row mask."""
    quality = embedding_quality_array(embedding, mask, years)
    missing = mask.sum(axis=1) == 0
    config = {
        "city_token": path.parent.name,
        "crop_metadata_name": metadata_name,
        "embedding_source": "GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL",
        "embedding_years": years,
        "join_policy": "direct crown-centered 10 m cell join by crown_cell_id",
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        tree_id=metadata["tree_id"].to_numpy(dtype=np.int64),
        crop_index=metadata["crop_index"].to_numpy(dtype=np.int64),
        row_index=metadata["row_index"].to_numpy(dtype=np.int64) if "row_index" in metadata.columns else np.arange(len(metadata), dtype=np.int64),
        crown_cell_id=np.asarray(cells["crown_cell_id"].astype(str).tolist(), dtype=str),
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
        embedding_source_code=source,
        missing_satellite_embedding=missing,
        used_original_satellite_embedding=original,
        used_additional_satellite_embedding=direct,
        config_json=np.asarray(json.dumps(config, indent=2)),
    )
    return missing


def process_city(city: str, args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    out_dir = Path(args.output_dir) / city
    out_path = out_dir / f"{city}_tree_id_centered_gee_inputs.npz"
    if out_path.exists() and not args.overwrite:
        with np.load(out_path, allow_pickle=False) as data:
            rows = int(data["tree_id"].shape[0])
            missing = int(np.asarray(data["missing_satellite_embedding"], dtype=bool).sum())
        return {"city_token": city, "status": "exists", "rows": rows, "missing_satellite_embedding": missing}

    years = list(args.embedding_years)
    metadata_path = find_crop_metadata(args, city)
    metadata = pd.read_csv(metadata_path, low_memory=False)
    cells = crown_cell_table(metadata)
    unique_cells = cells.drop_duplicates("crown_cell_id", keep="first")
    print(
        f"{city}: rows={len(metadata):,}; crown-centered cells={len(unique_cells):,}",
        flush=True,
    )

    embedding_lookup, embedding_stats = load_embedding_lookup(
        city,
        discover_embedding_files(Path(args.embedding_root), args.embedding_pattern, city, years),
        unique_cells,
    )

    print(
        f"{city}: assigning embeddings to {len(metadata):,} crop row(s); "
        f"exported_cells={len(embedding_lookup):,}",
        flush=True,
    )
    embedding, mask, source_code, used_original, used_supplemental = assign_embeddings(
        cells["crown_cell_id"].astype(str).to_numpy(), embedding_lookup, years
    )

    print(f"{city}: computing embedding quality metrics", flush=True)
    missing = write_sidecar(
        out_path, metadata, cells, years, embedding, mask, source_code,
        used_original, used_supplemental, metadata_path.name
    )
    missing_count = int(missing.sum())
    elapsed = time.perf_counter() - started
    print(
        f"{city}: wrote {out_path}; rows={len(metadata):,}; "
        f"missing={missing_count:,} ({missing_count / max(len(metadata), 1):.2%}); "
        f"exported_cells={embedding_stats['matched_cells']:,}; "
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
        "exported_matched_cells": int(embedding_stats["matched_cells"]),
        "elapsed_sec": float(elapsed),
        "output_name": out_path.name,
    }


def main() -> int:
    args = parse_args()
    if not Path(args.embedding_root).is_dir():
        raise FileNotFoundError(f"Embedding export directory does not exist: {args.embedding_root}")
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
