"""Read annual export CSVs and match them to canonical crown-cell identifiers.

File discovery and table matching live here so sidecar orchestration remains
independent of Earth Engine export-table layout details.
"""

from __future__ import annotations

from pathlib import Path
import time

import numpy as np
import pandas as pd
from pyproj import Transformer

from satellite_embedding_utils import EMBEDDING_COLUMNS


def discover_embedding_files(root: Path, pattern: str, city: str, years: list[int]) -> list[tuple[Path, int]]:
    """Find downloaded annual batch exports for one city."""
    files: list[tuple[Path, int]] = []
    for year in years:
        files.extend((path, year) for path in sorted(root.glob(pattern.format(city=city, year=year))))
    return files


def load_embedding_lookup(
    city: str,
    files: list[tuple[Path, int]],
    needed_cells: pd.DataFrame,
) -> tuple[dict[str, dict[int, np.ndarray]], dict[str, int]]:
    """Load annual exports into a crown-cell/year lookup."""
    lookup: dict[str, dict[int, np.ndarray]] = {}
    stats = {"files": len(files), "matched_rows": 0, "matched_cells": 0}
    if not files:
        return lookup, stats
    needed_ids = set(needed_cells["crown_cell_id"].astype(str))
    needed_epsgs = sorted(set(pd.to_numeric(needed_cells["crown_cell_epsg"], errors="coerce").dropna().astype(int)))
    started = time.perf_counter()
    for index, (path, expected_year) in enumerate(files, start=1):
        header = set(pd.read_csv(path, nrows=0).columns)
        usecols = [column for column in ["crown_cell_id", "latitude", "longitude", "embedding_year", *EMBEDDING_COLUMNS] if column in header]
        if not {"latitude", "longitude", "embedding_year"}.issubset(usecols):
            print(f"{city}: WARNING export missing coordinate/year columns: {path}", flush=True)
            continue
        frame = pd.read_csv(path, usecols=usecols)
        frame["embedding_year"] = pd.to_numeric(frame["embedding_year"], errors="coerce").astype("Int64")
        frame = frame.loc[frame["embedding_year"].eq(expected_year)].copy()
        if "crown_cell_id" in frame.columns:
            frame["match_key"] = frame["crown_cell_id"].astype(str)
        else:
            frame["match_key"] = coordinate_cell_keys(frame, needed_ids, needed_epsgs)
        frame = frame.loc[frame["match_key"].isin(needed_ids)].copy()
        if not frame.empty:
            for column in EMBEDDING_COLUMNS:
                frame[column] = pd.to_numeric(frame[column], errors="coerce")
            grouped = frame.groupby(["match_key", "embedding_year"], as_index=False)[EMBEDDING_COLUMNS].mean()
            for record in grouped.itertuples(index=False):
                vector = np.asarray([getattr(record, column) for column in EMBEDDING_COLUMNS], dtype=np.float32)
                lookup.setdefault(str(record.match_key), {})[int(record.embedding_year)] = np.nan_to_num(vector)
            stats["matched_rows"] += int(len(frame))
        if index == 1 or index == len(files) or index % 25 == 0:
            print(
                f"{city}: embedding file {index:,}/{len(files):,}; matched_rows={stats['matched_rows']:,}; "
                f"elapsed={time.perf_counter() - started:.1f}s",
                flush=True,
            )
    stats["matched_cells"] = len(lookup)
    return lookup, stats


def coordinate_cell_keys(frame: pd.DataFrame, needed_ids: set[str], needed_epsgs: list[int]) -> np.ndarray:
    """Recover cell identifiers for exports lacking an explicit crown_cell_id."""
    keys = np.full(len(frame), "", dtype=object)
    lon = pd.to_numeric(frame["longitude"], errors="coerce").to_numpy(dtype=np.float64)
    lat = pd.to_numeric(frame["latitude"], errors="coerce").to_numpy(dtype=np.float64)
    for epsg in needed_epsgs:
        x, y = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True).transform(lon, lat)
        col = np.floor(np.asarray(x) / 10.0).astype(np.int64)
        row = np.floor(np.asarray(y) / 10.0).astype(np.int64)
        candidates = np.asarray([f"epsg{epsg}_c{c}_r{r}" for c, r in zip(col, row)], dtype=object)
        matched = np.isin(candidates, list(needed_ids))
        keys[matched] = candidates[matched]
    return keys
