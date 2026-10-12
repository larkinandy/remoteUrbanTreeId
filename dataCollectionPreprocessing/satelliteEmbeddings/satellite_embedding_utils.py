"""Canonical constants and pure transformations for satellite embeddings.

This module is shared by export, sidecar construction, and validation so grid
geometry, city tokens, source codes, and quality definitions cannot drift.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
from pyproj import Transformer


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

# Stable values retained for compatibility with the frozen study sidecars.
SOURCE_MISSING = 0
SOURCE_ORIGINAL_MATCH = 1
SOURCE_DIRECT_CROWN_CELL = 2

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


def normalize_city(value: object) -> str:
    """Return the canonical compact token used for city directories."""
    token = re.sub(r"[^a-z0-9]+", "", str(value or "").lower())
    return CITY_ALIASES.get(token, token)


def crown_cell_table(metadata: pd.DataFrame) -> pd.DataFrame:
    """Calculate the crown-centered 10 m grid cell for every metadata row."""
    required = {"tree_id", "crop_index", "crown_x_utm", "crown_y_utm", "crown_epsg"}
    missing = sorted(required.difference(metadata.columns))
    if missing:
        raise ValueError(f"Crop metadata missing required columns: {missing}")

    pieces: list[pd.DataFrame] = []
    for epsg_value, group in metadata.groupby("crown_epsg", sort=False):
        epsg = int(epsg_value)
        x = pd.to_numeric(group["crown_x_utm"], errors="coerce").to_numpy(dtype=np.float64)
        y = pd.to_numeric(group["crown_y_utm"], errors="coerce").to_numpy(dtype=np.float64)
        if not (np.isfinite(x) & np.isfinite(y)).all():
            raise ValueError(f"Non-finite crown coordinates found for EPSG:{epsg}")
        col = np.floor(x / 10.0).astype(np.int64)
        row = np.floor(y / 10.0).astype(np.int64)
        center_x = col * 10.0 + 5.0
        center_y = row * 10.0 + 5.0
        lon, lat = Transformer.from_crs(f"EPSG:{epsg}", "EPSG:4326", always_xy=True).transform(center_x, center_y)
        pieces.append(
            pd.DataFrame(
                {
                    "tree_id": group["tree_id"].to_numpy(),
                    "crop_index": group["crop_index"].to_numpy(),
                    "row_index": group["row_index"].to_numpy() if "row_index" in group else group.index.to_numpy(),
                    "crown_cell_id": [f"epsg{epsg}_c{c}_r{r}" for c, r in zip(col, row)],
                    "crown_cell_epsg": epsg,
                    "crown_cell_col": col,
                    "crown_cell_row": row,
                    "crown_cell_center_x": center_x,
                    "crown_cell_center_y": center_y,
                    "crown_cell_lon": np.asarray(lon, dtype=np.float64),
                    "crown_cell_lat": np.asarray(lat, dtype=np.float64),
                },
                index=group.index,
            )
        )
    return pd.concat(pieces).sort_index()


def unique_crown_cell_points(metadata: pd.DataFrame) -> pd.DataFrame:
    """Return one Earth Engine point row per unique crown-centered cell."""
    cells = crown_cell_table(metadata)
    points = cells.rename(columns={"crown_cell_lon": "longitude", "crown_cell_lat": "latitude"}).copy()
    counts = points.groupby("crown_cell_id").size().rename("tree_centered_record_count")
    return points.drop_duplicates("crown_cell_id").merge(counts, on="crown_cell_id", how="left")


def embedding_quality_array(values: np.ndarray, mask: np.ndarray, years: list[int]) -> np.ndarray:
    """Calculate annual availability and pairwise embedding-similarity metrics."""
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
        for first in range(len(years)):
            for second in range(first + 1, len(years)):
                valid_pair = present[:, first] & present[:, second]
                dot = np.sum(normed[:, first, :] * normed[:, second, :], axis=1).astype(np.float32)
                dot[~valid_pair] = np.nan
                pair_dots.append(dot)
        if pair_dots:
            dots = np.vstack(pair_dots).T
            any_pair = np.isfinite(dots).any(axis=1)
            with np.errstate(invalid="ignore"):
                quality[any_pair, QUALITY_COLUMNS.index("embedding_mean_pairwise_dot")] = np.nanmean(dots[any_pair], axis=1)
                quality[any_pair, QUALITY_COLUMNS.index("embedding_min_pairwise_dot")] = np.nanmin(dots[any_pair], axis=1)
    return np.nan_to_num(quality, nan=1.0, posinf=1.0, neginf=0.0)
