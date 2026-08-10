"""Current Sentinel-2 cell feature utilities for multimodal model inputs.

This module owns the shared Sentinel-2 raw-band, spectral-index, and completed
export parsing contract used by the interpolated time-series pipeline. It does
not write the obsolete fixed-window products under ``E:/cell/sentinel2``.
"""

from __future__ import annotations

import math
import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd


S2_RAW_COLS = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]
S2_INDEX_COLS = [
    "NDVI",
    "GNDVI",
    "CIg",
    "CIre",
    "MTCI",
    "MCARI",
    "NDVIre1",
    "NDVIre2",
    "REPI",
    "NDII",
    "MSAVI",
    "LAI_re",
    "LAI_ndvi",
]
S2_COLS = S2_RAW_COLS + S2_INDEX_COLS
BAD_SCL_CLASSES = {0, 1, 3, 8, 9, 10, 11}


@dataclass(frozen=True)
class CompletedExport:
    path: Path
    city_token: str
    window_start: date
    window_end: date
    batch_index: int

    @property
    def window_key(self) -> str:
        return f"{self.window_start:%Y%m%d}_{self.window_end:%Y%m%d}"


def normalize_token(value: object) -> str:
    text = str(value or "").lower()
    text = text.replace("alberquerque", "albuquerque")
    return re.sub(r"[^a-z0-9]+", "", text)


def parse_completed_export(path: Path) -> CompletedExport | None:
    """Parse completed Sentinel export names, including Drive suffixes."""
    stem = path.stem
    match = re.search(
        r"(?:^|[/\\])(?:tree_centered_s2_raw15day_|s2_reduced_cells_|s2_(?:fast_)?(?:points_)?)?"
        r"(?P<city>[a-zA-Z0-9]+)_(?P<start>\d{8})_(?P<end>\d{8})_batch_(?P<batch>\d{5})(?:$|[-_].*)",
        stem,
        flags=re.IGNORECASE,
    )
    if not match:
        return None
    try:
        window_start = datetime.strptime(match.group("start"), "%Y%m%d").date()
        window_end = datetime.strptime(match.group("end"), "%Y%m%d").date()
        batch_index = int(match.group("batch"))
    except ValueError:
        return None
    return CompletedExport(
        path=path,
        city_token=normalize_token(match.group("city")),
        window_start=window_start,
        window_end=window_end,
        batch_index=batch_index,
    )


def token_matches(candidate: str, requested: set[str]) -> bool:
    if not requested:
        return True
    candidate = normalize_token(candidate)
    return any(token == candidate or token in candidate or candidate in token for token in requested)


def available_export_city_tokens(args) -> list[str]:
    tokens = set()
    for path in sorted(args.sentinel_dir.glob(args.input_pattern)):
        if not path.is_file() or path.stat().st_size <= 0:
            continue
        item = parse_completed_export(path)
        if item is not None:
            tokens.add(item.city_token)
    return sorted(tokens)


def discover_exports(args) -> dict[tuple[str, int], dict[str, list[CompletedExport]]]:
    requested = {normalize_token(city) for city in args.city_token}
    grouped: dict[tuple[str, int], dict[str, list[CompletedExport]]] = defaultdict(lambda: defaultdict(list))
    for path in sorted(args.sentinel_dir.glob(args.input_pattern)):
        if not path.is_file() or path.stat().st_size <= 0:
            continue
        item = parse_completed_export(path)
        if item is None:
            continue
        if not token_matches(item.city_token, requested):
            continue
        grouped[(item.city_token, item.batch_index)][item.window_key].append(item)
    return grouped


def choose_exports_for_batch(
    exports_by_window: dict[str, list[CompletedExport]],
) -> tuple[list[CompletedExport], int]:
    chosen = []
    duplicate_count = 0
    for _, exports in sorted(exports_by_window.items()):
        if len(exports) > 1:
            duplicate_count += len(exports) - 1
        selected = sorted(
            exports,
            key=lambda item: (item.path.stat().st_size, item.path.stat().st_mtime),
            reverse=True,
        )[0]
        chosen.append(selected)
    return chosen, duplicate_count


def truthy_series(values: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(values):
        return values.fillna(False)
    lowered = values.astype(str).str.strip().str.lower()
    return lowered.isin({"true", "1", "yes", "y", "t"})


def add_s2_indices(df: pd.DataFrame) -> pd.DataFrame:
    if all(column in df.columns for column in S2_INDEX_COLS):
        return df
    missing = [column for column in S2_RAW_COLS if column not in df.columns]
    if missing:
        raise ValueError(f"Sentinel-2 table lacks raw bands needed for indices: {missing}")
    x = df.copy()
    eps = 1e-10
    b = {column: pd.to_numeric(x[column], errors="coerce") for column in S2_RAW_COLS}
    x["NDVI"] = (b["B8"] - b["B4"]) / (b["B8"] + b["B4"] + eps)
    x["GNDVI"] = (b["B8"] - b["B3"]) / (b["B8"] + b["B3"] + eps)
    x["CIg"] = b["B8"] / (b["B3"] + eps) - 1
    x["CIre"] = b["B8A"] / (b["B5"] + eps) - 1
    x["MTCI"] = (b["B6"] - b["B5"]) / (b["B5"] - b["B4"] + eps)
    x["MCARI"] = ((b["B5"] - b["B4"]) - 0.2 * (b["B5"] - b["B3"])) * (b["B5"] / (b["B4"] + eps))
    x["NDVIre1"] = (b["B8"] - b["B5"]) / (b["B8"] + b["B5"] + eps)
    x["NDVIre2"] = (b["B8"] - b["B6"]) / (b["B8"] + b["B6"] + eps)
    x["REPI"] = 700 + 40 * (((b["B4"] + b["B7"]) / 2 - b["B5"]) / (b["B6"] - b["B5"] + eps))
    x["NDII"] = (b["B8"] - b["B11"]) / (b["B8"] + b["B11"] + eps)
    radicand = ((2 * b["B8"] + 1) ** 2 - 8 * (b["B8"] - b["B4"])).clip(lower=0)
    x["MSAVI"] = 0.5 * (2 * b["B8"] + 1 - np.sqrt(radicand))
    x["LAI_ndvi"] = np.clip(-np.log(((0.69 - x["NDVI"]) / 0.59).clip(lower=eps)), 0, 6) / 6
    x["LAI_re"] = np.clip(3.618 * x["CIre"] - 0.118, 0, 6) / 6
    return x.replace([np.inf, -np.inf], np.nan)


def normalize_chunk(chunk: pd.DataFrame, city_token: str, source_path: Path, args) -> pd.DataFrame:
    rename = {}
    lower_lookup = {str(column).lower(): column for column in chunk.columns}
    aliases = {
        "row_index": ["row_index", "reduced_id"],
        "acquisition_time": ["datetime", "acquisition_time", "date"],
        "source_image_id": ["source_image_id", "source_item_id", "image_id", "system:index"],
        "mgrs_tile": ["mgrs_tile", "MGRS_TILE"],
        "valid_pixel": ["valid_pixel"],
        "latitude": ["latitude", "lat"],
        "longitude": ["longitude", "lon"],
        "SCL": ["SCL", "scl"],
    }
    for target, choices in aliases.items():
        for choice in choices:
            source = lower_lookup.get(choice.lower())
            if source is not None:
                rename[source] = target
                break
    for band in S2_RAW_COLS:
        source = lower_lookup.get(band.lower())
        if source is not None:
            rename[source] = band
    frame = chunk.rename(columns=rename).copy()

    required = ["row_index", "acquisition_time", *S2_RAW_COLS]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"{source_path} is missing required columns: {missing}")

    frame["city_token"] = city_token
    frame["row_index"] = pd.to_numeric(frame["row_index"], errors="coerce")
    frame["acquisition_time"] = pd.to_datetime(frame["acquisition_time"], utc=True, errors="coerce")
    frame = frame.dropna(subset=["row_index", "acquisition_time"])
    if frame.empty:
        return frame
    frame["row_index"] = frame["row_index"].astype("int64")
    frame["date"] = frame["acquisition_time"].dt.date.astype(str)

    for column in S2_RAW_COLS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if "SCL" in frame.columns:
        frame["SCL"] = pd.to_numeric(frame["SCL"], errors="coerce")
    else:
        frame["SCL"] = np.nan
    if "valid_pixel" in frame.columns and not args.include_invalid_pixels:
        frame = frame.loc[truthy_series(frame["valid_pixel"])].copy()
    if args.drop_bad_scl:
        frame = frame.loc[~frame["SCL"].isin(BAD_SCL_CLASSES)].copy()
    return frame.loc[frame[S2_RAW_COLS].notna().any(axis=1)].copy()


def mode_or_nan(values: pd.Series) -> float:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    if clean.empty:
        return np.nan
    return float(clean.mode().iloc[0])


def aggregate_daily(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    if "source_image_id" not in frame.columns:
        frame = frame.copy()
        frame["source_image_id"] = frame.get("source_export", "unknown")
    aggregations: dict[str, object] = {column: "mean" for column in S2_COLS}
    aggregations.update({"acquisition_time": "min", "SCL": mode_or_nan, "source_image_id": "nunique"})
    if "latitude" in frame.columns:
        aggregations["latitude"] = "first"
    if "longitude" in frame.columns:
        aggregations["longitude"] = "first"

    grouped = frame.groupby(["city_token", "row_index", "date"], sort=True, as_index=False).agg(aggregations)
    grouped = grouped.rename(columns={"source_image_id": "source_image_count"})
    grouped["acquisition_time"] = pd.to_datetime(grouped["acquisition_time"], utc=True, errors="coerce")
    grouped = grouped.sort_values(["city_token", "row_index", "acquisition_time"], kind="stable")
    doy = grouped["acquisition_time"].dt.dayofyear.astype("float32")
    grouped["doy_sin"] = np.sin(2.0 * math.pi * doy / 365.25).astype("float32")
    grouped["doy_cos"] = np.cos(2.0 * math.pi * doy / 365.25).astype("float32")
    grouped["delta_days"] = (
        grouped.groupby(["city_token", "row_index"], sort=False)["acquisition_time"]
        .diff()
        .dt.days
        .fillna(0)
        .astype("float32")
    )
    return grouped
