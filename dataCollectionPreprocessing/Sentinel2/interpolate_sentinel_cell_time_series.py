"""Interpolate daily Sentinel-2 cell observations onto regular gap intervals.

This module is used by ``derive_tree_centered_sentinel_phenology.py``. Observed
rows are preserved. When consecutive observations are separated by more than
``gap_threshold_days``, regularly spaced rows are inserted every
``interpolation_interval_days`` and spectral values are linearly interpolated
between the two observations.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

import sentinel_cell_features as sentinel_base


METADATA_COLUMNS = (
    "sentinel_observed",
    "sentinel_interpolated",
    "data_quality_mask",
    "source_image_count",
    "interpolation_gap_days",
    "days_since_previous_observation",
    "days_until_next_observation",
    "delta_days",
    "doy_sin",
    "doy_cos",
)


def _date_column(frame: pd.DataFrame) -> pd.Series:
    if "acquisition_time" in frame.columns:
        values = pd.to_datetime(frame["acquisition_time"], utc=True, errors="coerce")
    elif "date" in frame.columns:
        values = pd.to_datetime(frame["date"], utc=True, errors="coerce")
    else:
        raise ValueError("Daily Sentinel observations require date or acquisition_time")
    return values.dt.normalize()


def _observed_row(row: pd.Series, observed_date: pd.Timestamp) -> dict[str, object]:
    out = row.to_dict()
    out["date"] = observed_date.tz_localize(None)
    out["sentinel_observed"] = 1.0
    out["sentinel_interpolated"] = 0.0
    out["data_quality_mask"] = 1.0
    out["interpolation_gap_days"] = 0.0
    out["days_since_previous_observation"] = 0.0
    out["days_until_next_observation"] = 0.0
    return out


def _interpolated_row(
    previous: pd.Series,
    following: pd.Series,
    previous_date: pd.Timestamp,
    following_date: pd.Timestamp,
    target_date: pd.Timestamp,
) -> dict[str, object]:
    span_days = float((following_date - previous_date).days)
    elapsed_days = float((target_date - previous_date).days)
    remaining_days = float((following_date - target_date).days)
    weight = elapsed_days / span_days
    out = previous.to_dict()
    out["date"] = target_date.tz_localize(None)
    out["acquisition_time"] = target_date
    for column in sentinel_base.S2_COLS:
        before = pd.to_numeric(pd.Series([previous.get(column)]), errors="coerce").iloc[0]
        after = pd.to_numeric(pd.Series([following.get(column)]), errors="coerce").iloc[0]
        if pd.notna(before) and pd.notna(after):
            out[column] = float(before) + (float(after) - float(before)) * weight
        else:
            out[column] = np.nan
    out["sentinel_observed"] = 0.0
    out["sentinel_interpolated"] = 1.0
    out["data_quality_mask"] = 1.0
    out["source_image_count"] = 0.0
    out["interpolation_gap_days"] = span_days
    out["days_since_previous_observation"] = elapsed_days
    out["days_until_next_observation"] = remaining_days
    out["delta_days"] = elapsed_days
    doy = float(target_date.dayofyear)
    out["doy_sin"] = float(np.sin(2.0 * np.pi * doy / 365.25))
    out["doy_cos"] = float(np.cos(2.0 * np.pi * doy / 365.25))
    return out


def interpolate_cell(group: pd.DataFrame, batch_index: int, args: argparse.Namespace) -> list[dict[str, object]]:
    """Return observed and interpolated rows for one Sentinel cell."""

    if group.empty:
        return []
    threshold = int(args.gap_threshold_days)
    interval = int(args.interpolation_interval_days)
    if threshold < 0:
        raise ValueError("gap_threshold_days must be nonnegative")
    if interval <= 0:
        raise ValueError("interpolation_interval_days must be positive")

    frame = group.copy()
    frame["_observation_date"] = _date_column(frame)
    frame = frame.dropna(subset=["_observation_date"])
    frame = frame.sort_values("_observation_date", kind="stable")
    frame = frame.drop_duplicates("_observation_date", keep="first").reset_index(drop=True)
    if frame.empty:
        return []

    rows: list[dict[str, object]] = []
    for position in range(len(frame)):
        current = frame.iloc[position]
        current_date = current["_observation_date"]
        observed = _observed_row(current.drop(labels=["_observation_date"]), current_date)
        observed["batch_index"] = int(batch_index)
        rows.append(observed)
        if position + 1 >= len(frame):
            continue
        following = frame.iloc[position + 1]
        following_date = following["_observation_date"]
        gap_days = int((following_date - current_date).days)
        if gap_days <= threshold:
            continue
        offset = interval
        while offset < gap_days:
            target_date = current_date + pd.Timedelta(days=offset)
            interpolated = _interpolated_row(current, following, current_date, following_date, target_date)
            interpolated.pop("_observation_date", None)
            interpolated["batch_index"] = int(batch_index)
            rows.append(interpolated)
            offset += interval
    return rows


def interpolate_time_series(daily: pd.DataFrame, batch_index: int, args: argparse.Namespace) -> pd.DataFrame:
    """Interpolate every ``row_index`` group in a daily Sentinel table."""

    if daily.empty:
        return pd.DataFrame(columns=["row_index", "date", *sentinel_base.S2_COLS, *METADATA_COLUMNS])
    if "row_index" not in daily.columns:
        raise ValueError("Daily Sentinel observations require row_index")

    rows: list[dict[str, object]] = []
    groups = daily.groupby("row_index", sort=False)
    progress_every = int(getattr(args, "progress_every_cells", 0) or 0)
    for group_number, (_row_index, group) in enumerate(groups, start=1):
        rows.extend(interpolate_cell(group, batch_index, args))
        if progress_every > 0 and group_number % progress_every == 0:
            print(f"Interpolated {group_number:,} Sentinel cell(s)", flush=True)
    if not rows:
        return pd.DataFrame(columns=["row_index", "date", *sentinel_base.S2_COLS, *METADATA_COLUMNS])

    result = pd.DataFrame(rows)
    result["date"] = pd.to_datetime(result["date"], errors="coerce")
    result = result.sort_values(["row_index", "date"], kind="stable").reset_index(drop=True)
    result["delta_days"] = (
        result.groupby("row_index", sort=False)["date"].diff().dt.days.fillna(0).astype("float32")
    )
    doy = result["date"].dt.dayofyear.astype("float32")
    result["doy_sin"] = np.sin(2.0 * np.pi * doy / 365.25).astype("float32")
    result["doy_cos"] = np.cos(2.0 * np.pi * doy / 365.25).astype("float32")
    return result
