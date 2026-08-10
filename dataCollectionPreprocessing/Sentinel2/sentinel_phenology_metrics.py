#!/usr/bin/env python3
"""Sentinel-2 phenology feature definitions and calculations."""

from __future__ import annotations

import math

import numpy as np


def clean_values(values: np.ndarray, outlier_abs: float = 0.0) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    if outlier_abs > 0:
        values = values.copy()
        values[np.abs(values) > outlier_abs] = 0.0
    return values


def safe_percentile(values: np.ndarray, percentile: float) -> float:
    values = np.asarray(values, dtype=np.float32)
    if values.size == 0:
        return 0.0
    return float(np.percentile(values, percentile))


def center_window(values: np.ndarray, fraction: float) -> np.ndarray:
    if values.ndim < 2 or values.size == 0:
        return values
    fraction = min(max(float(fraction), 0.05), 1.0)
    height, width = values.shape[:2]
    out_h = max(1, int(round(height * fraction)))
    out_w = max(1, int(round(width * fraction)))
    y0 = max(0, (height - out_h) // 2)
    x0 = max(0, (width - out_w) // 2)
    return values[y0 : y0 + out_h, x0 : x0 + out_w, ...]


_RESIZE_CACHE: dict[tuple[int, int, int, int], tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}


def resize_bilinear_2d(source: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    source = np.asarray(source, dtype=np.float32)
    out_h, out_w = shape
    in_h, in_w = source.shape
    if source.shape == shape:
        return source.copy()
    if in_h == 0 or in_w == 0 or out_h == 0 or out_w == 0:
        return np.zeros(shape, dtype=np.float32)
    key = (in_h, in_w, out_h, out_w)
    cached = _RESIZE_CACHE.get(key)
    if cached is None:
        y = np.linspace(0, in_h - 1, out_h, dtype=np.float32)
        x = np.linspace(0, in_w - 1, out_w, dtype=np.float32)
        y0 = np.floor(y).astype(np.int64)
        x0 = np.floor(x).astype(np.int64)
        y1 = np.clip(y0 + 1, 0, in_h - 1)
        x1 = np.clip(x0 + 1, 0, in_w - 1)
        wy = (y - y0).reshape(-1, 1)
        wx = (x - x0).reshape(1, -1)
        cached = (y0, x0, y1, x1, wy, wx)
        _RESIZE_CACHE[key] = cached
    y0, x0, y1, x1, wy, wx = cached
    top = source[y0[:, None], x0[None, :]] * (1.0 - wx) + source[y0[:, None], x1[None, :]] * wx
    bottom = source[y1[:, None], x0[None, :]] * (1.0 - wx) + source[y1[:, None], x1[None, :]] * wx
    return (top * (1.0 - wy) + bottom * wy).astype(np.float32)


def column_index(columns: list[str], name: str) -> int | None:
    try:
        return columns.index(name)
    except ValueError:
        return None


def doy_from_sin_cos(sin_values: np.ndarray, cos_values: np.ndarray) -> np.ndarray:
    angle = np.arctan2(sin_values, cos_values)
    angle = np.where(angle < 0, angle + 2.0 * math.pi, angle)
    return (angle / (2.0 * math.pi) * 365.25).astype(np.float32)


def doy_pair(value: float | None) -> tuple[float, float]:
    if value is None or not math.isfinite(value):
        return 0.0, 0.0
    angle = 2.0 * math.pi * float(value) / 365.25
    return float(math.sin(angle)), float(math.cos(angle))


def trapezoid_area(y: np.ndarray, x: np.ndarray) -> float:
    if y.size < 2 or x.size < 2:
        return float(y.mean()) if y.size else 0.0
    dx = np.diff(x.astype(np.float64, copy=False))
    avg_y = 0.5 * (y[:-1].astype(np.float64, copy=False) + y[1:].astype(np.float64, copy=False))
    return float(np.sum(dx * avg_y))


def phenology_feature_names() -> list[str]:
    names: list[str] = []
    for index_name in ("NDVI", "EVI2", "NDII", "NDVIre1", "CIre", "GNDVI", "LAI_ndvi"):
        for stat in (
            "mean",
            "median",
            "max",
            "min",
            "amplitude",
            "std",
            "iqr",
            "early_mean",
            "mid_mean",
            "late_mean",
            "late_minus_early",
        ):
            names.append(f"sentinel_pheno_{index_name}_{stat}")
    names.extend(
        [
            "sentinel_pheno_ndvi_frac_gt_0p50",
            "sentinel_pheno_ndvi_frac_gt_0p60",
            "sentinel_pheno_ndvi_frac_gt_0p70",
            "sentinel_pheno_ndvi_auc",
            "sentinel_pheno_ndvi_peak_doy_sin",
            "sentinel_pheno_ndvi_peak_doy_cos",
            "sentinel_pheno_ndvi_greenup_doy_sin",
            "sentinel_pheno_ndvi_greenup_doy_cos",
            "sentinel_pheno_ndvi_senescence_doy_sin",
            "sentinel_pheno_ndvi_senescence_doy_cos",
            "sentinel_pheno_ndvi_growing_season_length",
            "sentinel_pheno_ndvi_spring_slope",
            "sentinel_pheno_ndvi_fall_slope",
            "sentinel_pheno_clear_observation_fraction",
            "sentinel_pheno_interpolated_fraction",
            "sentinel_pheno_mean_source_image_count",
            "sentinel_pheno_max_interpolation_gap_days",
            "sentinel_pheno_mean_delta_days",
        ]
    )
    return names


SENTINEL_PHENOLOGY_COLUMNS = tuple(phenology_feature_names())


def sequence_series(values: np.ndarray, columns: list[str], name: str) -> np.ndarray:
    if name == "EVI2":
        b8_index = column_index(columns, "B8")
        b4_index = column_index(columns, "B4")
        if b8_index is None or b4_index is None:
            return np.zeros(values.shape[0], dtype=np.float32)
        b8 = values[:, b8_index]
        b4 = values[:, b4_index]
        return ((2.5 * (b8 - b4)) / (b8 + 2.4 * b4 + 1.0)).astype(np.float32)
    index = column_index(columns, name)
    if index is None or index >= values.shape[1]:
        return np.zeros(values.shape[0], dtype=np.float32)
    return values[:, index].astype(np.float32)


def seasonal_means(series: np.ndarray, doy: np.ndarray | None) -> tuple[float, float, float]:
    if series.size == 0:
        return 0.0, 0.0, 0.0
    if doy is None or doy.size != series.size:
        thirds = np.array_split(series, 3)
        return tuple(float(part.mean()) if part.size else 0.0 for part in thirds)  # type: ignore[return-value]
    early = series[(doy >= 60) & (doy < 150)]
    mid = series[(doy >= 150) & (doy < 240)]
    late = series[(doy >= 240) & (doy < 335)]
    return (
        float(early.mean()) if early.size else 0.0,
        float(mid.mean()) if mid.size else 0.0,
        float(late.mean()) if late.size else 0.0,
    )


def compute_sentinel_phenology(
    values: np.ndarray,
    mask: np.ndarray,
    columns: list[str],
    outlier_abs: float,
) -> np.ndarray:
    values = clean_values(values, outlier_abs)
    mask = np.asarray(mask, dtype=bool)
    if values.ndim != 2 or values.shape[0] == 0 or not mask.any():
        return np.zeros(len(SENTINEL_PHENOLOGY_COLUMNS), dtype=np.float32)
    values = values[mask]
    if values.size == 0:
        return np.zeros(len(SENTINEL_PHENOLOGY_COLUMNS), dtype=np.float32)

    doy = None
    doy_sin_index = column_index(columns, "doy_sin")
    doy_cos_index = column_index(columns, "doy_cos")
    if doy_sin_index is not None and doy_cos_index is not None and doy_cos_index < values.shape[1]:
        doy = doy_from_sin_cos(values[:, doy_sin_index], values[:, doy_cos_index])
        order = np.argsort(doy)
        values = values[order]
        doy = doy[order]

    output: list[float] = []
    ndvi = sequence_series(values, columns, "NDVI")
    for index_name in ("NDVI", "EVI2", "NDII", "NDVIre1", "CIre", "GNDVI", "LAI_ndvi"):
        series = clean_values(sequence_series(values, columns, index_name))
        q25 = safe_percentile(series, 25)
        q75 = safe_percentile(series, 75)
        early, mid, late = seasonal_means(series, doy)
        output.extend(
            [
                float(series.mean()) if series.size else 0.0,
                float(np.median(series)) if series.size else 0.0,
                float(series.max()) if series.size else 0.0,
                float(series.min()) if series.size else 0.0,
                float(series.max() - series.min()) if series.size else 0.0,
                float(series.std()) if series.size else 0.0,
                float(q75 - q25),
                early,
                mid,
                late,
                float(late - early),
            ]
        )

    output.extend(
        [
            float((ndvi > 0.50).mean()) if ndvi.size else 0.0,
            float((ndvi > 0.60).mean()) if ndvi.size else 0.0,
            float((ndvi > 0.70).mean()) if ndvi.size else 0.0,
        ]
    )

    if ndvi.size and doy is not None and doy.size == ndvi.size:
        peak_index = int(np.argmax(ndvi))
        peak_doy = float(doy[peak_index])
        min_value = float(ndvi.min())
        amplitude = max(float(ndvi.max() - ndvi.min()), 1e-6)
        threshold = min_value + 0.20 * amplitude
        above = np.nonzero(ndvi >= threshold)[0]
        greenup_doy = float(doy[above[0]]) if above.size else None
        after_peak = above[above >= peak_index]
        senescence_doy = float(doy[after_peak[-1]]) if after_peak.size else None
        season_length = (
            max(0.0, senescence_doy - greenup_doy)
            if greenup_doy is not None and senescence_doy is not None
            else 0.0
        )
        auc = float(trapezoid_area(ndvi, doy) / 365.25)
        spring_slope = float((ndvi[peak_index] - ndvi[0]) / max(doy[peak_index] - doy[0], 1.0)) if peak_index > 0 else 0.0
        fall_slope = (
            float((ndvi[-1] - ndvi[peak_index]) / max(doy[-1] - doy[peak_index], 1.0))
            if peak_index < ndvi.size - 1
            else 0.0
        )
        peak_sin, peak_cos = doy_pair(peak_doy)
        green_sin, green_cos = doy_pair(greenup_doy)
        sen_sin, sen_cos = doy_pair(senescence_doy)
    else:
        auc = float(ndvi.mean()) if ndvi.size else 0.0
        peak_sin = peak_cos = green_sin = green_cos = sen_sin = sen_cos = 0.0
        season_length = spring_slope = fall_slope = 0.0

    def mean_column(name: str) -> float:
        index = column_index(columns, name)
        if index is None or index >= values.shape[1]:
            return 0.0
        return float(clean_values(values[:, index]).mean())

    def max_column(name: str) -> float:
        index = column_index(columns, name)
        if index is None or index >= values.shape[1]:
            return 0.0
        series = clean_values(values[:, index])
        return float(series.max()) if series.size else 0.0

    output.extend(
        [
            auc,
            peak_sin,
            peak_cos,
            green_sin,
            green_cos,
            sen_sin,
            sen_cos,
            float(season_length),
            spring_slope,
            fall_slope,
            mean_column("sentinel_observed"),
            mean_column("sentinel_interpolated"),
            mean_column("source_image_count"),
            max_column("interpolation_gap_days"),
            mean_column("delta_days"),
        ]
    )
    return np.nan_to_num(np.asarray(output, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)



