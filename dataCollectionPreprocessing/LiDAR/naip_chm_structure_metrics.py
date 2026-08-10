#!/usr/bin/env python3
"""NAIP/CHM structural feature definitions and calculations."""

from __future__ import annotations

import math
from collections.abc import Mapping

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


def structure_feature_names() -> list[str]:
    names = [
        "structure_naip_valid_fraction",
        "structure_naip_center_valid_fraction",
        "structure_naip_veg_fraction_ndvi_gt_0p30",
        "structure_naip_veg_fraction_ndvi_gt_0p40",
        "structure_naip_veg_fraction_ndvi_gt_0p50",
        "structure_naip_veg_fraction_ndvi_gt_0p60",
        "structure_naip_center_veg_fraction_ndvi_gt_0p30",
        "structure_naip_center_veg_fraction_ndvi_gt_0p40",
        "structure_naip_ndvi_mean",
        "structure_naip_ndvi_median",
        "structure_naip_ndvi_std",
        "structure_naip_ndvi_iqr",
        "structure_naip_ndvi_p10",
        "structure_naip_ndvi_p90",
        "structure_naip_center_ndvi_mean",
        "structure_naip_center_ndvi_std",
        "structure_naip_veg_ndvi_mean",
        "structure_naip_veg_ndvi_std",
        "structure_naip_veg_blue_mean",
        "structure_naip_veg_green_mean",
        "structure_naip_veg_red_mean",
        "structure_naip_veg_nir_mean",
        "structure_naip_veg_brightness_mean",
        "structure_naip_veg_green_red_diff_mean",
        "structure_naip_veg_nir_red_diff_mean",
        "structure_naip_center_component_fraction",
        "structure_naip_center_component_bbox_fraction",
        "structure_naip_center_component_compactness",
        "structure_naip_center_component_aspect_ratio",
        "structure_naip_center_component_extent",
        "structure_naip_ndvi_texture_contrast",
        "structure_naip_ndvi_texture_edge_density",
        "structure_naip_brightness_texture_contrast",
        "structure_naip_brightness_texture_edge_density",
        "structure_chm_valid_fraction",
        "structure_chm_center_valid_fraction",
        "structure_chm_mean_m",
        "structure_chm_std_m",
        "structure_chm_p50_m",
        "structure_chm_p75_m",
        "structure_chm_p90_m",
        "structure_chm_p95_m",
        "structure_chm_max_m",
        "structure_chm_center_mean_m",
        "structure_chm_center_p95_m",
        "structure_chm_texture_contrast_m2",
        "structure_chm_texture_edge_density_gt_1m",
        "structure_veg_chm_valid_fraction",
        "structure_veg_chm_mean_m",
        "structure_veg_chm_std_m",
        "structure_veg_chm_p50_m",
        "structure_veg_chm_p75_m",
        "structure_veg_chm_p90_m",
        "structure_veg_chm_p95_m",
        "structure_veg_chm_max_m",
        "structure_veg_chm_center_mean_m",
        "structure_veg_chm_center_p95_m",
        "structure_veg_chm_texture_contrast_m2",
        "structure_veg_chm_texture_edge_density_gt_1m",
        "structure_veg_chm_fraction_gt_2m",
        "structure_veg_chm_fraction_gt_5m",
        "structure_veg_chm_fraction_gt_10m",
        "structure_veg_chm_fraction_gt_15m",
        "structure_canopy_volume_sum_per_pixel",
        "structure_canopy_volume_mean_x_area",
        "structure_canopy_p95_x_area",
        "structure_tree_like_fraction_ndvi_gt_0p40_height_gt_2m",
        "structure_low_vegetation_fraction_ndvi_gt_0p30_height_lt_2m",
        "structure_tall_nonveg_fraction_ndvi_lt_0p30_height_gt_2m",
        "structure_center_tree_like_fraction_ndvi_gt_0p40_height_gt_2m",
        "structure_center_low_vegetation_fraction_ndvi_gt_0p30_height_lt_2m",
        "structure_center_tall_nonveg_fraction_ndvi_lt_0p30_height_gt_2m",
        "structure_tree_ndvi_mean",
        "structure_tree_ndvi_std",
        "structure_tree_chm_mean_m",
        "structure_tree_chm_p95_m",
        "structure_tree_chm_max_m",
        "structure_tree_canopy_volume_sum_per_pixel",
        "structure_tree_ndvi_mean_x_chm_p95",
        "structure_tree_chm_p95_minus_p50_m",
        "structure_tree_chm_p95_minus_p10_m",
        "structure_tree_chm_cv",
        "structure_veg_chm_p95_minus_p50_m",
        "structure_veg_chm_p95_minus_p10_m",
        "structure_veg_chm_cv",
        "structure_tree_like_fraction_gt_5m",
        "structure_tree_like_fraction_gt_10m",
        "structure_center_tree_like_fraction_gt_5m",
        "structure_center_tree_like_fraction_gt_10m",
        "structure_tree_like_to_veg_area_ratio",
        "structure_low_veg_to_veg_area_ratio",
        "structure_tree_like_to_tall_area_ratio",
        "structure_tall_nonveg_to_tall_area_ratio",
        "structure_center_minus_ring_tree_like_fraction",
        "structure_center_minus_ring_low_veg_fraction",
        "structure_center_minus_ring_tall_nonveg_fraction",
        "structure_center_minus_ring_ndvi_mean",
        "structure_center_minus_ring_veg_chm_mean_m",
        "structure_center_minus_ring_veg_chm_p95_m",
        "structure_ring_tree_like_fraction",
        "structure_ring_veg_chm_mean_m",
        "structure_ring_veg_chm_p95_m",
        "structure_canopy_edge_to_area_ratio",
        "structure_ndvi_edge_to_veg_area_ratio",
        "structure_brightness_edge_to_veg_area_ratio",
        "structure_tree_naip_blue_mean",
        "structure_tree_naip_green_mean",
        "structure_tree_naip_red_mean",
        "structure_tree_naip_nir_mean",
        "structure_tree_naip_brightness_mean",
        "structure_tree_naip_green_red_diff_mean",
        "structure_tree_naip_nir_red_diff_mean",
        "structure_tree_naip_ndvi_iqr",
        "structure_tree_naip_brightness_std",
    ]
    return names


NAIP_CHM_STRUCTURE_COLUMNS = tuple(structure_feature_names())


def stats(values: np.ndarray) -> dict[str, float]:
    values = clean_values(values)
    if values.size == 0:
        return {key: 0.0 for key in ("mean", "median", "std", "iqr", "p10", "p50", "p75", "p90", "p95", "max")}
    p10, p25, p50, p75, p90, p95 = np.percentile(values, [10, 25, 50, 75, 90, 95])
    return {
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "std": float(values.std()),
        "iqr": float(p75 - p25),
        "p10": float(p10),
        "p50": float(p50),
        "p75": float(p75),
        "p90": float(p90),
        "p95": float(p95),
        "max": float(values.max()),
    }


def texture_summary(values: np.ndarray, valid: np.ndarray, edge_threshold: float) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float32)
    valid = np.asarray(valid, dtype=bool) & np.isfinite(values)
    if values.ndim != 2 or values.size == 0 or valid.sum() < 2:
        return 0.0, 0.0
    diffs = []
    if values.shape[1] > 1:
        pair_valid = valid[:, 1:] & valid[:, :-1]
        if pair_valid.any():
            diffs.append(values[:, 1:][pair_valid] - values[:, :-1][pair_valid])
    if values.shape[0] > 1:
        pair_valid = valid[1:, :] & valid[:-1, :]
        if pair_valid.any():
            diffs.append(values[1:, :][pair_valid] - values[:-1, :][pair_valid])
    if not diffs:
        return 0.0, 0.0
    diff = np.concatenate(diffs).astype(np.float32)
    abs_diff = np.abs(diff)
    return float(np.mean(diff * diff)), float(np.mean(abs_diff >= float(edge_threshold)))


def safe_ratio(numerator: float, denominator: float) -> float:
    denominator = float(denominator)
    if not math.isfinite(denominator) or abs(denominator) < 1e-6:
        return 0.0
    value = float(numerator) / denominator
    return value if math.isfinite(value) else 0.0


def ring_window(values: np.ndarray, fraction: float) -> np.ndarray:
    if values.ndim < 2 or values.size == 0:
        return values.reshape(0, *values.shape[2:]) if values.ndim >= 2 else values
    fraction = min(max(float(fraction), 0.05), 1.0)
    if fraction >= 1.0:
        return values.reshape(0, *values.shape[2:])
    height, width = values.shape[:2]
    out_h = max(1, int(round(height * fraction)))
    out_w = max(1, int(round(width * fraction)))
    y0 = max(0, (height - out_h) // 2)
    x0 = max(0, (width - out_w) // 2)
    mask = np.ones((height, width), dtype=bool)
    mask[y0 : y0 + out_h, x0 : x0 + out_w] = False
    return values[mask]


def fast_component_summary(mask: np.ndarray) -> tuple[float, float, float, float, float]:
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 2 or mask.size == 0 or not mask.any():
        return 0.0, 0.0, 0.0, 0.0, 0.0
    area = int(mask.sum())
    ys, xs = np.nonzero(mask)
    bbox_h = int(ys.max() - ys.min() + 1)
    bbox_w = int(xs.max() - xs.min() + 1)
    bbox_area = int(bbox_h * bbox_w)
    padded = np.pad(mask, 1, mode="constant", constant_values=False)
    center = padded[1:-1, 1:-1]
    perimeter = int(
        ((center) & ~padded[:-2, 1:-1]).sum()
        + ((center) & ~padded[2:, 1:-1]).sum()
        + ((center) & ~padded[1:-1, :-2]).sum()
        + ((center) & ~padded[1:-1, 2:]).sum()
    )
    area_fraction = float(area / mask.size)
    bbox_fraction = float(bbox_area / mask.size)
    compactness = float(4.0 * math.pi * area / max(perimeter * perimeter, 1))
    aspect_ratio = float(max(bbox_h, bbox_w) / max(min(bbox_h, bbox_w), 1))
    extent = float(area / max(bbox_area, 1))
    return area_fraction, bbox_fraction, compactness, aspect_ratio, extent


def center_connected_component(mask: np.ndarray) -> tuple[float, float, float, float, float]:
    # Kept only as an opt-in exact diagnostic. The fast summary is preferred for
    # production sidecars because Python BFS over hundreds of thousands of chips
    # dominates runtime.
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 2 or mask.size == 0 or not mask.any():
        return 0.0, 0.0, 0.0, 0.0, 0.0
    height, width = mask.shape
    cy, cx = height // 2, width // 2
    if not mask[cy, cx]:
        ys, xs = np.nonzero(mask)
        distances = (ys - cy) ** 2 + (xs - cx) ** 2
        nearest = int(np.argmin(distances))
        cy, cx = int(ys[nearest]), int(xs[nearest])

    visited = np.zeros_like(mask, dtype=bool)
    queue: list[tuple[int, int]] = [(cy, cx)]
    visited[cy, cx] = True
    coords = []
    perimeter = 0
    while queue:
        y, x = queue.pop()
        coords.append((y, x))
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            ny, nx = y + dy, x + dx
            if ny < 0 or ny >= height or nx < 0 or nx >= width or not mask[ny, nx]:
                perimeter += 1
                continue
            if not visited[ny, nx]:
                visited[ny, nx] = True
                queue.append((ny, nx))
    area = len(coords)
    ys = [coord[0] for coord in coords]
    xs = [coord[1] for coord in coords]
    bbox_h = max(ys) - min(ys) + 1
    bbox_w = max(xs) - min(xs) + 1
    bbox_area = bbox_h * bbox_w
    return (
        float(area / mask.size),
        float(bbox_area / mask.size),
        float(4.0 * math.pi * area / max(perimeter * perimeter, 1)),
        float(max(bbox_h, bbox_w) / max(min(bbox_h, bbox_w), 1)),
        float(area / max(bbox_area, 1)),
    )


def compute_naip_chm_structure(
    data: Mapping[str, np.ndarray],
    sample_index: int,
    center_fraction: float,
    vegetation_ndvi_threshold: float,
    tree_ndvi_threshold: float,
    tree_height_threshold_m: float,
    component_mode: str,
) -> np.ndarray:
    rgbnir = np.asarray(data["naip_rgbnir"][sample_index], dtype=np.float32) / 255.0
    ndvi = clean_values(data["naip_ndvi"][sample_index])
    valid = np.asarray(data["naip_valid_mask"][sample_index], dtype=bool)
    chm = np.asarray(data["chm"][sample_index], dtype=np.float32)
    chm_valid = np.asarray(data["chm_valid_mask"][sample_index], dtype=bool) & np.isfinite(chm)
    chm_clean = clean_values(chm)
    if "vegetation_chm" in data:
        veg_chm = clean_values(data["vegetation_chm"][sample_index])
    else:
        veg_chm = np.where((ndvi >= vegetation_ndvi_threshold) & chm_valid, chm_clean, 0.0).astype(np.float32)
    if "vegetation_chm_weight" in data:
        veg_weight = clean_values(data["vegetation_chm_weight"][sample_index])
        veg_chm_valid = chm_valid & (veg_weight > 0)
    else:
        veg_chm_valid = chm_valid & (ndvi >= vegetation_ndvi_threshold)

    center_valid = center_window(valid, center_fraction)
    center_ndvi = center_window(ndvi, center_fraction)
    center_chm = center_window(chm_clean, center_fraction)
    center_chm_valid = center_window(chm_valid, center_fraction)
    center_veg_chm = center_window(veg_chm, center_fraction)
    center_veg_chm_valid = center_window(veg_chm_valid, center_fraction)

    valid_ndvi = ndvi[valid]
    ndvi_stats = stats(valid_ndvi)
    center_valid_ndvi = center_ndvi[center_valid]
    center_ndvi_stats = stats(center_valid_ndvi)
    veg_mask = valid & (ndvi >= vegetation_ndvi_threshold)
    tree_veg_mask = valid & (ndvi >= tree_ndvi_threshold)
    veg_ndvi_stats = stats(ndvi[veg_mask])

    if rgbnir.ndim == 3 and rgbnir.shape[2] >= 4 and veg_mask.any():
        veg_pixels = rgbnir[veg_mask]
        blue_mean, green_mean, red_mean, nir_mean = [float(veg_pixels[:, i].mean()) for i in range(4)]
    else:
        blue_mean = green_mean = red_mean = nir_mean = 0.0
    brightness_image = (
        rgbnir[:, :, :3].mean(axis=2).astype(np.float32)
        if rgbnir.ndim == 3 and rgbnir.shape[2] >= 3
        else np.zeros_like(ndvi, dtype=np.float32)
    )
    brightness = float(np.mean([blue_mean, green_mean, red_mean])) if veg_mask.any() else 0.0
    ndvi_texture_contrast, ndvi_texture_edge_density = texture_summary(ndvi, valid, edge_threshold=0.10)
    brightness_texture_contrast, brightness_texture_edge_density = texture_summary(
        brightness_image,
        valid,
        edge_threshold=0.10,
    )

    if component_mode == "connected":
        (
            component_fraction,
            component_bbox_fraction,
            component_compactness,
            component_aspect_ratio,
            component_extent,
        ) = center_connected_component(tree_veg_mask)
    elif component_mode == "fast":
        (
            component_fraction,
            component_bbox_fraction,
            component_compactness,
            component_aspect_ratio,
            component_extent,
        ) = fast_component_summary(tree_veg_mask)
    else:
        component_fraction = component_bbox_fraction = component_compactness = component_aspect_ratio = component_extent = 0.0

    valid_chm = chm_clean[chm_valid]
    chm_stats = stats(valid_chm)
    center_valid_chm = center_chm[center_chm_valid]
    center_chm_stats = stats(center_valid_chm)
    valid_veg_chm = veg_chm[veg_chm_valid]
    veg_chm_stats = stats(valid_veg_chm)
    center_valid_veg_chm = center_veg_chm[center_veg_chm_valid]
    center_veg_chm_stats = stats(center_valid_veg_chm)
    chm_texture_contrast, chm_texture_edge_density = texture_summary(chm_clean, chm_valid, edge_threshold=1.0)
    veg_chm_texture_contrast, veg_chm_texture_edge_density = texture_summary(veg_chm, veg_chm_valid, edge_threshold=1.0)

    ndvi_on_chm = resize_bilinear_2d(ndvi, chm_clean.shape)
    valid_on_chm = resize_bilinear_2d(valid.astype(np.float32), chm_clean.shape) >= 0.5
    pixel_count = max(1, int(veg_chm.size))
    veg_area_fraction = float(veg_chm_valid.mean()) if veg_chm_valid.size else 0.0
    tree_like = valid_on_chm & (ndvi_on_chm >= tree_ndvi_threshold) & chm_valid & (chm_clean >= tree_height_threshold_m)
    low_veg = valid_on_chm & (ndvi_on_chm >= vegetation_ndvi_threshold) & (~chm_valid | (chm_clean < tree_height_threshold_m))
    tall_nonveg = valid_on_chm & (ndvi_on_chm < vegetation_ndvi_threshold) & chm_valid & (chm_clean >= tree_height_threshold_m)
    center_tree_like = center_window(tree_like, center_fraction)
    center_low_veg = center_window(low_veg, center_fraction)
    center_tall_nonveg = center_window(tall_nonveg, center_fraction)
    tree_ndvi = ndvi_on_chm[tree_like]
    tree_chm = chm_clean[tree_like]
    tree_ndvi_stats = stats(tree_ndvi)
    tree_chm_stats = stats(tree_chm)
    veg_on_chm = valid_on_chm & (ndvi_on_chm >= vegetation_ndvi_threshold)
    tall_area = chm_valid & (chm_clean >= tree_height_threshold_m)
    tree_like_gt_5m = tree_like & (chm_clean >= 5.0)
    tree_like_gt_10m = tree_like & (chm_clean >= 10.0)
    center_tree_like_gt_5m = center_window(tree_like_gt_5m, center_fraction)
    center_tree_like_gt_10m = center_window(tree_like_gt_10m, center_fraction)

    ring_tree_like = ring_window(tree_like, center_fraction)
    ring_low_veg = ring_window(low_veg, center_fraction)
    ring_tall_nonveg = ring_window(tall_nonveg, center_fraction)
    ring_ndvi = ring_window(ndvi, center_fraction)
    ring_valid = ring_window(valid, center_fraction).astype(bool)
    ring_ndvi_stats = stats(ring_ndvi[ring_valid]) if ring_ndvi.size and ring_valid.size else stats(np.asarray([], dtype=np.float32))
    ring_veg_chm = ring_window(veg_chm, center_fraction)
    ring_veg_chm_valid = ring_window(veg_chm_valid, center_fraction).astype(bool)
    ring_veg_chm_stats = (
        stats(ring_veg_chm[ring_veg_chm_valid])
        if ring_veg_chm.size and ring_veg_chm_valid.size
        else stats(np.asarray([], dtype=np.float32))
    )

    tree_area_fraction = float(tree_like.mean()) if tree_like.size else 0.0
    veg_area_on_chm_fraction = float(veg_on_chm.mean()) if veg_on_chm.size else 0.0
    tall_area_fraction = float(tall_area.mean()) if tall_area.size else 0.0
    center_tree_area_fraction = float(center_tree_like.mean()) if center_tree_like.size else 0.0
    center_low_veg_fraction = float(center_low_veg.mean()) if center_low_veg.size else 0.0
    center_tall_nonveg_fraction = float(center_tall_nonveg.mean()) if center_tall_nonveg.size else 0.0
    ring_tree_area_fraction = float(ring_tree_like.mean()) if ring_tree_like.size else 0.0
    ring_low_veg_fraction = float(ring_low_veg.mean()) if ring_low_veg.size else 0.0
    ring_tall_nonveg_fraction = float(ring_tall_nonveg.mean()) if ring_tall_nonveg.size else 0.0

    if rgbnir.ndim == 3 and rgbnir.shape[2] >= 4 and tree_veg_mask.any():
        tree_pixels = rgbnir[tree_veg_mask]
        tree_blue_mean, tree_green_mean, tree_red_mean, tree_nir_mean = [
            float(tree_pixels[:, i].mean()) for i in range(4)
        ]
        tree_brightness_values = tree_pixels[:, :3].mean(axis=1)
        tree_brightness_mean = float(tree_brightness_values.mean())
        tree_brightness_std = float(tree_brightness_values.std())
    else:
        tree_blue_mean = tree_green_mean = tree_red_mean = tree_nir_mean = 0.0
        tree_brightness_mean = tree_brightness_std = 0.0

    values = [
        float(valid.mean()) if valid.size else 0.0,
        float(center_valid.mean()) if center_valid.size else 0.0,
        float((valid_ndvi > 0.30).mean()) if valid_ndvi.size else 0.0,
        float((valid_ndvi > 0.40).mean()) if valid_ndvi.size else 0.0,
        float((valid_ndvi > 0.50).mean()) if valid_ndvi.size else 0.0,
        float((valid_ndvi > 0.60).mean()) if valid_ndvi.size else 0.0,
        float((center_valid_ndvi > 0.30).mean()) if center_valid_ndvi.size else 0.0,
        float((center_valid_ndvi > 0.40).mean()) if center_valid_ndvi.size else 0.0,
        ndvi_stats["mean"],
        ndvi_stats["median"],
        ndvi_stats["std"],
        ndvi_stats["iqr"],
        ndvi_stats["p10"],
        ndvi_stats["p90"],
        center_ndvi_stats["mean"],
        center_ndvi_stats["std"],
        veg_ndvi_stats["mean"],
        veg_ndvi_stats["std"],
        blue_mean,
        green_mean,
        red_mean,
        nir_mean,
        brightness,
        float(green_mean - red_mean),
        float(nir_mean - red_mean),
        component_fraction,
        component_bbox_fraction,
        component_compactness,
        component_aspect_ratio,
        component_extent,
        ndvi_texture_contrast,
        ndvi_texture_edge_density,
        brightness_texture_contrast,
        brightness_texture_edge_density,
        float(chm_valid.mean()) if chm_valid.size else 0.0,
        float(center_chm_valid.mean()) if center_chm_valid.size else 0.0,
        chm_stats["mean"],
        chm_stats["std"],
        chm_stats["p50"],
        chm_stats["p75"],
        chm_stats["p90"],
        chm_stats["p95"],
        chm_stats["max"],
        center_chm_stats["mean"],
        center_chm_stats["p95"],
        chm_texture_contrast,
        chm_texture_edge_density,
        float(veg_chm_valid.mean()) if veg_chm_valid.size else 0.0,
        veg_chm_stats["mean"],
        veg_chm_stats["std"],
        veg_chm_stats["p50"],
        veg_chm_stats["p75"],
        veg_chm_stats["p90"],
        veg_chm_stats["p95"],
        veg_chm_stats["max"],
        center_veg_chm_stats["mean"],
        center_veg_chm_stats["p95"],
        veg_chm_texture_contrast,
        veg_chm_texture_edge_density,
        float((valid_veg_chm > 2.0).mean()) if valid_veg_chm.size else 0.0,
        float((valid_veg_chm > 5.0).mean()) if valid_veg_chm.size else 0.0,
        float((valid_veg_chm > 10.0).mean()) if valid_veg_chm.size else 0.0,
        float((valid_veg_chm > 15.0).mean()) if valid_veg_chm.size else 0.0,
        float(valid_veg_chm.sum() / pixel_count) if valid_veg_chm.size else 0.0,
        float(veg_chm_stats["mean"] * veg_area_fraction),
        float(veg_chm_stats["p95"] * veg_area_fraction),
        float(tree_like.mean()) if tree_like.size else 0.0,
        float(low_veg.mean()) if low_veg.size else 0.0,
        float(tall_nonveg.mean()) if tall_nonveg.size else 0.0,
        float(center_tree_like.mean()) if center_tree_like.size else 0.0,
        float(center_low_veg.mean()) if center_low_veg.size else 0.0,
        float(center_tall_nonveg.mean()) if center_tall_nonveg.size else 0.0,
        tree_ndvi_stats["mean"],
        tree_ndvi_stats["std"],
        tree_chm_stats["mean"],
        tree_chm_stats["p95"],
        tree_chm_stats["max"],
        float(tree_chm.sum() / pixel_count) if tree_chm.size else 0.0,
        float(tree_ndvi_stats["mean"] * tree_chm_stats["p95"]),
        float(tree_chm_stats["p95"] - tree_chm_stats["p50"]),
        float(tree_chm_stats["p95"] - tree_chm_stats["p10"]),
        safe_ratio(tree_chm_stats["std"], tree_chm_stats["mean"]),
        float(veg_chm_stats["p95"] - veg_chm_stats["p50"]),
        float(veg_chm_stats["p95"] - veg_chm_stats["p10"]),
        safe_ratio(veg_chm_stats["std"], veg_chm_stats["mean"]),
        float(tree_like_gt_5m.mean()) if tree_like_gt_5m.size else 0.0,
        float(tree_like_gt_10m.mean()) if tree_like_gt_10m.size else 0.0,
        float(center_tree_like_gt_5m.mean()) if center_tree_like_gt_5m.size else 0.0,
        float(center_tree_like_gt_10m.mean()) if center_tree_like_gt_10m.size else 0.0,
        safe_ratio(tree_area_fraction, veg_area_on_chm_fraction),
        safe_ratio(float(low_veg.mean()) if low_veg.size else 0.0, veg_area_on_chm_fraction),
        safe_ratio(tree_area_fraction, tall_area_fraction),
        safe_ratio(float(tall_nonveg.mean()) if tall_nonveg.size else 0.0, tall_area_fraction),
        float(center_tree_area_fraction - ring_tree_area_fraction),
        float(center_low_veg_fraction - ring_low_veg_fraction),
        float(center_tall_nonveg_fraction - ring_tall_nonveg_fraction),
        float(center_ndvi_stats["mean"] - ring_ndvi_stats["mean"]),
        float(center_veg_chm_stats["mean"] - ring_veg_chm_stats["mean"]),
        float(center_veg_chm_stats["p95"] - ring_veg_chm_stats["p95"]),
        ring_tree_area_fraction,
        ring_veg_chm_stats["mean"],
        ring_veg_chm_stats["p95"],
        safe_ratio(veg_chm_texture_edge_density, veg_area_fraction),
        safe_ratio(ndvi_texture_edge_density, float((valid_ndvi > vegetation_ndvi_threshold).mean()) if valid_ndvi.size else 0.0),
        safe_ratio(brightness_texture_edge_density, float((valid_ndvi > vegetation_ndvi_threshold).mean()) if valid_ndvi.size else 0.0),
        tree_blue_mean,
        tree_green_mean,
        tree_red_mean,
        tree_nir_mean,
        tree_brightness_mean,
        float(tree_green_mean - tree_red_mean),
        float(tree_nir_mean - tree_red_mean),
        tree_ndvi_stats["iqr"],
        tree_brightness_std,
    ]
    out = np.asarray(values, dtype=np.float32)
    if out.shape[0] != len(NAIP_CHM_STRUCTURE_COLUMNS):
        raise RuntimeError(f"Structure feature length mismatch: {out.shape[0]} vs {len(NAIP_CHM_STRUCTURE_COLUMNS)}")
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)



