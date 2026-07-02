"""Analyze whether nearby different-type trees are linked to prediction errors.

This script reads a tree-level prediction CSV from randomForestTreeTypes.py and
flags each tree that has at least one different true tree_type within a given
distance. It then compares model performance for trees with and without nearby
different-type neighbors.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_PREDICTIONS = Path("C:/Users/larki/Desktop/PollenSense/training/TreeTypeRandomForest/test_tree_predictions.csv")
DEFAULT_OUT_DIR = Path("C:/Users/larki/Desktop/PollenSense/training/TreeTypeRandomForest/neighbor_diagnostics")
EARTH_RADIUS_M = 6_371_008.8
REQUIRED_COLUMNS = ["latitude", "longitude", "tree_type", "predicted_tree_type", "city"]


def local_xy_m(lat_lon: np.ndarray) -> np.ndarray:
    lat0 = np.deg2rad(lat_lon[:, 0].mean())
    x = np.deg2rad(lat_lon[:, 1]) * np.cos(lat0) * EARTH_RADIUS_M
    y = np.deg2rad(lat_lon[:, 0]) * EARTH_RADIUS_M
    return np.column_stack([x, y])


def grid_neighbors_within_radius(coords_m: np.ndarray, radius_m: float) -> list[tuple[np.ndarray, np.ndarray]]:
    cell_size = radius_m
    cells = np.floor(coords_m / cell_size).astype(np.int64)
    cell_index: dict[tuple[int, int], list[int]] = {}
    for i, cell in enumerate(cells):
        cell_index.setdefault((int(cell[0]), int(cell[1])), []).append(i)

    out = []
    for i, cell in enumerate(cells):
        candidates = []
        cx, cy = int(cell[0]), int(cell[1])
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                candidates.extend(cell_index.get((cx + dx, cy + dy), []))

        if not candidates:
            out.append((np.array([], dtype=int), np.array([], dtype=float)))
            continue

        candidate_idx = np.array(candidates, dtype=int)
        dist_m = np.linalg.norm(coords_m[candidate_idx] - coords_m[i], axis=1)
        keep = dist_m <= radius_m
        candidate_idx = candidate_idx[keep]
        dist_m = dist_m[keep]
        order = np.argsort(dist_m)
        out.append((candidate_idx[order], dist_m[order]))

    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare tree-type prediction errors for trees near different-type neighbors."
    )
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--radius-m", type=float, default=10.0)
    parser.add_argument("--group-cols", nargs="+", default=["city"])
    parser.add_argument("--min-group-size", type=int, default=2)
    return parser.parse_args()


def validate_columns(df: pd.DataFrame, group_cols: list[str]) -> None:
    missing = [c for c in REQUIRED_COLUMNS + group_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Prediction file is missing required columns: {missing}")


def add_neighbor_flags(df: pd.DataFrame, radius_m: float, group_cols: list[str], min_group_size: int) -> pd.DataFrame:
    work = df.copy()
    work["latitude"] = pd.to_numeric(work["latitude"], errors="coerce")
    work["longitude"] = pd.to_numeric(work["longitude"], errors="coerce")
    work = work.dropna(subset=["latitude", "longitude", "tree_type", "predicted_tree_type"]).reset_index(drop=True)

    work["has_neighbor_within_radius"] = False
    work["has_different_type_neighbor_within_radius"] = False
    work["nearest_neighbor_m"] = np.nan
    work["nearest_different_type_neighbor_m"] = np.nan
    work["neighbor_count_within_radius"] = 0
    work["different_type_neighbor_count_within_radius"] = 0

    grouped = work.groupby(group_cols, sort=False, dropna=False)

    for _, group in grouped:
        if len(group) < min_group_size:
            continue

        idx = group.index.to_numpy()
        lat_lon = group[["latitude", "longitude"]].to_numpy(dtype=float)
        coords_m = local_xy_m(lat_lon)
        neighbors = grid_neighbors_within_radius(coords_m, radius_m)
        true_types = group["tree_type"].astype(str).to_numpy()

        for local_i, (neighbor_idx, neighbor_dist_m) in enumerate(neighbors):
            keep = neighbor_idx != local_i
            neighbor_idx = neighbor_idx[keep]
            neighbor_dist_m = neighbor_dist_m[keep]
            if len(neighbor_idx) == 0:
                continue

            different = true_types[neighbor_idx] != true_types[local_i]
            global_i = idx[local_i]
            work.at[global_i, "has_neighbor_within_radius"] = True
            work.at[global_i, "neighbor_count_within_radius"] = int(len(neighbor_idx))
            work.at[global_i, "nearest_neighbor_m"] = float(neighbor_dist_m.min())

            if different.any():
                different_dist = neighbor_dist_m[different]
                work.at[global_i, "has_different_type_neighbor_within_radius"] = True
                work.at[global_i, "different_type_neighbor_count_within_radius"] = int(different.sum())
                work.at[global_i, "nearest_different_type_neighbor_m"] = float(different_dist.min())

    return work


def summarize_groups(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["correct"] = df["tree_type"].astype(str) == df["predicted_tree_type"].astype(str)
    rows = []
    for label, group in [
        ("different_type_neighbor_within_radius", df[df["has_different_type_neighbor_within_radius"]]),
        ("no_different_type_neighbor_within_radius", df[~df["has_different_type_neighbor_within_radius"]]),
        ("any_neighbor_within_radius", df[df["has_neighbor_within_radius"]]),
        ("no_neighbor_within_radius", df[~df["has_neighbor_within_radius"]]),
        ("all", df),
    ]:
        if group.empty:
            rows.append({"group": label, "n": 0, "accuracy": np.nan, "error_rate": np.nan})
            continue
        rows.append(
            {
                "group": label,
                "n": int(len(group)),
                "accuracy": float(group["correct"].mean()),
                "error_rate": float(1.0 - group["correct"].mean()),
                "mean_neighbor_count": float(group["neighbor_count_within_radius"].mean()),
                "mean_different_type_neighbor_count": float(group["different_type_neighbor_count_within_radius"].mean()),
                "median_nearest_neighbor_m": float(group["nearest_neighbor_m"].median(skipna=True)),
                "median_nearest_different_type_neighbor_m": float(
                    group["nearest_different_type_neighbor_m"].median(skipna=True)
                ),
            }
        )
    return pd.DataFrame(rows)


def save_class_reports(df: pd.DataFrame, out_dir: Path) -> None:
    for name, group in [
        ("different_type_neighbor", df[df["has_different_type_neighbor_within_radius"]]),
        ("no_different_type_neighbor", df[~df["has_different_type_neighbor_within_radius"]]),
    ]:
        if group.empty:
            continue
        labels = sorted(set(group["tree_type"].astype(str)) | set(group["predicted_tree_type"].astype(str)))
        rows = []
        y_true = group["tree_type"].astype(str)
        y_pred = group["predicted_tree_type"].astype(str)
        for label in labels:
            tp = int(((y_true == label) & (y_pred == label)).sum())
            fp = int(((y_true != label) & (y_pred == label)).sum())
            fn = int(((y_true == label) & (y_pred != label)).sum())
            support = int((y_true == label).sum())
            precision = tp / (tp + fp) if (tp + fp) else 0.0
            recall = tp / support if support else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
            rows.append(
                {
                    "label": label,
                    "precision": precision,
                    "recall": recall,
                    "f1_score": f1,
                    "support": support,
                }
            )
        pd.DataFrame(rows).to_csv(out_dir / f"{name}_classification_report.csv", index=False)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.predictions, low_memory=False)
    validate_columns(df, args.group_cols)
    flagged = add_neighbor_flags(
        df=df,
        radius_m=args.radius_m,
        group_cols=args.group_cols,
        min_group_size=args.min_group_size,
    )

    summary = summarize_groups(flagged)
    flagged.to_csv(args.out_dir / f"neighbor_flags_{int(args.radius_m)}m.csv", index=False)
    summary.to_csv(args.out_dir / f"neighbor_error_summary_{int(args.radius_m)}m.csv", index=False)
    save_class_reports(flagged, args.out_dir)

    print(summary)
    print(f"\nSaved diagnostics to: {args.out_dir}")


if __name__ == "__main__":
    main()
