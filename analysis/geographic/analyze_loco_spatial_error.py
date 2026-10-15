#!/usr/bin/env python3
"""Measure spatial autocorrelation of binary collapsed-k6 LOCO errors."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


NEIGHBOR_COUNTS = (4, 8, 16)
PERMUTATIONS = 999
RANDOM_SEED = 13


def parse_args() -> argparse.Namespace:
    """Parse the GIS prediction input and output directory."""
    parser = argparse.ArgumentParser(
        description="Calculate overall and labeled-group Moran's I for LOCO errors."
    )
    parser.add_argument("--gis-predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--city-label", required=True)
    return parser.parse_args()


def morans_i(
    coordinates: np.ndarray,
    values: np.ndarray,
    neighbors: int,
    permutations: int,
    rng: np.random.Generator,
) -> dict[str, float]:
    """Calculate row-standardized kNN Moran's I and a permutation test."""
    if len(values) <= neighbors:
        raise ValueError(f"Need more than {neighbors} observations")
    distances, indices = cKDTree(coordinates).query(coordinates, k=neighbors + 1)
    neighbor_indices = indices[:, 1:]
    centered = values.astype(float) - values.mean()
    denominator = np.dot(centered, centered)
    if denominator == 0:
        raise ValueError("Moran's I is undefined for a constant response")

    def statistic(vector: np.ndarray) -> float:
        return float(
            (vector[:, None] * vector[neighbor_indices]).sum()
            / neighbors
            / np.dot(vector, vector)
        )

    observed = statistic(centered)
    null = np.empty(permutations, dtype=float)
    for index in range(permutations):
        null[index] = statistic(rng.permutation(centered))
    null_mean = float(null.mean())
    null_sd = float(null.std(ddof=1))
    p_value = float(
        (1 + np.count_nonzero(np.abs(null - null_mean) >= abs(observed - null_mean)))
        / (permutations + 1)
    )
    return {
        "mean_neighbor_distance_m": float(distances[:, 1:].mean()),
        "morans_i": observed,
        "permutation_null_mean": null_mean,
        "permutation_z_score": (observed - null_mean) / null_sd,
        "permutation_p_two_sided": p_value,
    }


def main() -> int:
    """Analyze overall errors and each labeled collapsed-k6 group."""
    args = parse_args()
    frame = pd.read_csv(args.gis_predictions)
    required = {
        "tree_x",
        "tree_y",
        "labeled_k6_group",
        "predicted_k6_group",
        "k6_match",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    frame["prediction_error"] = 1 - frame["k6_match"].astype(int)

    rng = np.random.default_rng(RANDOM_SEED)
    rows: list[dict[str, object]] = []
    subsets = [("All groups", frame)] + [
        (f"Group {group}", frame.loc[frame["labeled_k6_group"].eq(group)])
        for group in range(1, 7)
    ]
    for label, subset in subsets:
        coordinates = subset[["tree_x", "tree_y"]].to_numpy(dtype=float)
        errors = subset["prediction_error"].to_numpy(dtype=float)
        for neighbors in NEIGHBOR_COUNTS:
            result: dict[str, object] = {
                "city": args.city_label,
                "labeled_group": label,
                "number_of_test_records": len(subset),
                "number_of_errors": int(errors.sum()),
                "error_rate": float(errors.mean()),
                "neighbors_k": neighbors,
                "permutations": PERMUTATIONS,
            }
            try:
                result.update(
                    morans_i(coordinates, errors, neighbors, PERMUTATIONS, rng)
                )
                result["status"] = "estimated"
            except ValueError as error:
                result.update(
                    {
                        "mean_neighbor_distance_m": np.nan,
                        "morans_i": np.nan,
                        "permutation_null_mean": np.nan,
                        "permutation_z_score": np.nan,
                        "permutation_p_two_sided": np.nan,
                        "status": str(error),
                    }
                )
            rows.append(result)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "rancho_cucamonga_loco_spatial_error_morans_i.csv"
    pd.DataFrame(rows).to_csv(output, index=False)
    print(output)
    print(pd.DataFrame(rows).loc[lambda data: data["neighbors_k"].eq(8)].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
