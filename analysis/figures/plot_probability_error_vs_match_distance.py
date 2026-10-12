#!/usr/bin/env python3
"""Prepare and plot independent-test uncertainty across crown-match distance."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, LogNorm
import numpy as np
import pandas as pd

from figure_config import canonical_collapse_zero_based
from figure_data import save_figure_formats


ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--predictions", type=Path,
        default=ROOT / "outputs" / "reviewer_revision" / "final_pruned_k11_test_top5_ensemble" / "ensemble_predictions.npz",
    )
    p.add_argument(
        "--shard-root", type=Path, required=True,
        help="Complete shard root, typically below $DATA_ROOT.",
    )
    p.add_argument(
        "--output-dir", type=Path,
        default=ROOT / "outputs" / "reviewer_revision" / "final_pruned_k11_test_match_distance_probability_error",
    )
    p.add_argument("--distance-bin-width", type=float, default=0.25)
    p.add_argument("--hexbin-gridsize", type=int, default=50)
    p.add_argument(
        "--group-space", choices=("k11", "collapsed_k6"), default="k11",
        help="Reporting label space; collapsed_k6 uses config/final_grouping.json.",
    )
    return p.parse_args()


def metadata_for_predictions(pred: np.lib.npyio.NpzFile, shard_root: Path) -> pd.DataFrame:
    wanted = pd.DataFrame(
        {
            "city_token": pred["city_token"].astype(str),
            "tree_id": pred["tree_id"].astype(np.int64),
            "crop_index": pred["crop_index"].astype(np.int64),
            "row_order": np.arange(len(pred["labels"]), dtype=np.int64),
        }
    )
    pieces = []
    wanted_cities = set(wanted.city_token)
    for path in sorted(shard_root.glob("*/*_metadata.csv")):
        city = path.name.split("_part", 1)[0].lower()
        if city not in wanted_cities:
            continue
        frame = pd.read_csv(
            path,
            usecols=lambda column: column in {"tree_id", "crop_index", "match_distance_m", "nearest_crown_distance_m"},
            low_memory=False,
        )
        distance_column = "match_distance_m" if "match_distance_m" in frame.columns else "nearest_crown_distance_m"
        frame = frame.rename(columns={distance_column: "match_distance_m"})
        frame["city_token"] = city
        pieces.append(frame[["city_token", "tree_id", "crop_index", "match_distance_m"]])
    if not pieces:
        raise RuntimeError(f"No metadata found under {shard_root}")
    lookup = pd.concat(pieces, ignore_index=True).drop_duplicates(
        ["city_token", "tree_id", "crop_index"], keep="first"
    )
    joined = wanted.merge(
        lookup, on=["city_token", "tree_id", "crop_index"], how="left", validate="one_to_one"
    ).sort_values("row_order")
    missing = int(joined.match_distance_m.isna().sum())
    if missing:
        raise RuntimeError(f"Missing crown-match distance for {missing:,} prediction records")
    return joined.reset_index(drop=True)


def correlation(x: np.ndarray, y: np.ndarray, ranked: bool = False) -> tuple[float, float]:
    if ranked:
        x = pd.Series(x).rank(method="average").to_numpy(dtype=float)
        y = pd.Series(y).rank(method="average").to_numpy(dtype=float)
    x = x - x.mean()
    y = y - y.mean()
    value = float(np.sum(x * y) / np.sqrt(np.sum(x * x) * np.sum(y * y)))
    t_value = value * math.sqrt((len(x) - 2) / max(1.0 - value * value, 1e-15))
    # With >160,000 observations, the normal approximation is indistinguishable
    # from the Student-t tail at reportable precision.
    p_value = float(math.erfc(abs(t_value) / math.sqrt(2.0)))
    return value, p_value


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with np.load(args.predictions, allow_pickle=True) as pred:
        labels = pred["labels"].astype(np.int64)
        probabilities = pred["probabilities"].astype(np.float64)
        metadata = metadata_for_predictions(pred, args.shard_root)
    if len(labels) != len(metadata):
        raise RuntimeError("Prediction and metadata row counts differ")
    reporting_k = probabilities.shape[1]
    collapse_map = canonical_collapse_zero_based() if args.group_space == "collapsed_k6" else None
    if collapse_map is not None:
        if len(collapse_map) != probabilities.shape[1]:
            raise ValueError(
                f"Collapse map has {len(collapse_map)} entries for "
                f"{probabilities.shape[1]} source classes"
            )
        reporting_k = int(collapse_map.max()) + 1
        if collapse_map.min() < 0 or set(collapse_map.tolist()) != set(range(reporting_k)):
            raise ValueError("Collapsed groups must be contiguous one-based identifiers")
        collapsed_probabilities = np.zeros((len(probabilities), reporting_k), dtype=np.float64)
        for source_group, collapsed_group in enumerate(collapse_map):
            collapsed_probabilities[:, collapsed_group] += probabilities[:, source_group]
        labels = collapse_map[labels]
        probabilities = collapsed_probabilities
    true_probability = probabilities[np.arange(len(labels)), labels]
    probability_error = 1.0 - true_probability
    distance = metadata.match_distance_m.to_numpy(dtype=float)
    valid = np.isfinite(distance) & np.isfinite(probability_error)
    distance, probability_error = distance[valid], probability_error[valid]

    pearson_r, pearson_p = correlation(distance, probability_error)
    spearman_rho, spearman_p = correlation(distance, probability_error, ranked=True)
    edges = np.arange(0.0, max(5.0, float(np.ceil(distance.max() / args.distance_bin_width) * args.distance_bin_width)) + args.distance_bin_width, args.distance_bin_width)
    bins = pd.cut(distance, bins=edges, include_lowest=True, right=False)
    binned = (
        pd.DataFrame({"distance": distance, "probability_error": probability_error, "bin": bins})
        .groupby("bin", observed=True)
        .agg(
            records=("probability_error", "size"),
            distance_median=("distance", "median"),
            error_mean=("probability_error", "mean"),
            error_median=("probability_error", "median"),
            error_q25=("probability_error", lambda x: x.quantile(0.25)),
            error_q75=("probability_error", lambda x: x.quantile(0.75)),
        )
        .reset_index(drop=True)
    )
    binned.to_csv(args.output_dir / "match_distance_probability_error_binned.csv", index=False)

    summary = {
        "records": int(len(distance)),
        "reporting_groups": reporting_k,
        "source_groups": int(len(collapse_map)) if collapse_map is not None else reporting_k,
        "label_space": args.group_space,
        "collapse_map_zero_based": collapse_map.tolist() if collapse_map is not None else None,
        "probability_error_definition": (
            f"1 - summed ensemble probability assigned to the actual collapsed k={reporting_k} class"
            if collapse_map is not None
            else f"1 - ensemble probability assigned to the actual k={reporting_k} class"
        ),
        "distance_median_m": float(np.median(distance)),
        "distance_q25_m": float(np.quantile(distance, 0.25)),
        "distance_q75_m": float(np.quantile(distance, 0.75)),
        "probability_error_mean": float(probability_error.mean()),
        "probability_error_median": float(np.median(probability_error)),
        "pearson_r": pearson_r,
        "pearson_p": pearson_p,
        "spearman_rho": spearman_rho,
        "spearman_p": spearman_p,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    cmap = LinearSegmentedColormap.from_list(
        "tree_density", ["#F7FBFF", "#C6DBEF", "#6BAED6", "#2171B5", "#08306B"]
    )
    fig, ax = plt.subplots(figsize=(11.2, 8.2), dpi=220)
    ax.set_facecolor("#F7F9FA")
    hb = ax.hexbin(
        distance, probability_error, gridsize=args.hexbin_gridsize,
        extent=(0.0, max(5.0, float(distance.max())), 0.0, 1.0),
        mincnt=1, cmap=cmap, norm=LogNorm(), linewidths=0,
    )
    ax.plot(
        binned.distance_median, binned.error_median,
        color="white", linewidth=4.8, alpha=0.95, zorder=3,
    )
    ax.plot(
        binned.distance_median, binned.error_median,
        color="#B2182B", linewidth=2.3, marker="o", markersize=3.4,
        label="Median within 0.25 m bins", zorder=4,
    )
    ax.set_xlim(0.0, max(5.0, float(distance.max())))
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("Tree-center to matched-crown distance (m)", fontsize=14)
    ax.set_ylabel(r"True-class probability error ($1-p_{true}$)", fontsize=14)
    ax.yaxis.set_major_formatter(lambda value, position: f"{value:.0%}")
    ax.grid(True, color="#D9E0E5", linewidth=0.75, alpha=0.55)
    ax.tick_params(labelsize=11.5, colors="#354354")
    ax.set_title(
        "Classification uncertainty across crown-matching distance",
        loc="left", fontsize=18, fontweight="bold", color="#202B38", pad=36,
    )
    ax.text(
        0, 1.025,
        f"Final pruned k=11 test ensemble, reported at k={reporting_k} · n = {len(distance):,} · "
        f"Spearman ρ = {spearman_rho:.3f}",
        transform=ax.transAxes, fontsize=11.8, color="#607184", va="bottom",
    )
    ax.legend(loc="upper left", frameon=True, facecolor="white", edgecolor="#B7C1C9", fontsize=10.5)
    cbar = fig.colorbar(hb, ax=ax, pad=0.015, fraction=0.042)
    cbar.set_label("Test records per hexagon (log scale)", fontsize=11)
    cbar.ax.tick_params(labelsize=10)
    fig.subplots_adjust(left=0.11, right=0.88, bottom=0.11, top=0.86)
    save_figure_formats(
        fig,
        args.output_dir / f"test_{args.group_space}_probability_error_vs_match_distance",
        dpi=600,
    )
    plt.close(fig)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
