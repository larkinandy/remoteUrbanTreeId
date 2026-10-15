#!/usr/bin/env python3
"""Plot collapsed-k6 performance and support for all sensitivity-analysis strata."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from figure_data import save_figure_formats


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = ROOT / "analysis" / "figures" / "provenance" / "collapsed_k6_sensitivity_strata.csv"
DEFAULT_OUTPUT = ROOT / "outputs" / "manuscript_figures" / "collapsed_k6_sensitivity_strata_performance"

ANALYSIS_ORDER = (
    "Reference",
    "Vegetation height",
    "Taxon provenance",
    "CHM tall vegetation within Sentinel cell",
    "LiDAR year mismatch",
    "Within-cell NAIP vegetation fraction",
    "Crown-match distance",
    "Sentinel-2 interpolation",
    "Duplicate coordinates",
    "NAIP year mismatch",
)

ROW_LABELS = {
    ("Reference", "All test records"): "Reference: all records",
    ("Vegetation height", "Height >=2 m"): "Height ≥2 m",
    ("Vegetation height", "Height <2 m"): "Height <2 m",
    ("Taxon provenance", "Core taxa"): "Core taxa",
    ("Taxon provenance", "Rare taxa"): "Rare taxa",
    ("CHM tall vegetation within Sentinel cell", ">=25%"): "CHM tall cover ≥25%",
    ("CHM tall vegetation within Sentinel cell", "<25%"): "CHM tall cover <25%",
    ("LiDAR year mismatch", ">=5 years"): "LiDAR mismatch ≥5 y",
    ("LiDAR year mismatch", "<5 years"): "LiDAR mismatch <5 y",
    ("Within-cell NAIP vegetation fraction", ">=25%"): "NAIP vegetation ≥25%",
    ("Within-cell NAIP vegetation fraction", "<25%"): "NAIP vegetation <25%",
    ("Crown-match distance", ">=2.5 m"): "Match distance ≥2.5 m",
    ("Crown-match distance", "<2.5 m"): "Match distance <2.5 m",
    ("Sentinel-2 interpolation", ">=25% interpolated days"): "Interpolation ≥25%",
    ("Sentinel-2 interpolation", "<25% interpolated days"): "Interpolation <25%",
    ("Duplicate coordinates", ">=5% duplicate records"): "Duplicate records ≥5%",
    ("Duplicate coordinates", "<5% duplicate records"): "Duplicate records <5%",
    ("NAIP year mismatch", ">=2 years"): "NAIP mismatch ≥2 y",
    ("NAIP year mismatch", "<2 years"): "NAIP mismatch <2 y",
}

METRICS = (
    ("accuracy", "Accuracy", "o", "#202b35", 0.13),
    ("macro_f1", "Macro-F1", "D", "white", 0.00),
    ("weighted_f1", "Weighted-F1", "s", "#89949d", -0.13),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Frozen sensitivity-strata table.")
    parser.add_argument("--output-prefix", type=Path, default=DEFAULT_OUTPUT, help="Output path without extension.")
    parser.add_argument("--dpi", type=int, default=600, help="PNG resolution.")
    return parser.parse_args()


def ordered_rows(data: pd.DataFrame) -> pd.DataFrame:
    analysis_rank = {name: rank for rank, name in enumerate(ANALYSIS_ORDER)}
    result = data.copy()
    result["analysis_rank"] = result["analysis"].map(analysis_rank)
    if result["analysis_rank"].isna().any():
        missing = result.loc[result["analysis_rank"].isna(), "analysis"].unique()
        raise ValueError(f"Unexpected analyses: {missing.tolist()}")
    result = result.sort_values(["analysis_rank", "stratum_order"]).reset_index(drop=True)
    result["display_label"] = [ROW_LABELS[(a, c)] for a, c in zip(result["analysis"], result["criterion"])]
    result["max_delta"] = np.nan
    result.loc[result["analysis"].eq("Reference"), "max_delta"] = 0.0
    metric_names = [metric for metric, _, _, _, _ in METRICS]
    for analysis in ANALYSIS_ORDER[1:]:
        indices = result.index[result["analysis"].eq(analysis)].tolist()
        if len(indices) != 2:
            raise ValueError(f"Expected two strata for {analysis!r}; found {len(indices)}")
        first, second = result.loc[indices[0]], result.loc[indices[1]]
        delta = max(abs(float(first[metric]) - float(second[metric])) for metric in metric_names)
        result.loc[indices[0], "max_delta"] = delta
    return result


def shade_pairs(axes: tuple[plt.Axes, ...], data: pd.DataFrame, positions: np.ndarray) -> None:
    for group_number, analysis in enumerate(ANALYSIS_ORDER[1:], start=1):
        indices = np.flatnonzero(data["analysis"].eq(analysis).to_numpy())
        if len(indices) != 2:
            continue
        ys = positions[indices]
        if group_number % 2:
            for ax in axes:
                ax.axhspan(ys.min() - 0.48, ys.max() + 0.48, color="#f2f4f5", zorder=-2)


def main() -> int:
    args = parse_args()
    data = ordered_rows(pd.read_csv(args.input))
    positions = np.arange(len(data))[::-1]

    plt.rcParams.update({
        "font.family": "Arial",
        "font.size": 8.5,
        "axes.labelsize": 9.5,
        "axes.titlesize": 10,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.2,
        "legend.fontsize": 7.5,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
    })

    fig, (performance_ax, support_ax) = plt.subplots(
        1, 2, figsize=(9.2, 7.0), sharey=True,
        gridspec_kw={"width_ratios": [1.35, 0.65], "wspace": 0.05},
    )
    shade_pairs((performance_ax, support_ax), data, positions)

    for y, (_, row) in zip(positions, data.iterrows()):
        values = row[[metric for metric, _, _, _, _ in METRICS]].astype(float).to_numpy()
        performance_ax.hlines(y, values.min(), values.max(), color="#aeb6bc", linewidth=1.35, zorder=1)

    for metric, label, marker, facecolor, offset in METRICS:
        performance_ax.scatter(
            data[metric], positions + offset, marker=marker, s=31,
            facecolor=facecolor, edgecolor="#202b35", linewidth=0.85,
            label=label, zorder=3,
        )

    performance_ax.set_yticks(positions)
    performance_ax.set_yticklabels(data["display_label"])
    performance_ax.set_xlim(0.58, 0.83)
    performance_ax.set_xticks(np.arange(0.60, 0.81, 0.05))
    performance_ax.set_xlabel("Independent-test performance")
    performance_ax.set_title("a   Performance by data stratum", loc="left", fontweight="bold", pad=8)
    performance_ax.grid(axis="x", color="#dde2e6", linewidth=0.7)
    performance_ax.tick_params(axis="y", length=0, pad=6)
    performance_ax.legend(loc="lower left", frameon=False, ncol=3, columnspacing=1.0, handletextpad=0.4)

    support_ax.set_xlim(0, 1)
    support_ax.set_xticks([])
    support_ax.set_xlabel("")
    support_ax.set_title("b   Support and maximum difference", loc="left", fontweight="bold", pad=24)
    support_ax.tick_params(axis="y", left=False, labelleft=False)
    support_ax.text(0.63, 1.0, "Records (n)", transform=support_ax.transAxes, ha="right", va="top", fontsize=7.5, color="#5f6d79")
    support_ax.text(0.96, 1.0, "Max Δ", transform=support_ax.transAxes, ha="right", va="top", fontsize=7.5, color="#5f6d79")
    for y, (_, row) in zip(positions, data.iterrows()):
        support_ax.text(0.63, y, f"{int(row['records']):,}", va="center", ha="right", fontsize=8.3, color="#34424f")
        if pd.notna(row["max_delta"]):
            support_ax.text(0.96, y, f"{float(row['max_delta']):.3f}", va="center", ha="right", fontsize=8.3, color="#34424f")

    for ax in (performance_ax, support_ax):
        ax.grid(axis="y", visible=False)
        for spine in ("top", "right", "left"):
            ax.spines[spine].set_visible(False)
        ax.spines["bottom"].set_color("#77838d")
    support_ax.spines["bottom"].set_visible(False)

    fig.text(
        0.99, 0.02, "Collapsed k = 6; sensitivity strata evaluated post hoc without retraining",
        ha="right", va="bottom", fontsize=7.4, color="#5f6d79",
    )
    fig.subplots_adjust(left=0.285, right=0.965, top=0.945, bottom=0.12)

    for path in save_figure_formats(fig, args.output_prefix, args.dpi):
        print(path)
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
