#!/usr/bin/env python3
"""Plot signed performance differences between paired sensitivity-analysis strata."""

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
DEFAULT_OUTPUT = ROOT / "outputs" / "manuscript_figures" / "collapsed_k6_sensitivity_strata_differences"

ORDER = (
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

LABELS = {
    "Vegetation height": "Height: ≥2 m − <2 m",
    "Taxon provenance": "Taxa: core − rare",
    "CHM tall vegetation within Sentinel cell": "CHM tall cover: ≥25% − <25%",
    "LiDAR year mismatch": "LiDAR mismatch: ≥5 y − <5 y",
    "Within-cell NAIP vegetation fraction": "NAIP vegetation: ≥25% − <25%",
    "Crown-match distance": "Match distance: ≥2.5 m − <2.5 m",
    "Sentinel-2 interpolation": "Interpolation: ≥25% − <25%",
    "Duplicate coordinates": "Duplicates: ≥5% − <5%",
    "NAIP year mismatch": "NAIP mismatch: ≥2 y − <2 y",
}

METRICS = (
    ("accuracy", "Accuracy", "o", "#202b35", 0.10),
    ("macro_f1", "Macro-F1", "D", "white", 0.00),
    ("weighted_f1", "Weighted-F1", "s", "#89949d", -0.10),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Frozen paired-strata performance table.")
    parser.add_argument("--output-prefix", type=Path, default=DEFAULT_OUTPUT, help="Output path without extension.")
    parser.add_argument("--dpi", type=int, default=600, help="PNG resolution.")
    return parser.parse_args()


def paired_differences(data: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    for analysis in ORDER:
        pair = data.loc[data["analysis"].eq(analysis)].sort_values("stratum_order")
        if len(pair) != 2:
            raise ValueError(f"Expected two ordered strata for {analysis!r}; found {len(pair)}")
        first, second = pair.iloc[0], pair.iloc[1]
        row: dict[str, float | str] = {"analysis": analysis}
        for metric, _, _, _, _ in METRICS:
            row[metric] = 100.0 * (float(first[metric]) - float(second[metric]))
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> int:
    args = parse_args()
    differences = paired_differences(pd.read_csv(args.input))

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

    fig, ax = plt.subplots(figsize=(5.55, 4.9))
    positions = np.arange(len(differences))[::-1]

    for y, (_, row) in zip(positions, differences.iterrows()):
        values = row[[metric for metric, _, _, _, _ in METRICS]].astype(float).to_numpy()
        ax.hlines(y, values.min(), values.max(), color="#aeb6bc", linewidth=1.5, zorder=1)

    for metric, label, marker, facecolor, offset in METRICS:
        ax.scatter(
            differences[metric], positions + offset, marker=marker, s=34,
            facecolor=facecolor, edgecolor="#202b35", linewidth=0.9,
            label=label, zorder=3,
        )

    ax.axvline(0, color="#333333", linewidth=0.9, linestyle=(0, (3, 3)), zorder=0)
    ax.set_yticks(positions)
    ax.set_yticklabels([LABELS[value] for value in differences["analysis"]])
    ax.set_xlim(-8.5, 18.5)
    ax.set_xticks(np.arange(-5, 16, 5))
    ax.set_xlabel("First stratum minus second stratum (percentage points)")
    ax.set_title("d   Sensitivity to data strata", loc="left", fontweight="bold", pad=8)
    ax.grid(axis="x", color="#dde2e6", linewidth=0.7)
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", length=0, pad=5)
    ax.legend(loc="lower right", frameon=False, ncol=1, handletextpad=0.5)

    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color("#77838d")
    ax.text(
        1.0, -0.16, "Collapsed k = 6; independent test set",
        transform=ax.transAxes, ha="right", va="top", fontsize=7.2, color="#5f6d79",
    )
    fig.subplots_adjust(left=0.47, right=0.985, top=0.91, bottom=0.22)

    for path in save_figure_formats(fig, args.output_prefix, args.dpi):
        print(path)
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
