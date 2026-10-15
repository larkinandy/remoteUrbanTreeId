#!/usr/bin/env python3
"""Plot target-city performance distributions across geographic training regimes."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

from figure_data import save_figure_formats


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = ROOT / "analysis" / "geographic" / "provenance" / "final_training_regime_by_target_city.csv"
DEFAULT_OUTPUT = ROOT / "outputs" / "manuscript_figures" / "geographic_training_regime_boxplots"

METRICS = (("accuracy", "Accuracy"), ("macro_f1", "Macro-F1"), ("weighted_f1", "Weighted-F1"))
REGIMES = (
    ("Default multimodal", "Default model", "#d8dde1", "o"),
    ("Leave one city out", "Leave one city out", "#9da8b0", "s"),
    ("One-city trained", "One-city trained", "white", "D"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Frozen target-city performance table.")
    parser.add_argument("--output-prefix", type=Path, default=DEFAULT_OUTPUT, help="Output path without extension.")
    parser.add_argument("--dpi", type=int, default=600, help="PNG resolution.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data = pd.read_csv(args.input)

    plt.rcParams.update({
        "font.family": "Arial",
        "font.size": 8.5,
        "axes.labelsize": 9.5,
        "axes.titlesize": 10,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "legend.fontsize": 7.5,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
    })

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    centers = np.arange(len(METRICS))
    offsets = (-0.25, 0.0, 0.25)
    rng = np.random.default_rng(2026)

    for offset, (regime, _, facecolor, marker) in zip(offsets, REGIMES):
        values = [data.loc[data["model"].eq(regime), metric].to_numpy() for metric, _ in METRICS]
        positions = centers + offset
        boxes = ax.boxplot(
            values, positions=positions, widths=0.19, patch_artist=True, showfliers=False,
            boxprops={"facecolor": facecolor, "edgecolor": "#34424f", "linewidth": 1.0},
            medianprops={"color": "#17242e", "linewidth": 1.5},
            whiskerprops={"color": "#5f6d79", "linewidth": 1.0},
            capprops={"color": "#5f6d79", "linewidth": 1.0},
        )
        for x, metric_values in zip(positions, values):
            jitter = rng.uniform(-0.045, 0.045, size=len(metric_values))
            ax.scatter(
                np.full(len(metric_values), x) + jitter, metric_values,
                marker=marker, s=20, facecolor=facecolor, edgecolor="#202b35",
                linewidth=0.65, alpha=0.95, zorder=3,
            )

    ax.set_xticks(centers)
    ax.set_xticklabels([label for _, label in METRICS])
    ax.set_xlim(-0.55, len(METRICS) - 0.45)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Independent-test performance")
    ax.set_title("Geographic sensitivity across training regimes", loc="left", fontweight="bold", pad=8)
    ax.grid(axis="y", color="#dde2e6", linewidth=0.7)
    ax.grid(axis="x", visible=False)
    ax.legend(
        handles=[Patch(facecolor=color, edgecolor="#34424f", label=label) for _, label, color, _ in REGIMES],
        loc="lower left", frameon=False, ncol=3, columnspacing=1.2, handlelength=1.5,
    )

    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color("#77838d")
    ax.text(
        1.0, -0.16,
        "n = 22 target cities per regime; boxes show the city-level IQR and center lines show medians",
        transform=ax.transAxes, ha="right", va="top", fontsize=7.3, color="#5f6d79",
    )
    fig.subplots_adjust(left=0.11, right=0.985, top=0.91, bottom=0.23)

    for path in save_figure_formats(fig, args.output_prefix, args.dpi):
        print(path)
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
