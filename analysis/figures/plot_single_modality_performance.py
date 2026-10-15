#!/usr/bin/env python3
"""Plot independent-test performance of the single-modality stage-2 models."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from figure_data import save_figure_formats


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = ROOT / "analysis" / "figures" / "provenance" / "single_modality_collapsed_k6_performance.csv"
DEFAULT_OUTPUT = ROOT / "outputs" / "manuscript_figures" / "single_modality_collapsed_k6_performance"

LABELS = {
    "final_pruned_multimodal_top5_ensemble": "Final multimodal ensemble",
    "raw_sentinel": "Raw Sentinel-2 time series",
    "naip_chm_interaction": "NAIP–CHM interaction",
    "sentinel_phenology": "Sentinel-2 phenology",
    "tree_centered_naip": "Tree-centered NAIP",
    "tree_centered_structure": "Tree-centered structure",
    "satellite_embedding": "Satellite embedding",
    "tree_centered_chm": "Tree-centered CHM",
}

METRICS = (
    ("accuracy", "Accuracy", "o", "#202b35", 0.10),
    ("macro_f1", "Macro-F1", "D", "white", 0.00),
    ("weighted_f1", "Weighted-F1", "s", "#89949d", -0.10),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Frozen collapsed-k6 performance table.")
    parser.add_argument("--output-prefix", type=Path, default=DEFAULT_OUTPUT, help="Output path without extension.")
    parser.add_argument("--dpi", type=int, default=600, help="PNG resolution.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data = pd.read_csv(args.input)
    data["is_reference"] = data["is_reference"].astype(str).str.lower().map({"true": True, "false": False})
    reference = data.loc[data["is_reference"]]
    singles = data.loc[~data["is_reference"]].sort_values("macro_f1", ascending=False)
    plot_data = pd.concat([reference, singles], ignore_index=True)

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

    fig, ax = plt.subplots(figsize=(5.55, 4.9))
    positions = np.arange(len(plot_data))[::-1]

    for y, (_, row) in zip(positions, plot_data.iterrows()):
        values = row[[metric for metric, _, _, _, _ in METRICS]].astype(float).to_numpy()
        line_color = "#26333d" if bool(row["is_reference"]) else "#aeb6bc"
        line_width = 2.0 if bool(row["is_reference"]) else 1.5
        ax.hlines(y, values.min(), values.max(), color=line_color, linewidth=line_width, zorder=1)

    for metric, label, marker, facecolor, offset in METRICS:
        ax.scatter(
            plot_data[metric], positions + offset, marker=marker, s=34,
            facecolor=facecolor, edgecolor="#202b35", linewidth=0.9,
            label=label, zorder=3,
        )

    ax.axhline(positions[0] - 0.5, color="#9aa4ac", linewidth=0.8, linestyle=(0, (3, 3)))
    ax.set_yticks(positions)
    ax.set_yticklabels([LABELS[value] for value in plot_data["modality"]])
    ax.set_xlim(0.38, 0.82)
    ax.set_xticks(np.arange(0.4, 0.81, 0.1))
    ax.set_xlabel("Independent-test performance")
    ax.set_title("c   Predictive sufficiency of individual modalities", loc="left", fontweight="bold", pad=8)
    ax.grid(axis="x", color="#dde2e6", linewidth=0.7)
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", length=0, pad=5)
    ax.legend(loc="lower right", frameon=False, ncol=1, handletextpad=0.5)

    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color("#77838d")
    ax.text(
        1.0, -0.16,
        f"Collapsed k = 6; n = {int(plot_data['records'].max()):,} independent test records",
        transform=ax.transAxes, ha="right", va="top", fontsize=7.2, color="#5f6d79",
    )
    fig.subplots_adjust(left=0.39, right=0.985, top=0.91, bottom=0.22)

    for path in save_figure_formats(fig, args.output_prefix, args.dpi):
        print(path)
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
