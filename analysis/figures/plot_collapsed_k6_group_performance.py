#!/usr/bin/env python3
"""Plot height-screened performance and support for the six collapsed output groups."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from figure_data import save_figure_formats


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = ROOT / "analysis" / "figures" / "provenance" / "collapsed_k6_group_performance_height_screened.csv"
DEFAULT_OUTPUT = ROOT / "outputs" / "manuscript_figures" / "collapsed_k6_group_performance_height_screened"

METRICS = (
    ("precision", "Precision", "o", "#202b35", 0.12),
    ("recall", "Recall", "D", "white", 0.00),
    ("f1", "F1", "s", "#89949d", -0.12),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Frozen group-performance table.")
    parser.add_argument("--output-prefix", type=Path, default=DEFAULT_OUTPUT, help="Output path without extension.")
    parser.add_argument("--dpi", type=int, default=600, help="PNG resolution.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data = pd.read_csv(args.input)
    data["is_reference"] = data["is_reference"].astype(str).str.lower().map({"true": True, "false": False})
    positions = np.arange(len(data))[::-1]

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

    fig, (performance_ax, support_ax) = plt.subplots(
        1, 2, figsize=(8.2, 4.6), sharey=True,
        gridspec_kw={"width_ratios": [1.55, 0.45], "wspace": 0.05},
    )

    for index, y in enumerate(positions):
        if index > 0 and index % 2 == 0:
            for ax in (performance_ax, support_ax):
                ax.axhspan(y - 0.48, y + 0.48, color="#f2f4f5", zorder=-2)
        row = data.iloc[index]
        values = row[[metric for metric, _, _, _, _ in METRICS]].astype(float).to_numpy()
        color = "#26333d" if bool(row["is_reference"]) else "#aeb6bc"
        width = 1.9 if bool(row["is_reference"]) else 1.35
        performance_ax.hlines(y, values.min(), values.max(), color=color, linewidth=width, zorder=1)

    for metric, label, marker, facecolor, offset in METRICS:
        performance_ax.scatter(
            data[metric], positions + offset, marker=marker, s=34,
            facecolor=facecolor, edgecolor="#202b35", linewidth=0.9,
            label=label, zorder=3,
        )

    performance_ax.axhline(positions[0] - 0.5, color="#9aa4ac", linewidth=0.8, linestyle=(0, (3, 3)))
    performance_ax.set_yticks(positions)
    performance_ax.set_yticklabels(["Reference: all groups", *data["group"].iloc[1:].tolist()])
    performance_ax.set_xlim(0.64, 0.93)
    performance_ax.set_xticks(np.arange(0.65, 0.91, 0.05))
    performance_ax.set_xlabel("Independent-test performance")
    performance_ax.set_title("a   Performance by collapsed group", loc="left", fontweight="bold", pad=8)
    performance_ax.grid(axis="x", color="#dde2e6", linewidth=0.7)
    performance_ax.tick_params(axis="y", length=0, pad=6)
    performance_ax.legend(loc="lower right", frameon=False, ncol=3, columnspacing=1.0, handletextpad=0.4)

    support_ax.set_xlim(0, 1)
    support_ax.set_xticks([])
    support_ax.set_xlabel("")
    support_ax.set_title("b   Test records (n)", loc="left", fontweight="bold", pad=8)
    support_ax.tick_params(axis="y", left=False, labelleft=False)
    for y, records in zip(positions, data["records"]):
        support_ax.text(0.92, y, f"{int(records):,}", va="center", ha="right", fontsize=8.5, color="#34424f")

    for ax in (performance_ax, support_ax):
        ax.grid(axis="y", visible=False)
        for spine in ("top", "right", "left"):
            ax.spines[spine].set_visible(False)
        ax.spines["bottom"].set_color("#77838d")
    support_ax.spines["bottom"].set_visible(False)

    fig.text(
        0.985, 0.025,
        "Trees ≥2 m; reference metrics are macro averages across the six groups",
        ha="right", va="bottom", fontsize=7.4, color="#5f6d79",
    )
    fig.subplots_adjust(left=0.23, right=0.965, top=0.91, bottom=0.20)

    for path in save_figure_formats(fig, args.output_prefix, args.dpi):
        print(path)
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
