#!/usr/bin/env python3
"""Plot collapsed-k6 leave-one-branch-out effects for the final stage-2 model."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from figure_data import save_figure_formats


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = ROOT / "analysis" / "figures" / "provenance" / "ablation_collapsed_k6_effects.csv"
DEFAULT_OUTPUT = ROOT / "outputs" / "manuscript_figures" / "ablation_collapsed_k6_effects"

ORDER = (
    "tree_centered_naip",
    "naip_chm_interaction",
    "tree_centered_chm",
    "satellite_embedding",
    "sentinel_phenology",
    "raw_sentinel",
)

LABELS = {
    "tree_centered_naip": "Tree-centered NAIP",
    "naip_chm_interaction": "NAIP–CHM interaction",
    "tree_centered_chm": "Tree-centered CHM",
    "satellite_embedding": "Satellite embedding",
    "sentinel_phenology": "Sentinel-2 phenology",
    "raw_sentinel": "Raw Sentinel-2 time series",
}

METRICS = (
    ("accuracy", "Accuracy", "o", "#202b35", 0.10),
    ("macro_f1", "Macro-F1", "D", "white", 0.00),
    ("weighted_f1", "Weighted-F1", "s", "#89949d", -0.10),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Consolidated ablation summary table.")
    parser.add_argument("--output-prefix", type=Path, default=DEFAULT_OUTPUT, help="Output path without extension.")
    parser.add_argument("--dpi", type=int, default=600, help="PNG resolution.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data = pd.read_csv(args.input)
    data = data.loc[data["label_space"].eq("collapsed_k6")].copy()
    data["ablated_modality"] = pd.Categorical(data["ablated_modality"], categories=ORDER, ordered=True)
    data = data.sort_values("ablated_modality")

    missing = set(ORDER) - set(data["ablated_modality"].astype(str))
    if missing:
        raise ValueError(f"Missing collapsed-k6 ablations: {sorted(missing)}")

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
    positions = np.arange(len(data))[::-1]

    for metric, label, marker, facecolor, offset in METRICS:
        effect = data[f"delta_{metric}"].to_numpy(dtype=float) * 100.0
        low = data[f"{metric}_ci_low"].to_numpy(dtype=float) * 100.0
        high = data[f"{metric}_ci_high"].to_numpy(dtype=float) * 100.0
        xerr = np.vstack((effect - low, high - effect))
        ax.errorbar(
            effect,
            positions + offset,
            xerr=xerr,
            fmt=marker,
            markersize=5.7,
            markerfacecolor=facecolor,
            markeredgecolor="#202b35",
            markeredgewidth=0.9,
            ecolor="#89949d",
            elinewidth=1.2,
            capsize=2.2,
            capthick=1.0,
            linestyle="none",
            label=label,
            zorder=3,
        )

    ax.axvline(0, color="#77838d", linewidth=1.0, linestyle=(0, (3, 3)), zorder=1)
    ax.set_yticks(positions)
    ax.set_yticklabels([LABELS[str(value)] for value in data["ablated_modality"]])
    ax.set_xlim(-0.55, 3.15)
    ax.set_xticks(np.arange(-0.5, 3.1, 0.5))
    ax.set_xlabel("Full minus ablated performance (percentage points)")
    ax.set_title("D. Marginal Effects of Branch Removal", loc="left", fontweight="bold", pad=8)
    ax.grid(axis="x", color="#dde2e6", linewidth=0.7)
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", length=0, pad=5)
    ax.legend(loc="upper right", frameon=False, ncol=1, handletextpad=0.5)

    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color("#77838d")
    ax.text(
        1.0,
        -0.16,
        "Collapsed k = 6; paired-bootstrap 95% CIs (2,000 replicates)",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=7.2,
        color="#5f6d79",
    )
    fig.subplots_adjust(left=0.39, right=0.985, top=0.91, bottom=0.22)

    for path in save_figure_formats(fig, args.output_prefix, args.dpi):
        print(path)
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
