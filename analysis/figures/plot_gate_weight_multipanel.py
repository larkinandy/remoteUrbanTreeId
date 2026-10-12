#!/usr/bin/env python3
"""Render the final gate-distribution and contextual-deviation multipanel figure."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from figure_data import load_gate_weight_tables, save_figure_formats


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = ROOT / "analysis" / "figures" / "provenance"
DEFAULT_OUTPUT = ROOT / "outputs" / "reviewer_revision" / "final_model_gate_weight_multipanel"
MODALITIES = (
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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    p.add_argument(
        "--box-summary", type=Path,
        default=DEFAULT_INPUT / "gate_weight_boxplot_summary.csv",
        help="Frozen QA-screened boxplot statistics.",
    )
    p.add_argument("--output-prefix", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--minimum-city-support", type=int, default=100)
    p.add_argument("--dpi", type=int, default=600)
    return p.parse_args()


def box_statistics(row: pd.Series, label: str) -> dict[str, float | str | list]:
    return {
        "label": label,
        "whislo": float(row["whislo"]),
        "q1": float(row["q1"]),
        "med": float(row["median"]),
        "q3": float(row["q3"]),
        "whishi": float(row["whishi"]),
        "mean": float(row["mean"]),
        "fliers": [],
    }


def main() -> int:
    a = parse_args()
    box_summary = pd.read_csv(a.box_summary).set_index("modality")
    tables = load_gate_weight_tables(a.input_dir, a.minimum_city_support)
    height, provenance, city_ranges = tables.height, tables.provenance, tables.city_ranges

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
    fig, (box_ax, dev_ax) = plt.subplots(
        1, 2, figsize=(11.2, 4.9), sharey=True,
        gridspec_kw={"width_ratios": [0.92, 1.08], "wspace": 0.08},
    )
    positions = np.arange(len(MODALITIES))[::-1]

    stats = [
        box_statistics(box_summary.loc[modality], LABELS[modality])
        for modality in MODALITIES
    ]
    artists = box_ax.bxp(
        stats, positions=positions, vert=False, widths=0.47, showmeans=True,
        showfliers=False, patch_artist=True,
        boxprops={"facecolor": "#b7c0c8", "edgecolor": "#34424f", "linewidth": 1.0, "alpha": 0.9},
        whiskerprops={"color": "#5f6d79", "linewidth": 1.2},
        capprops={"color": "#5f6d79", "linewidth": 1.2},
        medianprops={"color": "#18242e", "linewidth": 1.7},
        meanprops={"marker": "D", "markerfacecolor": "white", "markeredgecolor": "#18242e", "markersize": 5.2},
    )
    box_ax.set_yticks(positions)
    box_ax.set_yticklabels([LABELS[m] for m in MODALITIES])
    box_ax.set_xlim(0, 1)
    box_ax.set_xlabel("Sample-level modality-gate weight")
    box_ax.set_title("a   Sample-level gate distributions", loc="left", fontweight="bold", pad=8)
    box_ax.grid(axis="x", color="#dde2e6", linewidth=0.7)
    box_ax.grid(axis="y", visible=False)
    box_ax.tick_params(axis="y", length=0, pad=5)
    box_ax.legend(
        handles=[
            Line2D([0], [0], color="#18242e", linewidth=1.7, label="Median"),
            Line2D([0], [0], marker="D", linestyle="none", markerfacecolor="white", markeredgecolor="#18242e", markersize=5.2, label="Mean"),
        ],
        loc="upper right", frameon=False, ncol=2, handlelength=1.8, columnspacing=1.0,
    )

    for y, modality in zip(positions, MODALITIES):
        row = city_ranges.loc[modality]
        dev_ax.hlines(y, row["min"], row["max"], color="#b5b5b5", linewidth=4.0, zorder=1)
        dev_ax.vlines([row["min"], row["max"]], y - 0.045, y + 0.045, color="#777777", linewidth=0.8, zorder=2)

    marker_specs = (
        (height, "height_class", "2–5 m", "^", "#111111", "2–5 m height", 0.16),
        (height, "height_class", "5–10 m", "s", "#555555", "5–10 m height", 0.08),
        (height, "height_class", ">=10 m", "D", "#999999", "≥10 m height", 0.00),
        (provenance, "taxon_provenance", "Core taxa", "o", "white", "Core taxa", -0.08),
        (provenance, "taxon_provenance", "Rare taxa", "X", "#111111", "Rare taxa", -0.16),
    )
    for frame, column, value, marker, face, label, offset in marker_specs:
        subset = frame.loc[frame[column].eq(value)].set_index("modality")
        xs = np.asarray([subset.loc[m, "deviation_pp"] for m in MODALITIES])
        dev_ax.scatter(
            xs, positions + offset, marker=marker, s=29, facecolor=face,
            edgecolor="#111111", linewidth=0.7, label=label, zorder=4,
        )

    maximum = max(
        abs(city_ranges[["min", "max"]].to_numpy()).max(),
        abs(height["deviation_pp"]).max(), abs(provenance["deviation_pp"]).max(),
    )
    limit = np.ceil(maximum + 0.5)
    dev_ax.axvline(0, color="#222222", linewidth=0.8, linestyle=(0, (3, 3)), zorder=0)
    dev_ax.set_xlim(-limit, limit)
    dev_ax.set_xlabel("Difference from overall mean gate weight\n(percentage points)")
    dev_ax.set_title("b   Contextual differences in mean weights", loc="left", fontweight="bold", pad=8)
    dev_ax.grid(axis="x", color="#e2e2e2", linewidth=0.7)
    dev_ax.grid(axis="y", visible=False)
    dev_ax.tick_params(axis="y", left=False, labelleft=False)

    city_count_min = int(city_ranges["count"].min())
    city_count_max = int(city_ranges["count"].max())
    range_label = (
        f"City mean range (n={city_count_min})" if city_count_min == city_count_max
        else f"City mean range (n={city_count_min}–{city_count_max})"
    )
    range_handle = Line2D([0], [0], color="#b5b5b5", linewidth=4.0, label=range_label)
    handles, legend_labels = dev_ax.get_legend_handles_labels()
    dev_ax.legend(
        [range_handle, *handles], [range_label, *legend_labels],
        loc="upper center", bbox_to_anchor=(0.5, -0.20), ncol=3, frameon=False,
        handlelength=1.8, columnspacing=1.0,
    )

    for ax in (box_ax, dev_ax):
        for spine in ("top", "right", "left"):
            ax.spines[spine].set_visible(False)
        ax.spines["bottom"].set_color("#77838d")
    box_ax.text(
        1.0, -0.235,
        f"n = {int(box_summary['records'].max()):,} QA-screened test records; whiskers = 1.5×IQR; outliers not shown",
        transform=box_ax.transAxes, ha="right", va="top", fontsize=7.2, color="#5f6d79",
    )
    fig.subplots_adjust(left=0.215, right=0.99, top=0.91, bottom=0.29)

    for path in save_figure_formats(fig, a.output_prefix, a.dpi):
        print(path)
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
