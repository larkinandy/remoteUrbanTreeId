#!/usr/bin/env python3
"""Plot contextual deviations from overall mean modality-gate weights."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = ROOT / "outputs" / "reviewer_revision" / "final_gate_weight_stratified_analysis"
DEFAULT_OUTPUT = DEFAULT_INPUT / "gate_weight_context_deviation"
MODALITIES = (
    "tree_centered_naip",
    "tree_centered_chm",
    "naip_chm_interaction",
    "sentinel_phenology",
    "raw_sentinel",
    "satellite_embedding",
)
LABELS = {
    "tree_centered_naip": "NAIP",
    "tree_centered_chm": "CHM",
    "naip_chm_interaction": "NAIP–CHM interaction",
    "sentinel_phenology": "Sentinel-2 phenology",
    "raw_sentinel": "Raw Sentinel-2",
    "satellite_embedding": "Satellite embedding",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    p.add_argument("--output-prefix", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--minimum-city-support", type=int, default=100)
    p.add_argument("--dpi", type=int, default=600)
    return p.parse_args()


def main() -> int:
    a = parse_args()
    overall = pd.read_csv(a.input_dir / "gate_weights_overall.csv").set_index("modality")["mean"]
    city = pd.read_csv(a.input_dir / "gate_weights_by_city.csv")
    height = pd.read_csv(a.input_dir / "gate_weights_by_height_class_qa_screened.csv")
    provenance = pd.read_csv(a.input_dir / "gate_weights_by_taxon_provenance.csv")

    city = city.loc[city["support"].ge(a.minimum_city_support)].copy()
    city["deviation_pp"] = 100.0 * (city["mean"] - city["modality"].map(overall))
    height = height.loc[height["height_class"].isin(["2–5 m", "5–10 m", ">=10 m"])].copy()
    height["deviation_pp"] = 100.0 * (height["mean"] - height["modality"].map(overall))
    provenance["deviation_pp"] = 100.0 * (provenance["mean"] - provenance["modality"].map(overall))

    city_ranges = city.groupby("modality")["deviation_pp"].agg(["min", "max", "count"])
    marker_specs = (
        (height, "height_class", "2–5 m", "^", "#111111", "2–5 m height", 0.16),
        (height, "height_class", "5–10 m", "s", "#4d4d4d", "5–10 m height", 0.08),
        (height, "height_class", ">=10 m", "D", "#8a8a8a", "≥10 m height", 0.00),
        (provenance, "taxon_provenance", "Core taxa", "o", "white", "Core taxa", -0.08),
        (provenance, "taxon_provenance", "Rare taxa", "X", "#111111", "Rare taxa", -0.16),
    )

    plt.rcParams.update({
        "font.family": "Arial",
        "font.size": 9,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 8,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
    })
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    y_base = np.arange(len(MODALITIES))[::-1]

    for y, modality in zip(y_base, MODALITIES):
        row = city_ranges.loc[modality]
        ax.hlines(y, row["min"], row["max"], color="#b5b5b5", linewidth=4.5, zorder=1)
        ax.vlines([row["min"], row["max"]], y - 0.055, y + 0.055, color="#7f7f7f", linewidth=0.9, zorder=2)

    for frame, column, value, marker, face, label, offset in marker_specs:
        subset = frame.loc[frame[column].eq(value)].set_index("modality")
        xs = np.asarray([subset.loc[m, "deviation_pp"] for m in MODALITIES])
        ys = y_base + offset
        ax.scatter(
            xs, ys, marker=marker, s=38, facecolor=face, edgecolor="#111111",
            linewidth=0.8, label=label, zorder=4,
        )

    maximum = max(
        abs(city_ranges[["min", "max"]].to_numpy()).max(),
        abs(height["deviation_pp"]).max(),
        abs(provenance["deviation_pp"]).max(),
    )
    limit = np.ceil((maximum + 0.5) / 1.0) * 1.0
    ax.axvline(0, color="#222222", linewidth=0.9, linestyle=(0, (3, 3)), zorder=0)
    ax.set_xlim(-limit, limit)
    ax.set_yticks(y_base)
    ax.set_yticklabels([LABELS[m] for m in MODALITIES])
    ax.set_xlabel("Difference from overall mean gate weight (percentage points)")
    ax.set_title("Contextual differences in mean modality-gate weights", loc="left", fontweight="bold", pad=10)
    ax.grid(axis="x", color="#e6e6e6", linewidth=0.7, zorder=0)
    ax.grid(axis="y", visible=False)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color("#777777")
    ax.tick_params(axis="y", length=0)

    city_count_min = int(city_ranges["count"].min())
    city_count_max = int(city_ranges["count"].max())
    range_label = (
        f"City mean range (n={city_count_min})"
        if city_count_min == city_count_max
        else f"City mean range (n={city_count_min}–{city_count_max})"
    )
    range_handle = plt.Line2D([0], [0], color="#b5b5b5", linewidth=4.5, label=range_label)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        [range_handle, *handles], [range_label, *labels], loc="upper center",
        bbox_to_anchor=(0.5, -0.17), ncol=3, frameon=False,
        handlelength=2.2, columnspacing=1.4,
    )
    fig.subplots_adjust(left=0.27, right=0.98, top=0.88, bottom=0.29)

    a.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "pdf", "svg"):
        path = a.output_prefix.with_suffix(f".{extension}")
        fig.savefig(path, dpi=a.dpi if extension == "png" else None, bbox_inches="tight", facecolor="white")
        print(path)
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
