#!/usr/bin/env python3
"""Create the publication gate-weight boxplot and contextual-deviation multipanel figure."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = ROOT / "outputs" / "reviewer_revision" / "final_gate_weight_stratified_analysis"
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
    p.add_argument("--output-prefix", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--minimum-city-support", type=int, default=100)
    p.add_argument("--dpi", type=int, default=600)
    return p.parse_args()


def box_statistics(values: np.ndarray, label: str) -> dict[str, float | str]:
    q1, median, q3 = np.quantile(values, [0.25, 0.50, 0.75])
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    within = values[(values >= lower) & (values <= upper)]
    return {
        "label": label,
        "whislo": float(within.min()),
        "q1": float(q1),
        "med": float(median),
        "q3": float(q3),
        "whishi": float(within.max()),
        "mean": float(values.mean()),
        "fliers": [],
    }


def main() -> int:
    a = parse_args()
    records = pd.read_csv(a.input_dir / "gate_weight_stratification_records.csv.gz")
    records = records.loc[records["qa_screened"].astype(bool)].copy()
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
        box_statistics(records[f"gate_{modality}"].to_numpy(dtype=float), LABELS[modality])
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
        1.0, -0.235, f"n = {len(records):,} QA-screened test records; whiskers = 1.5×IQR; outliers not shown",
        transform=box_ax.transAxes, ha="right", va="top", fontsize=7.2, color="#5f6d79",
    )
    fig.subplots_adjust(left=0.215, right=0.99, top=0.91, bottom=0.29)

    a.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "pdf", "svg"):
        path = a.output_prefix.with_suffix(f".{extension}")
        fig.savefig(path, dpi=a.dpi if extension == "png" else None, bbox_inches="tight", facecolor="white")
        print(path)
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
