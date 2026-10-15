#!/usr/bin/env python3
"""Prepare publication-facing Sentinel-2 interpolation-gap distributions.

The event histogram uses the saved gap-event frequency table. The city boxplots
use each record's longest interpolation gap. Records without interpolation are
assigned the expected five-day interval between sequential observations, as
specified for the manuscript analysis. The companion Node renderer converts the
prepared tables into publication-ready SVG and PNG figures.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BLUE = "#0b78b5"
DARK = "#25384c"
GRID = "#d8dee5"
NO_GAP_DAYS = 5.0
HISTOGRAM_BIN_DAYS = 5
HISTOGRAM_OVERFLOW_START = 120


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot Sentinel-2 interpolation-gap event and city distributions."
    )
    parser.add_argument(
        "--statistics-dir",
        type=Path,
        required=True,
        help="Directory containing the saved interpolation summary products.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="Frozen final stage-two record manifest whose row order matches the record-level NPZ.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/manuscript_figures/sentinel_interpolation_gaps"),
        help="Output directory for figures and plotted-data tables.",
    )
    parser.add_argument(
        "--complete-shard-root",
        type=Path,
        required=True,
        help="Root containing complete stage-two shards used to count observed five-day transitions.",
    )
    return parser.parse_args()


def set_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.labelsize": 10,
            "axes.edgecolor": "#9aabba",
            "axes.linewidth": 0.8,
            "xtick.color": "#607284",
            "ytick.color": "#607284",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def save_figure(figure: plt.Figure, output_stem: Path) -> None:
    figure.savefig(output_stem.with_suffix(".svg"), bbox_inches="tight")
    figure.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(output_stem.with_suffix(".png"), dpi=600, bbox_inches="tight")


def count_no_interpolation_events(manifest_path: Path, shard_root: Path) -> dict[str, int]:
    """Count adjacent observed sequence positions with temporal spacing of 1–14 days."""
    manifest = pd.read_csv(
        manifest_path,
        usecols=["city_token", "source_key", "local_row"],
    )
    event_count = 0
    selected_records = 0
    for (city, source_key), group in manifest.groupby(
        ["city_token", "source_key"], sort=False
    ):
        shard_path = shard_root / city / f"{source_key}_inputs.npz"
        with np.load(shard_path, allow_pickle=False) as archive:
            columns = archive["sentinel_feature_columns"]
            observed_index = int(np.flatnonzero(columns == "sentinel_observed")[0])
            rows = group["local_row"].to_numpy(dtype=np.int64)
            observed = archive["sentinel_sequence"][rows, :, observed_index] > 0.5
            valid = archive["sentinel_sequence_mask"][rows].astype(bool)
        observed &= valid
        # Consecutive positions on the regularized grid are five days apart;
        # two adjacent observed positions therefore define a no-interpolation event.
        normal_transition = observed[:, :-1] & observed[:, 1:]
        event_count += int(normal_transition.sum())
        selected_records += len(rows)
    return {
        "selected_records": selected_records,
        "no_interpolation_events_0_to_14_days": event_count,
    }


def build_event_histogram(source: Path, no_interpolation_events: int) -> pd.DataFrame:
    event_data = pd.read_csv(source)
    days = event_data["gap_days"].to_numpy(dtype=float)
    counts = event_data["gap_events"].to_numpy(dtype=np.int64)
    bin_starts = np.arange(15, HISTOGRAM_OVERFLOW_START, HISTOGRAM_BIN_DAYS)
    rows: list[dict[str, object]] = [
        {
            "bin_start_days": 0,
            "bin_end_days_exclusive": 15,
            "bin_label": "No gap",
            "gap_events": no_interpolation_events,
            "count_type": "observed_adjacent_events_without_interpolation",
        }
    ]
    for start in bin_starts:
        stop = start + HISTOGRAM_BIN_DAYS
        rows.append(
            {
                "bin_start_days": int(start),
                "bin_end_days_exclusive": int(stop),
                "bin_label": f"{start}–{stop - 1}",
                "gap_events": int(counts[(days >= start) & (days < stop)].sum()),
                "count_type": "interpolation_gap_events",
            }
        )
    rows.append(
        {
            "bin_start_days": HISTOGRAM_OVERFLOW_START,
            "bin_end_days_exclusive": np.nan,
            "bin_label": f"≥{HISTOGRAM_OVERFLOW_START}",
            "gap_events": int(counts[days >= HISTOGRAM_OVERFLOW_START].sum()),
            "count_type": "interpolation_gap_events",
        }
    )
    return pd.DataFrame(rows)


def plot_event_histogram(histogram: pd.DataFrame, output_dir: Path) -> None:
    figure, axis = plt.subplots(figsize=(7.1, 4.2))
    positions = np.arange(len(histogram))
    axis.bar(
        positions,
        histogram["gap_events"] / 1_000,
        color=BLUE,
        edgecolor="white",
        linewidth=0.45,
        width=0.92,
    )
    tick_positions = list(range(0, len(histogram) - 1, 3)) + [len(histogram) - 1]
    axis.set_xticks(tick_positions)
    axis.set_xticklabels(histogram.iloc[tick_positions]["bin_label"], rotation=45, ha="right")
    axis.set_xlabel("Interpolation-gap duration (days)")
    axis.set_ylabel("Gap events (thousands)")
    axis.set_title("Sentinel-2 interpolation-gap events")
    axis.grid(axis="y", color=GRID, linewidth=0.7)
    axis.set_axisbelow(True)
    axis.spines[["top", "right"]].set_visible(False)
    total_events = int(histogram["gap_events"].sum())
    axis.text(
        0.99,
        0.96,
        f"n = {total_events:,} events",
        transform=axis.transAxes,
        ha="right",
        va="top",
        color="#607284",
    )
    figure.tight_layout()
    save_figure(figure, output_dir / "Figure_S_sentinel_interpolation_gap_events")
    plt.close(figure)


def load_record_distributions(
    record_npz: Path, manifest_path: Path
) -> tuple[pd.Series, np.ndarray]:
    manifest = pd.read_csv(manifest_path, usecols=["city_token"])
    with np.load(record_npz, allow_pickle=False) as archive:
        gap_days = archive["longest_interpolation_gap_days"].astype(np.float32, copy=True)
    if len(manifest) != len(gap_days):
        raise ValueError(
            f"Manifest and record arrays differ in length: {len(manifest):,} versus {len(gap_days):,}."
        )
    gap_days[gap_days <= 0] = NO_GAP_DAYS
    return manifest["city_token"], gap_days


def city_display_name(token: str) -> str:
    replacements = {
        "auroraco": "Aurora (CO)",
        "washingtondc": "Washington (DC)",
        "siouxfalls": "Sioux Falls",
        "sanfrancisco": "San Francisco",
        "sanjose": "San Jose",
        "newyork": "New York",
        "stlouis": "St. Louis",
        "desmoines": "Des Moines",
        "coloradosprings": "Colorado Springs",
        "oklahomacity": "Oklahoma City",
        "ranchocucamonga": "Rancho Cucamonga",
        "huntingtonbeach": "Huntington Beach",
        "lasvegas": "Las Vegas",
        "gardengrove": "Garden Grove",
        "santarosa": "Santa Rosa",
        "sandiego": "San Diego",
        "capecoral": "Cape Coral",
    }
    return replacements.get(token, token.replace("_", " ").title())


def summarize_cities(cities: pd.Series, gaps: np.ndarray) -> pd.DataFrame:
    frame = pd.DataFrame({"city": cities.to_numpy(), "gap_days": gaps})
    rows: list[dict[str, object]] = []
    for city, values in frame.groupby("city", sort=True)["gap_days"]:
        array = values.to_numpy()
        q1, median, q3 = np.quantile(array, [0.25, 0.5, 0.75])
        iqr = q3 - q1
        lower_values = array[array >= q1 - 1.5 * iqr]
        upper_values = array[array <= q3 + 1.5 * iqr]
        rows.append(
            {
                "city": city,
                "records": len(array),
                "minimum": float(array.min()),
                "lower_whisker": float(lower_values.min()),
                "q1": float(q1),
                "median": float(median),
                "q3": float(q3),
                "upper_whisker": float(upper_values.max()),
                "maximum": float(array.max()),
            }
        )
    summary = pd.DataFrame(rows)
    summary["city_label"] = summary["city"].map(city_display_name)
    return summary.sort_values(["median", "q3", "city_label"], ascending=[True, True, True])


def plot_city_boxplots(
    cities: pd.Series, gaps: np.ndarray, city_summary: pd.DataFrame, output_dir: Path
) -> None:
    city_arrays = {
        city: gaps[cities.to_numpy() == city]
        for city in city_summary["city"]
    }
    data = [city_arrays[city] for city in city_summary["city"]]
    labels = city_summary["city_label"].tolist()
    figure_height = max(7.4, 0.25 * len(labels) + 1.7)
    figure, axis = plt.subplots(figsize=(7.2, figure_height))
    axis.boxplot(
        data,
        vert=False,
        tick_labels=labels,
        showfliers=False,
        whis=1.5,
        widths=0.62,
        patch_artist=True,
        boxprops={"facecolor": BLUE, "edgecolor": DARK, "linewidth": 0.7, "alpha": 0.82},
        medianprops={"color": "white", "linewidth": 1.2},
        whiskerprops={"color": DARK, "linewidth": 0.7},
        capprops={"color": DARK, "linewidth": 0.7},
    )
    axis.axvline(NO_GAP_DAYS, color="#d45a00", linestyle="--", linewidth=1.0)
    axis.set_xscale("log")
    axis.set_xlim(4.5, max(1100, float(np.nanmax(gaps)) * 1.05))
    axis.set_xlabel("Longest interpolation-gap duration per record (days, logarithmic scale)")
    axis.set_ylabel("City")
    axis.set_title("Sentinel-2 interpolation gaps by city")
    axis.grid(axis="x", which="major", color=GRID, linewidth=0.7)
    axis.set_axisbelow(True)
    axis.spines[["top", "right"]].set_visible(False)
    axis.text(
        NO_GAP_DAYS,
        1.01,
        "No interpolation (5-day interval)",
        transform=axis.get_xaxis_transform(),
        ha="left",
        va="bottom",
        color="#a64b00",
        fontsize=8,
    )
    figure.tight_layout()
    save_figure(figure, output_dir / "Figure_S_sentinel_longest_gap_by_city")
    plt.close(figure)


def main() -> None:
    arguments = parse_args()
    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    set_style()

    no_gap_summary_path = arguments.output_dir / "no_interpolation_event_summary.json"
    if no_gap_summary_path.exists():
        with no_gap_summary_path.open("r", encoding="utf-8") as stream:
            no_gap_summary = json.load(stream)
    else:
        no_gap_summary = count_no_interpolation_events(
            arguments.manifest, arguments.complete_shard_root
        )
        with no_gap_summary_path.open("w", encoding="utf-8") as stream:
            json.dump(no_gap_summary, stream, indent=2)
    histogram = build_event_histogram(
        arguments.statistics_dir / "gap_duration_distribution_by_event.csv",
        int(no_gap_summary["no_interpolation_events_0_to_14_days"]),
    )
    histogram.to_csv(arguments.output_dir / "gap_event_histogram_data.csv", index=False)
    cities, gaps = load_record_distributions(
        arguments.statistics_dir / "record_level_distributions.npz", arguments.manifest
    )
    city_summary = summarize_cities(cities, gaps)
    city_summary.to_csv(arguments.output_dir / "city_longest_gap_boxplot_summary.csv", index=False)



if __name__ == "__main__":
    main()
