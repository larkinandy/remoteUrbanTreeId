#!/usr/bin/env python3
"""Plot sequential attrition from municipal inventories to the analytical dataset.

The values are frozen study counts. Each removal is conditional on all preceding
screens, while percentages use the original inventory total as the denominator.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import pandas as pd


RAW_RECORDS = 5_427_599


@dataclass(frozen=True)
class AttritionStage:
    section: str
    key: str
    label: str
    before: int
    removed: int
    after: int
    cities: int | None = None
    final: bool = False


STAGES = (
    AttritionStage("CITY-LEVEL FILTERS", "raw", "All municipal inventory records", RAW_RECORDS, 0, RAW_RECORDS, 60),
    AttritionStage("CITY-LEVEL FILTERS", "duplicate_city", "Cities with <50% unique coordinates", RAW_RECORDS, 1_719_778, 3_707_821, 39),
    AttritionStage("CITY-LEVEL FILTERS", "other_city", "Cities lacking required remote-sensing inputs", 3_707_821, 48_120, 3_659_701, 36),
    AttritionStage("RECORD PREPROCESSING", "coordinates", "Invalid or missing coordinates", 3_659_701, 12_957, 3_646_744),
    AttritionStage("RECORD PREPROCESSING", "duplicates", "Duplicate inventory coordinates", 3_646_744, 110_064, 3_536_680),
    AttritionStage("RECORD PREPROCESSING", "crown_match", "No detected crown within 5 m", 3_536_680, 1_260_793, 2_275_887),
    AttritionStage("MULTIMODAL QA HARD SCREENS", "center_vegetation", "Low center vegetation", 2_275_887, 165_242, 2_110_645),
    AttritionStage("MULTIMODAL QA HARD SCREENS", "chm_coverage", "CHM/LiDAR coverage <50%", 2_110_645, 146_907, 1_963_738),
    AttritionStage("MULTIMODAL QA HARD SCREENS", "sentinel", "Missing Sentinel-2 phenology", 1_963_738, 27_306, 1_936_432),
    AttritionStage("MULTIMODAL QA HARD SCREENS", "embedding", "Missing satellite embedding", 1_936_432, 5_411, 1_931_021),
    AttritionStage("TAXONOMIC ELIGIBILITY", "taxon_label", "Missing or ineligible taxonomic label", 1_931_021, 71_451, 1_859_570),
    AttritionStage("TAXONOMIC ELIGIBILITY", "taxon_support", "Taxa represented by <50 records", 1_859_570, 7_602, 1_851_968),
    AttritionStage("FINAL DATASET", "final", "Final analytical dataset", 1_851_968, 0, 1_851_968, 36, True),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/manuscript_figures/dataset_attrition"),
        help="Directory receiving SVG, PDF, PNG, and count-table outputs.",
    )
    parser.add_argument(
        "--basename",
        default="final_dataset_attrition",
        help="Output filename stem.",
    )
    return parser.parse_args()


def write_counts(path: Path) -> None:
    rows = []
    for stage in STAGES:
        rows.append(
            {
                "section": stage.section,
                "stage": stage.key,
                "label": stage.label,
                "records_before": stage.before,
                "records_removed": stage.removed,
                "records_after": stage.after,
                "percent_raw_removed": 100 * stage.removed / RAW_RECORDS,
                "percent_raw_retained": 100 * stage.after / RAW_RECORDS,
                "cities_retained": stage.cities,
            }
        )
    pd.DataFrame(rows).to_csv(path, index=False)


def rounded_box(
    ax: plt.Axes,
    xy: tuple[float, float],
    width: float,
    height: float,
    text_value: str,
    *,
    facecolor: str,
    edgecolor: str,
    text_color: str,
    fontsize: float = 9.2,
    linewidth: float = 1.2,
) -> None:
    box = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        transform=ax.transAxes,
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=linewidth,
        zorder=2,
    )
    ax.add_patch(box)
    ax.text(
        xy[0] + width / 2,
        xy[1] + height / 2,
        text_value,
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=fontsize,
        color=text_color,
        linespacing=1.25,
        zorder=3,
    )


def arrow(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    color: str,
    *,
    connectionstyle: str = "arc3,rad=0",
) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            transform=ax.transAxes,
            arrowstyle="-|>",
            mutation_scale=11,
            linewidth=1.35,
            color=color,
            connectionstyle=connectionstyle,
            shrinkA=2,
            shrinkB=2,
            zorder=1,
        )
    )


def plot_attrition_panel(ax: plt.Axes) -> None:
    blue = "#5f86c9"
    orange = "#c87561"
    green = "#5f9a86"
    text = "#263342"
    muted = "#647386"
    grid = "#dce5ef"

    ax.set_facecolor("white")

    section_gap = 0.52
    section_changes = sum(
        STAGES[index].section != STAGES[index - 1].section
        for index in range(1, len(STAGES))
    )
    plot_height = len(STAGES) + section_gap * section_changes
    y_positions: list[float] = []
    cursor = plot_height - 1
    for index, stage in enumerate(STAGES):
        if index and stage.section != STAGES[index - 1].section:
            cursor -= section_gap
        y_positions.append(cursor)
        cursor -= 1

    panel = FancyBboxPatch(
        (-0.70, -0.75),
        1.92,
        plot_height + 1.3,
        boxstyle="round,pad=0.02,rounding_size=0.025",
        linewidth=1,
        edgecolor="#ccd7e5",
        facecolor="white",
        transform=ax.transData,
        clip_on=False,
        zorder=-10,
    )
    ax.add_patch(panel)

    section_first: dict[str, int] = {}
    for index, stage in enumerate(STAGES):
        section_first.setdefault(stage.section, index)

    for index, stage in enumerate(STAGES):
        y = y_positions[index]
        retained = stage.after / RAW_RECORDS
        removed = stage.removed / RAW_RECORDS
        color = green if stage.final else blue
        ax.barh(y, retained, height=0.58, color=color, edgecolor="none", zorder=2)
        if stage.removed:
            ax.barh(y, removed, left=retained, height=0.58, color=orange, edgecolor="none", zorder=2)

        ax.text(-0.67, y, stage.label, ha="left", va="center", fontsize=8.7, fontweight="semibold", color=text)
        if stage.key == "raw":
            annotation = f"{stage.after:,} (100.0%); {stage.cities} cities"
        elif stage.final:
            annotation = (
                f"{stage.after:,} records ({100 * retained:.1f}% of raw); "
                "579 taxa; 36 cities"
            )
        else:
            annotation = f"−{stage.removed:,} ({100 * removed:.1f}% of raw)\n{stage.after:,} remaining"
            if stage.cities is not None:
                annotation += f"; {stage.cities} cities"
        ax.text(min(retained + removed + 0.012, 1.015), y, annotation, ha="left", va="center", fontsize=7.5, color=muted)

        if section_first[stage.section] == index:
            ax.text(
                -0.67,
                y + 0.58,
                stage.section,
                ha="left",
                va="bottom",
                fontsize=7.6,
                fontweight="bold",
                color=muted,
            )

    ax.set_xlim(-0.70, 1.20)
    ax.set_ylim(-0.85, plot_height - 0.1)
    ax.set_yticks([])
    ax.set_xticks([0.25, 0.50, 0.75, 1.00], labels=["25%", "50%", "75%", "100%"])
    ax.tick_params(axis="x", colors=muted, labelsize=7.5)
    ax.grid(axis="x", color=grid, linestyle=(0, (3, 4)), linewidth=0.8, zorder=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.set_title("A. Inventory and analytical-dataset attrition", loc="left", fontsize=12.2, fontweight="bold", color=text, pad=12)


def plot_partition_panel(ax: plt.Axes) -> None:
    text = "#263342"
    muted = "#647386"
    black = "#202830"
    mid_gray = "#707982"
    light_gray = "#f1f2f3"
    lighter_gray = "#fafafa"

    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("B. Fixed partitions and their analytical roles", loc="left", fontsize=12.2, fontweight="bold", color=text, pad=12)

    rounded_box(
        ax,
        (0.16, 0.875),
        0.68,
        0.08,
        "Final analytical dataset\n1,851,968 records · 579 taxa · 36 cities",
        facecolor=light_gray,
        edgecolor=black,
        text_color=text,
        fontsize=9.8,
        linewidth=1.5,
    )
    ax.text(0.50, 0.845, "Fixed, group-stratified split", transform=ax.transAxes, ha="center", va="center", fontsize=8.2, color=muted)
    arrow(ax, (0.50, 0.875), (0.50, 0.815), black)

    label_x = 0.01
    grid_x0 = 0.19
    gap = 0.045
    column_w = (0.98 - grid_x0 - 2 * gap) / 3
    column_x = {
        "train": grid_x0,
        "validation": grid_x0 + column_w + gap,
        "test": grid_x0 + 2 * (column_w + gap),
    }
    headers = {
        "train": "TRAINING\n1,590,460 (85.9%)",
        "validation": "VALIDATION\n100,000 (5.4%)",
        "test": "INDEPENDENT TEST\n161,508 (8.7%)",
    }
    lane_faces = {"train": "#f4f4f4", "validation": "#fafafa", "test": "#eeeeee"}
    for key, x in column_x.items():
        lane = FancyBboxPatch(
            (x - 0.012, 0.115),
            column_w + 0.024,
            0.705,
            boxstyle="round,pad=0.004,rounding_size=0.018",
            transform=ax.transAxes,
            facecolor=lane_faces[key],
            edgecolor="none",
            zorder=0,
        )
        ax.add_patch(lane)
        rounded_box(
            ax,
            (x, 0.725),
            column_w,
            0.085,
            headers[key],
            facecolor=light_gray,
            edgecolor=black,
            text_color=text,
            fontsize=8.0,
            linewidth=1.2,
        )

    row_specs = {
        "discriminator": (0.475, "DISCRIMINATOR"),
        "stage_two": (0.175, "STAGE-TWO\nMODEL"),
    }
    cell_h = 0.19
    for _, (y, label) in row_specs.items():
        ax.text(
            label_x + 0.08,
            y + cell_h / 2,
            label,
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=8.3,
            fontweight="bold",
            color=text,
        )

    cell_text = {
        ("discriminator", "train"): "Fit discriminator\nusing core taxa\n(≥500 records)",
        ("discriminator", "validation"): "Select checkpoint;\nestimate taxon centroids;\ndefine group hierarchy",
        ("discriminator", "test"): "Independent\ndiscriminator\nevaluation only",
        ("stage_two", "train"): "Fit 11-class model\nusing all 579\nassigned taxa",
        ("stage_two", "validation"): "Select checkpoints;\nevaluate and select\nprobability-sum collapse",
        ("stage_two", "test"): "Final k=11 and\ncollapsed-k6\nperformance evaluation",
    }
    for row_key, (y, _) in row_specs.items():
        for column_key, x in column_x.items():
            rounded_box(
                ax,
                (x, y),
                column_w,
                cell_h,
                cell_text[(row_key, column_key)],
                facecolor="white",
                edgecolor=mid_gray if column_key != "test" else black,
                text_color=text,
                fontsize=8.0,
                linewidth=1.1 if column_key != "test" else 1.6,
            )

    for key, x in column_x.items():
        center_x = x + column_w / 2
        arrow(ax, (0.50, 0.875), (center_x, 0.812), mid_gray, connectionstyle="arc3,rad=0")
        arrow(
            ax,
            (center_x, row_specs["discriminator"][0]),
            (center_x, row_specs["stage_two"][0] + cell_h),
            black,
        )

    ax.text(
        0.59,
        0.075,
        "Test records were not used to define groups, fit models,\nor select checkpoints.",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=8.8,
        fontweight="bold",
        color=black,
    )


def plot(path_stem: Path) -> None:
    text = "#263342"
    muted = "#647386"
    fig = plt.figure(figsize=(16.2, 9.2), facecolor="#f7f9fc")
    grid_spec = fig.add_gridspec(1, 2, width_ratios=(1.05, 0.95), wspace=0.08)
    left_ax = fig.add_subplot(grid_spec[0, 0])
    right_ax = fig.add_subplot(grid_spec[0, 1])
    plot_attrition_panel(left_ax)
    plot_partition_panel(right_ax)

    fig.suptitle("From municipal inventories to independent model evaluation", x=0.03, y=0.985, ha="left", fontsize=18, fontweight="bold", color=text)
    fig.text(
        0.03,
        0.947,
        "Sequential filtering, fixed data partitioning, and use of each partition in the two-stage modeling workflow.",
        fontsize=9.8,
        color=muted,
    )
    fig.text(
        0.03,
        0.018,
        "Panel A removals are conditional on preceding screens; percentages use the original 5,427,599 records as the denominator.",
        fontsize=8.2,
        color=muted,
    )

    fig.subplots_adjust(left=0.03, right=0.985, top=0.90, bottom=0.07)
    fig.savefig(path_stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(path_stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(path_stem.with_suffix(".png"), dpi=600, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    path_stem = args.output_dir / args.basename
    write_counts(path_stem.with_name(f"{args.basename}_counts.csv"))
    plot(path_stem)
    print(f"Wrote {path_stem}.svg, .pdf, .png, and count table")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
