#!/usr/bin/env python3
"""Render aligned k11 and collapsed_k6 discrimination-space SVG panels."""

from __future__ import annotations

import argparse
import csv
import html
from pathlib import Path

import numpy as np

from plot_discrimination_space import (
    K11_MANUSCRIPT_COLORS,
    COLLAPSED_K6_COLORS,
    LABELS,
    LABEL_OFFSETS,
    pretty_taxon,
)
from figure_config import manuscript_collapse_zero_based


# This map is in the manuscript display order used by the input CSV, after the
# internal k11 identifiers have been reordered for a contiguous legend. The
# model-space collapse remains the mapping in config/final_grouping.json.
HYBRID_MAP = manuscript_collapse_zero_based()


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--k11-csv",
        type=Path,
        default=root / "analysis/figures/provenance/manuscript_k11_discrimination_space.csv",
    )
    parser.add_argument(
        "--collapsed-k6-csv",
        type=Path,
        default=root / "analysis/figures/provenance/manuscript_collapsed_k6_discrimination_space.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "outputs/manuscript_figures/combined_k11_collapsed_k6_discrimination_space_hybrid_legend.svg",
    )
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str | int | float]]:
    """Read and type a frozen discrimination-space table."""
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return [
        {
            "taxon": row["taxon"],
            "group": int(row["group"]) - 1,
            "taxon_type": row["taxon_type"],
            "support": int(row["records"]),
            "x": float(row["umap_1"]),
            "y": float(row["umap_2"]),
        }
        for row in rows
    ]


def draw_panel(
    svg: list[str], rows: list[dict], colors: tuple[str, ...], panel_x: float,
    title: str, panel_letter: str, bounds: tuple[float, float, float, float], maximum: int,
) -> None:
    plot_y, plot_size = 190.0, 900.0
    xmin, xmax, ymin, ymax = bounds

    def sx(value: float) -> float:
        return panel_x + (value - xmin) / (xmax - xmin) * plot_size

    def sy(value: float) -> float:
        return plot_y + plot_size - (value - ymin) / (ymax - ymin) * plot_size

    def radius(value: int) -> float:
        return 3.3 + 14.7 * np.sqrt(value / maximum)

    svg.extend(
        [
            f'<text x="{panel_x}" y="142" fill="#1f2a36" font-family="Arial" font-size="26" font-weight="700">{panel_letter}. {html.escape(title)}</text>',
            f'<rect x="{panel_x}" y="{plot_y}" width="{plot_size}" height="{plot_size}" rx="10" fill="#fafbfc" stroke="#cbd5dc" stroke-width="1.5"/>',
        ]
    )
    core = [row for row in rows if row["taxon_type"] == "core"]
    rare = [row for row in rows if row["taxon_type"] != "core"]
    for row in sorted(core, key=lambda item: item["support"], reverse=True):
        selected = row["taxon"] in LABELS
        stroke = "#27313a" if selected else "#ffffff"
        stroke_width = 2.4 if selected else 1.5
        svg.append(
            f'<circle cx="{sx(row["x"]):.2f}" cy="{sy(row["y"]):.2f}" '
            f'r="{radius(row["support"]):.2f}" fill="{colors[row["group"]]}" '
            f'fill-opacity="0.74" stroke="{stroke}" stroke-width="{stroke_width}"/>'
        )
    for row in rare:
        px, py, arm = sx(row["x"]), sy(row["y"]), 4.2
        color = colors[row["group"]]
        for stroke, width in (("#ffffff", 3.8), (color, 2.0)):
            svg.append(f'<line x1="{px-arm:.2f}" y1="{py:.2f}" x2="{px+arm:.2f}" y2="{py:.2f}" stroke="{stroke}" stroke-width="{width}" stroke-linecap="round"/>')
            svg.append(f'<line x1="{px:.2f}" y1="{py-arm:.2f}" x2="{px:.2f}" y2="{py+arm:.2f}" stroke="{stroke}" stroke-width="{width}" stroke-linecap="round"/>')

    label_rows = sorted(
        [row for row in core if row["taxon"] in LABELS],
        key=lambda item: (item["group"], -item["support"]),
    )
    offsets = {group: 0 for group in range(len(colors))}
    for row in label_rows:
        default_direction = 1 if offsets[row["group"]] % 2 == 0 else -1
        default_vertical = -20 if offsets[row["group"]] % 3 else 20
        offsets[row["group"]] += 1
        px, py = sx(row["x"]), sy(row["y"])
        dx, dy = LABEL_OFFSETS.get(row["taxon"], (12 * default_direction, default_vertical))
        direction = 1 if dx >= 0 else -1
        tx, ty = px + dx, py + dy
        anchor = "start" if direction > 0 else "end"
        name = html.escape(pretty_taxon(row["taxon"]))
        svg.append(f'<line x1="{px:.2f}" y1="{py:.2f}" x2="{tx:.2f}" y2="{ty:.2f}" stroke="#7b8994" stroke-width="1"/>')
        svg.append(
            f'<text x="{tx + 4 * direction:.2f}" y="{ty + 5:.2f}" text-anchor="{anchor}" '
            f'fill="#1f2a36" font-family="Arial" font-size="14" font-style="italic" '
            f'stroke="#ffffff" stroke-width="3.5" paint-order="stroke" stroke-linejoin="round">{name}</text>'
        )
    svg.append(f'<text x="{panel_x + plot_size/2}" y="1128" text-anchor="middle" fill="#5c6c7a" font-family="Arial" font-size="18">UMAP 1</text>')
    svg.append(f'<text x="{panel_x-47}" y="{plot_y+plot_size/2}" text-anchor="middle" fill="#5c6c7a" font-family="Arial" font-size="18" transform="rotate(-90 {panel_x-47} {plot_y+plot_size/2})">UMAP 2</text>')


def main() -> None:
    args = parse_args()
    k11_rows = read_rows(args.k11_csv)
    collapsed_k6_rows = read_rows(args.collapsed_k6_csv)
    core = [row for row in k11_rows if row["taxon_type"] == "core"]
    xs = np.asarray([row["x"] for row in core])
    ys = np.asarray([row["y"] for row in core])
    xpad, ypad = (xs.max() - xs.min()) * 0.04, (ys.max() - ys.min()) * 0.04
    bounds = (xs.min() - xpad, xs.max() + xpad, ys.min() - ypad, ys.max() + ypad)
    maximum = max(row["support"] for row in core)

    svg = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="2100" height="1450" viewBox="0 0 2100 1450" role="img" aria-labelledby="title desc">',
        '<title id="title">Discriminator separation space for k11 and collapsed_k6</title>',
        '<desc id="desc">Two aligned UMAP panels with a shared hierarchy legend showing how eleven remote-sensing-derived taxon groups collapse into six groups.</desc>',
        '<rect width="2380" height="1450" fill="#ffffff"/>',
        '<defs><marker id="legend-arrow" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto"><path d="M0,0 L7,3.5 L0,7 Z" fill="#7b8994"/></marker></defs>',
        '<text x="70" y="58" fill="#1f2a36" font-family="Arial" font-size="38" font-weight="700">Discriminator separation space before and after group collapse</text>',
        '<text x="70" y="94" fill="#5c6c7a" font-family="Arial" font-size="19">Identical taxon-centroid coordinates are shown in both panels; colors indicate the assigned group at each reporting level.</text>',
    ]
    draw_panel(svg, k11_rows, K11_MANUSCRIPT_COLORS, 90.0, "Fine partition (k11)", "A", bounds, maximum)
    draw_panel(svg, collapsed_k6_rows, COLLAPSED_K6_COLORS, 1110.0, "Collapsed partition (collapsed_k6)", "B", bounds, maximum)

    svg.append('<text x="90" y="1184" fill="#1f2a36" font-family="Arial" font-size="24" font-weight="700">Shared group hierarchy</text>')
    svg.append('<text x="375" y="1184" fill="#7b8994" font-family="Arial" font-size="15">k11 groups</text>')
    svg.append('<text x="510" y="1184" fill="#7b8994" font-family="Arial" font-size="15">collapsed_k6 group</text>')
    svg.append('<circle cx="1395" cy="1171" r="4" fill="#7b8994" fill-opacity="0.55"/><circle cx="1427" cy="1171" r="8" fill="#7b8994" fill-opacity="0.55"/><circle cx="1471" cy="1171" r="13" fill="#7b8994" fill-opacity="0.55"/>')
    svg.append('<text x="1495" y="1177" fill="#5c6c7a" font-family="Arial" font-size="14">area = validation support</text>')
    svg.append('<line x1="1790" y1="1171" x2="1802" y2="1171" stroke="#596773" stroke-width="2.4" stroke-linecap="round"/><line x1="1796" y1="1165" x2="1796" y2="1177" stroke="#596773" stroke-width="2.4" stroke-linecap="round"/>')
    svg.append('<text x="1814" y="1177" fill="#5c6c7a" font-family="Arial" font-size="14">rare-extension taxon</text>')
    total_records = sum(row["support"] for row in core)
    for collapsed_group in range(6):
        column, row_index = collapsed_group % 3, collapsed_group // 3
        left, top = 90 + column * 650, 1204 + row_index * 112
        center_y = top + 43
        sources = np.flatnonzero(HYBRID_MAP == collapsed_group).tolist()
        members = [item for item in core if item["group"] in sources]
        record_percent = 100.0 * sum(item["support"] for item in members) / total_records
        svg.append(f'<rect x="{left}" y="{top}" width="560" height="88" rx="10" fill="#fbfcfd" stroke="#d5dee5" stroke-width="1.2"/>')
        for source_index, source_group in enumerate(sources):
            cx = left + 30 + 38 * source_index
            svg.append(f'<circle cx="{cx}" cy="{center_y}" r="13" fill="{K11_MANUSCRIPT_COLORS[source_group]}" fill-opacity="0.9"/>')
            svg.append(f'<text x="{cx}" y="{center_y+5}" text-anchor="middle" fill="#ffffff" font-family="Arial" font-size="12" font-weight="700" stroke="#27313a" stroke-width="0.7" paint-order="stroke">{source_group+1}</text>')
        arrow_x = left + 165
        svg.append(f'<line x1="{arrow_x}" y1="{center_y}" x2="{arrow_x+45}" y2="{center_y}" stroke="#7b8994" stroke-width="1.8" marker-end="url(#legend-arrow)"/>')
        svg.append(f'<circle cx="{left+240}" cy="{center_y}" r="15" fill="{COLLAPSED_K6_COLORS[collapsed_group]}" fill-opacity="0.9"/>')
        svg.append(f'<text x="{left+240}" y="{center_y+5}" text-anchor="middle" fill="#ffffff" font-family="Arial" font-size="13" font-weight="700" stroke="#27313a" stroke-width="0.7" paint-order="stroke">{collapsed_group+1}</text>')
        svg.append(f'<text x="{left+272}" y="{center_y-4}" fill="#1f2a36" font-family="Arial" font-size="16" font-weight="700">{len(members)} taxa</text>')
        svg.append(f'<text x="{left+272}" y="{center_y+19}" fill="#5c6c7a" font-family="Arial" font-size="15">{record_percent:.1f}% of records</text>')

    svg.append('</svg>')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(svg) + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
