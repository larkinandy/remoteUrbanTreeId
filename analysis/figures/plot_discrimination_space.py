#!/usr/bin/env python3
"""Render a validated k11 or collapsed_k6 manuscript discrimination-space SVG."""

from __future__ import annotations

import argparse
import csv
import html
from pathlib import Path

import numpy as np

from figure_config import canonical_collapse_zero_based, colors


# The k11 palette is nested within collapsed_k6:
# (1,2)->blue; (3,4,11)->orange; (5,6,9)->green; 7->vermillion;
# 8->magenta; 10->purple. This makes the relationship between the two
# reporting levels visually explicit while retaining distinguishable shades.
K11_COLORS = colors("k11_analytical")
K11_MANUSCRIPT_COLORS = colors("k11_manuscript")
COLLAPSED_K6_COLORS = colors("collapsed_k6")
# Manuscript examples selected to make four aspects of the partition visible:
# one representative of every k11 group; Acer spanning groups 3, 4, 6, and 7;
# two difficult k11 boundary taxa that improve after collapsed_k6; and two
# taxa that remain difficult boundary cases after collapse.
LABELS = {
    # Representative taxa for manuscript k11 groups 1--11.
    "washingtonia_robusta", "magnolia_grandiflora", "acer_platanoides",
    "prunus_cerasifera", "tilia_cordata", "gleditsia_triacanthos",
    "liquidambar_styraciflua", "quercus_palustris", "lagerstroemia_indica",
    "platanus_acerifolia", "pinus_nigra",
    # Additional Acer examples demonstrating that groups are not taxonomic.
    "acer_palmatum", "acer_negundo", "acer_saccharinum",
    # Difficult under k11 but substantially improved by collapsed_k6.
    "acer_macrophyllum", "quercus_macrocarpa",
    # Difficult boundary cases that remain under collapsed_k6.
    "koelreuteria_paniculata", "populus_tremuloides",
}

# Fixed offsets keep edge labels inside the panel and separate the two
# lower-left boundary examples. Remaining labels use the group-wise offsets.
LABEL_OFFSETS = {
    "washingtonia_robusta": (14, -18),
    "pinus_nigra": (14, -20),
    "populus_tremuloides": (14, 28),
}


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coordinates", type=Path, default=root / "analysis/figures/provenance/core_taxon_umap_coordinates.csv")
    parser.add_argument("--partition", type=Path, default=root / "config/core_k11_partition.npz")
    parser.add_argument("--output-dir", type=Path, default=root / "outputs/manuscript_figures")
    parser.add_argument("--output-name", default="k11_discrimination_space_manuscript.svg")
    parser.add_argument(
        "--group-space", choices=("k11", "collapsed_k6"), default="k11",
        help="Reporting label space; collapsed_k6 uses config/final_grouping.json.",
    )
    parser.add_argument("--title", default="Discriminator separation space and assigned groups")
    parser.add_argument(
        "--subtitle",
        default="UMAP of core-taxon centroids; point area scales with validation support.",
    )
    parser.add_argument("--rare-embeddings", type=Path, default=None)
    parser.add_argument("--expanded-partition", type=Path, default=None)
    parser.add_argument("--rare-neighbors", type=int, default=8)
    parser.add_argument(
        "--group-order",
        default=None,
        help=(
            "Optional comma-separated old one-based group identifiers in the desired new order. "
            "Assignments are relabeled for visualization only."
        ),
    )
    parser.add_argument(
        "--hybrid-legend", action="store_true",
        help="Draw the canonical k11-to-collapsed_k6 hierarchy without changing assignments.",
    )
    return parser.parse_args()


def pretty_taxon(value: str) -> str:
    return " ".join(part.capitalize() if index == 0 else part for index, part in enumerate(value.split("_")))


def append_group_legend(
    svg: list[str], rows: list[dict], palette: tuple[str, ...],
    group_count: int, hybrid_collapse: np.ndarray | None,
) -> None:
    """Append either the simple group legend or canonical hierarchy legend."""
    svg.append(
        f'<text x="1010" y="150" fill="#1f2a36" font-family="Arial" font-size="23" '
        f'font-weight="700">{"Group hierarchy" if hybrid_collapse is not None else "Assigned group"}</text>'
    )
    total_records = sum(int(row["support"]) for row in rows)
    if hybrid_collapse is None:
        legend_step = 55 if group_count > 7 else 65
        for group in range(group_count):
            y = 192 + legend_step * group
            members = [row for row in rows if row["group"] == group]
            percent = 100.0 * sum(int(row["support"]) for row in members) / total_records
            svg.append(f'<circle cx="1025" cy="{y-6}" r="12" fill="{palette[group]}" fill-opacity="0.82"/>')
            svg.append(f'<text x="1025" y="{y-1}" text-anchor="middle" fill="#ffffff" font-family="Arial" font-size="12" font-weight="700" stroke="#27313a" stroke-width="0.7" paint-order="stroke">{group+1}</text>')
            svg.append(f'<text x="1052" y="{y}" fill="#5c6c7a" font-family="Arial" font-size="15">{len(members)} taxa · {percent:.1f}% of records</text>')
        return
    svg.append('<text x="1022" y="174" fill="#7b8994" font-family="Arial" font-size="13" font-weight="700">k11 groups</text>')
    svg.append('<text x="1190" y="174" fill="#7b8994" font-family="Arial" font-size="13" font-weight="700">collapsed_k6 group</text>')
    for collapsed_group in range(6):
        source_groups = np.flatnonzero(hybrid_collapse == collapsed_group).tolist()
        top, center_y = 188 + 84 * collapsed_group, 222 + 84 * collapsed_group
        members = [row for row in rows if row["group"] in source_groups]
        percent = 100.0 * sum(int(row["support"]) for row in members) / total_records
        svg.append(f'<rect x="1005" y="{top}" width="575" height="68" rx="9" fill="#fbfcfd" stroke="#d5dee5" stroke-width="1.2"/>')
        for source_index, source_group in enumerate(source_groups):
            cx = 1030 + 37 * source_index
            svg.append(f'<circle cx="{cx}" cy="{center_y}" r="12" fill="{palette[source_group]}" fill-opacity="0.88"/>')
            svg.append(f'<text x="{cx}" y="{center_y+5}" text-anchor="middle" fill="#ffffff" font-family="Arial" font-size="12" font-weight="700" stroke="#27313a" stroke-width="0.7" paint-order="stroke">{source_group+1}</text>')
        svg.append(f'<line x1="1142" y1="{center_y}" x2="1180" y2="{center_y}" stroke="#7b8994" stroke-width="1.8" marker-end="url(#legend-arrow)"/>')
        svg.append(f'<circle cx="1205" cy="{center_y}" r="14" fill="{COLLAPSED_K6_COLORS[collapsed_group]}" fill-opacity="0.88"/>')
        svg.append(f'<text x="1205" y="{center_y+5}" text-anchor="middle" fill="#ffffff" font-family="Arial" font-size="13" font-weight="700" stroke="#27313a" stroke-width="0.7" paint-order="stroke">{collapsed_group+1}</text>')
        svg.append(f'<text x="1232" y="{center_y-3}" fill="#1f2a36" font-family="Arial" font-size="15" font-weight="700">{len(members)} taxa</text>')
        svg.append(f'<text x="1232" y="{center_y+18}" fill="#5c6c7a" font-family="Arial" font-size="14">{percent:.1f}% of records</text>')


def append_point_legend(svg: list[str], group_count: int) -> None:
    """Append point-size, rare-taxon, and UMAP interpretation keys."""
    title_y = 820 if group_count > 7 else 700
    circle_y, label_y, interpretation_y = title_y + 48, title_y + 90, title_y + 145
    svg.extend([
        f'<text x="1010" y="{title_y}" fill="#1f2a36" font-family="Arial" font-size="20" font-weight="700">Point size</text>',
        f'<circle cx="1040" cy="{circle_y}" r="5" fill="#7b8994" fill-opacity="0.55"/><circle cx="1125" cy="{circle_y}" r="11" fill="#7b8994" fill-opacity="0.55"/><circle cx="1225" cy="{circle_y}" r="17" fill="#7b8994" fill-opacity="0.55"/>',
        f'<text x="1010" y="{label_y}" fill="#5c6c7a" font-family="Arial" font-size="15">fewer records</text><text x="1190" y="{label_y}" fill="#5c6c7a" font-family="Arial" font-size="15">more records</text>',
        f'<line x1="1035" y1="{label_y + 32}" x2="1047" y2="{label_y + 32}" stroke="#596773" stroke-width="2.4" stroke-linecap="round"/><line x1="1041" y1="{label_y + 26}" x2="1041" y2="{label_y + 38}" stroke="#596773" stroke-width="2.4" stroke-linecap="round"/>',
        f'<text x="1060" y="{label_y + 38}" fill="#5c6c7a" font-family="Arial" font-size="15">rare-extension taxon (fixed size)</text>',
        f'<text x="1010" y="{interpretation_y}" fill="#1f2a36" font-family="Arial" font-size="20" font-weight="700">Interpretation</text>',
        f'<text x="1010" y="{interpretation_y + 32}" fill="#5c6c7a" font-family="Arial" font-size="15">Nearby centroids indicate taxa</text><text x="1010" y="{interpretation_y + 54}" fill="#5c6c7a" font-family="Arial" font-size="15">with similar learned signatures.</text>',
        f'<text x="1010" y="{interpretation_y + 89}" fill="#5c6c7a" font-family="Arial" font-size="15">UMAP preserves local neighborhoods;</text><text x="1010" y="{interpretation_y + 111}" fill="#5c6c7a" font-family="Arial" font-size="15">axis values have no physical units.</text>',
    ])


def main() -> None:
    args = parse_args()
    if args.group_space == "collapsed_k6" and args.group_order:
        raise ValueError("--group-order is display-only for k11 and cannot be combined with collapsed_k6")
    if args.group_space == "collapsed_k6" and args.hybrid_legend:
        raise ValueError("--hybrid-legend requires the k11 group space")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with args.coordinates.open(newline="", encoding="utf-8") as handle:
        coordinates = list(csv.DictReader(handle))
    with np.load(args.partition, allow_pickle=True) as data:
        taxa = data["label_columns"].astype(str)
        assignment = data["assignments"].astype(int)
        support = data["validation_counts"].astype(int)
        full_validation_centroids = (
            np.asarray(data["full_validation_centroids"], dtype=np.float32)
            if "full_validation_centroids" in data.files else None
        )
    group_relabel = None
    if args.group_order:
        old_order = np.asarray([int(value.strip()) - 1 for value in args.group_order.split(",")], dtype=int)
        original_group_count = int(assignment.max()) + 1
        if sorted(old_order.tolist()) != list(range(original_group_count)):
            raise ValueError("--group-order must contain every original group exactly once")
        group_relabel = np.empty(original_group_count, dtype=int)
        group_relabel[old_order] = np.arange(original_group_count, dtype=int)
        assignment = group_relabel[assignment]
    collapse = None
    if args.group_space == "collapsed_k6":
        collapse = canonical_collapse_zero_based()
        original_group_count = int(assignment.max()) + 1
        if len(collapse) != original_group_count:
            raise ValueError(
                f"Canonical collapse has {len(collapse)} entries but the partition has "
                f"{original_group_count} original groups"
            )
        assignment = collapse[assignment]
    lookup = {taxon: index for index, taxon in enumerate(taxa)}
    group_count = int(assignment.max()) + 1
    colors = (
        COLLAPSED_K6_COLORS if collapse is not None and group_count == 6
        else K11_MANUSCRIPT_COLORS if group_relabel is not None and group_count == 11
        else K11_COLORS
    )
    if group_count > len(colors):
        raise ValueError(f"Only {len(colors)} colors are configured for {group_count} groups")
    hybrid_collapse = None
    if args.hybrid_legend:
        hybrid_collapse = canonical_collapse_zero_based()
        if group_relabel is not None:
            displayed = np.empty_like(hybrid_collapse)
            displayed[group_relabel] = hybrid_collapse
            hybrid_collapse = displayed
        if group_count != 11 or len(hybrid_collapse) != group_count:
            raise ValueError("--hybrid-legend requires the frozen k11 partition")
    rows = []
    for row in coordinates:
        taxon = row["taxon"]
        index = lookup[taxon]
        rows.append({"taxon": taxon, "group": int(assignment[index]), "support": int(support[index]),
                     "x": float(row["umap_1"]), "y": float(row["umap_2"]), "taxon_type": "core"})

    rare_rows = []
    if args.rare_embeddings is not None or args.expanded_partition is not None:
        if args.rare_embeddings is None or args.expanded_partition is None:
            raise ValueError("--rare-embeddings and --expanded-partition must be supplied together")
        if full_validation_centroids is None:
            raise ValueError("The core partition must contain full_validation_centroids for rare-taxon projection")
        core_centroids = full_validation_centroids.copy()
        core_centroids /= np.maximum(np.linalg.norm(core_centroids, axis=1, keepdims=True), 1.0e-12)
        coordinate_lookup = {row["taxon"]: np.asarray([row["x"], row["y"]], dtype=float) for row in rows}
        core_xy = np.stack([coordinate_lookup[taxon] for taxon in taxa])
        with np.load(args.rare_embeddings, allow_pickle=True) as rare_data:
            rare_embedding = np.asarray(rare_data["embeddings"], dtype=np.float32)
            rare_taxon = rare_data["taxon"].astype(str)
            rare_split = rare_data["rare_split"].astype(str)
        with np.load(args.expanded_partition, allow_pickle=True) as expanded:
            expanded_taxa = expanded["label_columns"].astype(str)
            expanded_assignment = expanded["assignments"].astype(int)
            core_count = int(np.asarray(expanded["core_taxon_count"]).reshape(-1)[0])
            rare_development_counts = expanded["rare_development_counts"].astype(int)
            rare_test_counts = expanded["rare_test_counts"].astype(int)
        if collapse is not None:
            expanded_assignment = collapse[expanded_assignment]
        elif group_relabel is not None:
            expanded_assignment = group_relabel[expanded_assignment]
        expanded_lookup = {taxon: index for index, taxon in enumerate(expanded_taxa)}
        rare_names = expanded_taxa[core_count:]
        for rare_index, taxon in enumerate(rare_names):
            mask = (rare_taxon == taxon) & (rare_split == "development")
            values = rare_embedding[mask]
            if not len(values):
                continue
            values /= np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1.0e-12)
            centroid = values.mean(axis=0)
            centroid /= max(float(np.linalg.norm(centroid)), 1.0e-12)
            similarity = core_centroids @ centroid
            neighbor_count = min(max(1, int(args.rare_neighbors)), len(similarity))
            nearest = np.argpartition(similarity, -neighbor_count)[-neighbor_count:]
            local = similarity[nearest]
            weights = np.exp((local - local.max()) / 0.03)
            weights /= weights.sum()
            xy = (core_xy[nearest] * weights[:, None]).sum(axis=0)
            expanded_index = expanded_lookup[taxon]
            rare_rows.append({
                "taxon": taxon,
                "group": int(expanded_assignment[expanded_index]),
                "support": int(rare_development_counts[rare_index] + rare_test_counts[rare_index]),
                "x": float(xy[0]),
                "y": float(xy[1]),
                "taxon_type": "rare_extension",
            })

    plot_x, plot_y, plot_size = 90.0, 120.0, 850.0
    xs, ys = np.asarray([r["x"] for r in rows]), np.asarray([r["y"] for r in rows])
    # UMAP axes have arbitrary units. Scale each dimension independently into
    # the same square physical plotting area so neither dimension is visually
    # compressed by a larger numerical range on the other axis.
    xpad = (xs.max() - xs.min()) * 0.04
    ypad = (ys.max() - ys.min()) * 0.04
    xmin, xmax = xs.min() - xpad, xs.max() + xpad
    ymin, ymax = ys.min() - ypad, ys.max() + ypad
    def sx(value: float) -> float:
        return plot_x + (value - xmin) / (xmax - xmin) * plot_size

    def sy(value: float) -> float:
        return plot_y + plot_size - (value - ymin) / (ymax - ymin) * plot_size

    maximum = max(r["support"] for r in rows)

    def radius(value: int) -> float:
        return 3.1 + 13.9 * np.sqrt(value / maximum)

    svg = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="1680" height="1100" viewBox="0 0 1680 1100" role="img" aria-labelledby="title desc">',
        f'<title id="title">{html.escape(args.title)}</title>',
        f'<desc id="desc">UMAP projection of {len(rows)} core-taxon centroids colored by {group_count} assigned groups.</desc>',
        '<rect width="1680" height="1100" fill="#ffffff"/>',
        '<defs><marker id="legend-arrow" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto"><path d="M0,0 L7,3.5 L0,7 Z" fill="#7b8994"/></marker></defs>',
        f'<text x="70" y="58" fill="#1f2a36" font-family="Arial" font-size="38" font-weight="700">{html.escape(args.title)}</text>',
        f'<text x="70" y="92" fill="#5c6c7a" font-family="Arial" font-size="19">{html.escape(args.subtitle)}</text>',
        f'<rect x="{plot_x}" y="{plot_y}" width="{plot_size}" height="{plot_size}" rx="10" fill="#fafbfc" stroke="#cbd5dc" stroke-width="1.5"/>',
    ]
    for row in sorted(rows, key=lambda item: item["support"], reverse=True):
        selected = row["taxon"] in LABELS
        stroke = "#27313a" if selected else "#ffffff"
        stroke_width = 2.4 if selected else 1.5
        svg.append(f'<circle cx="{sx(row["x"]):.2f}" cy="{sy(row["y"]):.2f}" r="{radius(row["support"]):.2f}" fill="{colors[row["group"]]}" fill-opacity="0.74" stroke="{stroke}" stroke-width="{stroke_width}"/>')
    for row in rare_rows:
        px, py, arm = sx(row["x"]), sy(row["y"]), 4.2
        color = colors[row["group"]]
        svg.append(f'<line x1="{px-arm:.2f}" y1="{py:.2f}" x2="{px+arm:.2f}" y2="{py:.2f}" stroke="#ffffff" stroke-width="3.8" stroke-linecap="round"/>')
        svg.append(f'<line x1="{px:.2f}" y1="{py-arm:.2f}" x2="{px:.2f}" y2="{py+arm:.2f}" stroke="#ffffff" stroke-width="3.8" stroke-linecap="round"/>')
        svg.append(f'<line x1="{px-arm:.2f}" y1="{py:.2f}" x2="{px+arm:.2f}" y2="{py:.2f}" stroke="{color}" stroke-width="2.0" stroke-linecap="round"/>')
        svg.append(f'<line x1="{px:.2f}" y1="{py-arm:.2f}" x2="{px:.2f}" y2="{py+arm:.2f}" stroke="{color}" stroke-width="2.0" stroke-linecap="round"/>')

    label_rows = [row for row in rows if row["taxon"] in LABELS]
    label_rows.sort(key=lambda item: (item["group"], -item["support"]))
    group_offsets = {group: 0 for group in range(group_count)}
    for row in label_rows:
        default_direction = 1 if group_offsets[row["group"]] % 2 == 0 else -1
        default_vertical = -20 if group_offsets[row["group"]] % 3 else 20
        group_offsets[row["group"]] += 1
        px, py = sx(row["x"]), sy(row["y"])
        dx, vertical = LABEL_OFFSETS.get(
            row["taxon"], (12 * default_direction, default_vertical)
        )
        direction = 1 if dx >= 0 else -1
        tx, ty = px + dx, py + vertical
        anchor = "start" if direction > 0 else "end"
        name = html.escape(pretty_taxon(row["taxon"]))
        svg.append(f'<line x1="{px:.2f}" y1="{py:.2f}" x2="{tx:.2f}" y2="{ty:.2f}" stroke="#7b8994" stroke-width="1"/>')
        svg.append(f'<text x="{tx + 4 * direction:.2f}" y="{ty + 5:.2f}" text-anchor="{anchor}" fill="#1f2a36" font-family="Arial" font-size="14" font-style="italic" stroke="#ffffff" stroke-width="3.5" paint-order="stroke" stroke-linejoin="round">{name}</text>')

    svg.extend([
        '<text x="515" y="1012" text-anchor="middle" fill="#5c6c7a" font-family="Arial" font-size="17">UMAP 1</text>',
        '<text x="42" y="545" text-anchor="middle" fill="#5c6c7a" font-family="Arial" font-size="17" transform="rotate(-90 42 545)">UMAP 2</text>',
    ])
    append_group_legend(svg, rows, colors, group_count, hybrid_collapse)
    append_point_legend(svg, group_count)
    svg.append('</svg>')
    output = args.output_dir / args.output_name
    output.write_text("\n".join(svg) + "\n", encoding="utf-8")
    csv_output = output.with_suffix(".csv")
    with csv_output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["taxon", "group", "taxon_type", "records", "umap_1", "umap_2"])
        writer.writeheader()
        for row in rows + rare_rows:
            writer.writerow({"taxon": row["taxon"], "group": row["group"] + 1,
                             "taxon_type": row["taxon_type"], "records": row["support"],
                             "umap_1": row["x"], "umap_2": row["y"]})
    print(output)


if __name__ == "__main__":
    main()
