"""Prepare and plot taxon error against discriminator prototype margin.

The calculation uses the final pruned model's top-five independent-test
ensemble, QA screening, frozen group assignments, and development-only rare
centroids. Rendering is performed from the resulting taxon-level table.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from figure_config import (
    analytical_to_manuscript_zero_based,
    canonical_collapse_zero_based,
    colors,
)
from figure_data import boolean_series, save_figure_formats


K11_TO_COLLAPSED_K6 = canonical_collapse_zero_based()
K11_TO_MANUSCRIPT = analytical_to_manuscript_zero_based()
COLLAPSED_K6_COLORS = colors("collapsed_k6")
K11_MANUSCRIPT_COLORS = colors("k11_manuscript")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--predictions", type=Path, default=Path("outputs/reviewer_revision/final_pruned_k11_test_top5_ensemble/ensemble_predictions.npz"))
    p.add_argument("--core-partition", type=Path, default=Path("config/core_k11_partition.npz"))
    p.add_argument("--expanded-partition", type=Path, default=Path("config/final_k11_partition.npz"))
    p.add_argument("--rare-embeddings", type=Path, default=Path("outputs/reviewer_revision/revised_groupnorm_interaction_k11_rare_extension/rare_taxon_embeddings.npz"))
    p.add_argument("--shard-root", type=Path, required=True, help="Complete shard root, typically below $DATA_ROOT.")
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--dataset-label", default="final pruned height-screened test ensemble")
    p.add_argument("--group-space", choices=("collapsed_k6", "k11"), default="collapsed_k6")
    p.add_argument(
        "--exclude-height-warning",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Exclude records carrying the retained 2 m vegetation-height QA warning.",
    )
    return p.parse_args()


def normalized_rows(x: np.ndarray) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def prediction_metadata(pred: np.lib.npyio.NpzFile, shard_root: Path) -> tuple[np.ndarray, np.ndarray]:
    wanted = pd.DataFrame({
        "city_token": pred["city_token"].astype(str),
        "tree_id": pred["tree_id"].astype(np.int64),
        "crop_index": pred["crop_index"].astype(np.int64),
        "row_order": np.arange(len(pred["labels"]), dtype=np.int64),
    })
    pieces = []
    for path in sorted(shard_root.glob("*/*_metadata.csv")):
        city = path.name.split("_part", 1)[0].lower()
        target = wanted.loc[wanted.city_token == city, ["tree_id", "crop_index"]]
        if target.empty:
            continue
        frame = pd.read_csv(
            path,
            usecols=lambda c: c in {
                "tree_id", "crop_index", "scientific_name",
                "qa_flag_low_vegetated_height_with_lidar",
            },
            low_memory=False,
        )
        frame["city_token"] = city
        pieces.append(frame)
    lookup = pd.concat(pieces, ignore_index=True).drop_duplicates(["city_token", "tree_id", "crop_index"])
    joined = wanted.merge(lookup, on=["city_token", "tree_id", "crop_index"], how="left", validate="many_to_one").sort_values("row_order")
    if joined.scientific_name.isna().any():
        raise RuntimeError(f"Missing taxon metadata for {joined.scientific_name.isna().sum():,} prediction rows")
    sci = joined.scientific_name.astype(str).str.strip().str.replace(r"\s+", " ", regex=True)
    tokens = sci.str.extract(r"^([A-Z][A-Za-z-]{2,})(?:\s+([a-z][a-z-]{2,}))?", expand=True)
    genus = tokens[0].fillna("").str.lower()
    species = tokens[1].fillna("").str.lower()
    plausible_species = species.ne("") & ~species.isin({"sp", "spp", "species", "hybrid", "unknown"})
    taxa = (genus + "_" + species).where(plausible_species, genus).to_numpy()
    warning = boolean_series(joined["qa_flag_low_vegetated_height_with_lidar"]).to_numpy(dtype=bool)
    return taxa, warning


def taxon_centroids(
    core_path: Path, expanded_path: Path, rare_path: Path
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Build core centroids and development-only rare-extension centroids."""
    with np.load(core_path, allow_pickle=True) as core, np.load(
        expanded_path, allow_pickle=True
    ) as expanded, np.load(rare_path, allow_pickle=True) as rare:
        taxa = expanded["label_columns"].astype(str)
        assignments = expanded["assignments"].astype(np.int64)
        core_taxa = core["label_columns"].astype(str)
        core_count = int(np.asarray(expanded["core_taxon_count"]).reshape(-1)[0])
        if core_count != len(core_taxa) or not np.array_equal(taxa[:core_count], core_taxa):
            raise ValueError("Expanded partition does not begin with the frozen core-taxon order")
        centroid_by_taxon = {
            taxon: centroid
            for taxon, centroid in zip(core_taxa, core["full_validation_centroids"].astype(np.float64))
        }
        rare_taxa = rare["taxon"].astype(str)
        rare_split = rare["rare_split"].astype(str)
        rare_embeddings = rare["embeddings"].astype(np.float64)
    development = rare_split == "development"
    rare_frame = pd.DataFrame(
        {"taxon": rare_taxa[development], "row": np.flatnonzero(development)}
    )
    for taxon, positions in rare_frame.groupby("taxon", sort=False).indices.items():
        source_rows = rare_frame.iloc[np.asarray(positions)]["row"].to_numpy(dtype=np.int64)
        centroid_by_taxon[taxon] = rare_embeddings[source_rows].mean(axis=0)
    missing = [taxon for taxon in taxa if taxon not in centroid_by_taxon]
    if missing:
        raise ValueError(f"No development centroid available for {len(missing)} taxa")
    return taxa, assignments, np.stack([centroid_by_taxon[taxon] for taxon in taxa]), core_count


def rank_correlation(frame: pd.DataFrame) -> tuple[float, float]:
    """Calculate Spearman correlation and its large-sample two-sided P value."""
    x = frame.prototype_margin.rank(method="average").to_numpy(dtype=float)
    y = frame.misclassification_error.rank(method="average").to_numpy(dtype=float)
    x, y = x - x.mean(), y - y.mean()
    rho = float(np.sum(x * y) / np.sqrt(np.sum(x * x) * np.sum(y * y)))
    t_stat = rho * np.sqrt((len(frame) - 2) / max(1.0 - rho * rho, 1e-15))
    return rho, float(math.erfc(abs(t_stat) / math.sqrt(2.0)))


def render_taxon_margin(
    frame: pd.DataFrame,
    plot_colors: tuple[str, ...],
    group_count: int,
    dataset_label: str,
    output_dir: Path,
) -> tuple[float, float]:
    """Render the core/rare multipanel from a prepared taxon-level table."""
    rho, pvalue = rank_correlation(frame)
    core = frame[frame.frequency_category.eq("core")]
    rare = frame[frame.frequency_category.eq("rare extension")]
    subsets = (("core", core, "o", 22, 0), ("rare extension", rare, "+", 34, 1.35))
    figure, (core_axis, rare_axis) = plt.subplots(
        1, 2, figsize=(17.2, 7.6), dpi=220, sharex=True, sharey=True,
        gridspec_kw={"wspace": 0.12},
    )
    density_cmap = LinearSegmentedColormap.from_list("density", ["#F7F9FA", "#D4D4D4", "#8D8D8D"])
    hexbins = []
    for axis, (_, subset, marker, size, marker_width) in zip((core_axis, rare_axis), subsets):
        axis.set_facecolor("#F7F9FA")
        hexbins.append(axis.hexbin(
            subset.prototype_margin, subset.misclassification_error,
            gridsize=17, mincnt=1, cmap=density_cmap, linewidths=0, alpha=.45,
        ))
        for group, color in enumerate(plot_colors, start=1):
            values = subset[subset.group == group]
            axis.scatter(
                values.prototype_margin, values.misclassification_error,
                marker=marker, s=size, linewidths=marker_width,
                color=color, alpha=.92, zorder=3,
            )
            if len(values) >= 3 and values.prototype_margin.nunique() >= 2:
                x, y = values.prototype_margin.to_numpy(), values.misclassification_error.to_numpy()
                slope = np.sum((x - x.mean()) * (y - y.mean())) / np.sum((x - x.mean()) ** 2)
                intercept = y.mean() - slope * x.mean()
                line_x = np.linspace(values.prototype_margin.quantile(.03), values.prototype_margin.quantile(.97), 80)
                axis.plot(line_x, intercept + slope * line_x, color=color, linewidth=2.2,
                          alpha=.98, label=f"Group {group}", zorder=4)
        axis.set_xlabel("Prototype margin\n(nearest competing group − assigned group)", fontsize=13)
        axis.set_ylim(-.025, min(1.025, max(1.0, frame.misclassification_error.max() + .04)))
        axis.yaxis.set_major_formatter(lambda value, position: f"{value:.0%}")
        axis.grid(True, color="#D9E0E5", linewidth=.85, alpha=.85)
        axis.tick_params(labelsize=11, colors="#354354")
    core_axis.set_ylabel("Taxon misclassification error", fontsize=14)
    core_axis.set_title(f"A  Core discriminator taxa (n = {len(core):,})", loc="left", fontsize=14, fontweight="bold", color="#202B38", pad=12)
    rare_axis.set_title(f"B  Rare-extension taxa (n = {len(rare):,})", loc="left", fontsize=14, fontweight="bold", color="#202B38", pad=12)
    rare_axis.legend(ncol=2 if group_count == 6 else 3, frameon=True, facecolor="white",
                     edgecolor="#B7C1C9", fontsize=9.3 if group_count == 11 else 10.5,
                     loc="upper right")
    figure.suptitle("Taxon error declines with separation from competing groups",
                    x=0.075, ha="left", fontsize=20, fontweight="bold", color="#202B38", y=.985)
    figure.text(
        .075, .925,
        f"{len(frame):,} taxa · {dataset_label} · Spearman ρ = {rho:.3f}, P = {pvalue:.1e} · "
        f"medians: margin {frame.prototype_margin.median():.3f}, error {frame.misclassification_error.median():.1%}",
        fontsize=11.8, color="#607184", va="bottom",
    )
    density_max = max(float(collection.get_array().max()) for collection in hexbins)
    for collection in hexbins:
        collection.set_clim(1.0, density_max)
    colorbar = figure.colorbar(hexbins[-1], ax=[core_axis, rare_axis], pad=.015, fraction=.025)
    colorbar.set_label("Taxa per hexagon", fontsize=11, color="#607184")
    colorbar.ax.tick_params(labelsize=10, colors="#607184")
    figure.subplots_adjust(left=.075, right=.93, bottom=.11, top=.88, wspace=.12)
    save_figure_formats(figure, output_dir / "taxon_error_vs_prototype_margin", dpi=600)
    plt.close(figure)
    return rho, pvalue


def main() -> None:
    args = parse_args()
    if args.output_dir is None:
        args.output_dir = Path(
            "outputs/reviewer_revision/final_pruned_"
            f"{args.group_space}_test_taxon_margin_error"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pred = np.load(args.predictions, allow_pickle=True)
    taxa_for_rows, height_warning = prediction_metadata(pred, args.shard_root)
    probabilities = pred["probabilities"].astype(np.float64)
    all_test_records = len(probabilities)
    if args.exclude_height_warning:
        keep = ~height_warning
        taxa_for_rows = taxa_for_rows[keep]
        probabilities = probabilities[keep]
    print(
        f"2 m vegetation-height screen: retained={len(probabilities):,}/{all_test_records:,}; "
        f"excluded={all_test_records - len(probabilities):,}"
    )
    if args.group_space == "collapsed_k6":
        model_probabilities = np.column_stack(
            [probabilities[:, K11_TO_COLLAPSED_K6 == g].sum(axis=1) for g in range(6)]
        )
    else:
        model_probabilities = probabilities
    predicted = model_probabilities.argmax(axis=1)

    taxa, analytical_groups, centroids, core_count = taxon_centroids(
        args.core_partition, args.expanded_partition, args.rare_embeddings
    )
    groups = (
        K11_TO_COLLAPSED_K6[analytical_groups]
        if args.group_space == "collapsed_k6"
        else analytical_groups
    )
    display_groups = groups if args.group_space == "collapsed_k6" else K11_TO_MANUSCRIPT[groups]
    group_count = 6 if args.group_space == "collapsed_k6" else 11
    plot_colors = COLLAPSED_K6_COLORS if args.group_space == "collapsed_k6" else K11_MANUSCRIPT_COLORS
    taxon_to_group = dict(zip(taxa, groups))
    true_group = np.asarray([taxon_to_group[t] for t in taxa_for_rows], dtype=np.int64)
    rows = pd.DataFrame({"taxon": taxa_for_rows, "correct": predicted == true_group})
    performance = rows.groupby("taxon", sort=False).correct.agg(["size", "mean"]).rename(columns={"size": "support", "mean": "accuracy"})

    unit = normalized_rows(centroids)
    prototypes = np.stack([unit[groups == g].mean(axis=0) for g in range(group_count)])
    prototypes = normalized_rows(prototypes)
    # Avoid a Windows BLAS/DLL fault observed in the lightweight UMAP env.
    distance = 1.0 - np.sum(unit[:, None, :] * prototypes[None, :, :], axis=2)
    own = distance[np.arange(len(taxa)), groups]
    competing = distance.copy()
    competing[np.arange(len(taxa)), groups] = np.inf
    margin = competing.min(axis=1) - own

    out = pd.DataFrame({"taxon": taxa, "group": display_groups + 1, "prototype_margin": margin}).join(performance, on="taxon")
    out["misclassification_error"] = 1.0 - out.accuracy
    out["frequency_category"] = np.where(
        np.arange(len(out)) < core_count, "core", "rare extension"
    )
    out = out.dropna(subset=["accuracy"]).sort_values(["group", "prototype_margin"])
    out.to_csv(args.output_dir / "taxon_error_vs_prototype_margin.csv", index=False)
    pd.DataFrame(
        [{
            "all_test_records": all_test_records,
            "height_screened_records": len(probabilities),
            "excluded_warning_records": all_test_records - len(probabilities),
            "group_space": args.group_space,
        }]
    ).to_csv(args.output_dir / "prototype_margin_height_screen_audit.csv", index=False)

    rho, pvalue = render_taxon_margin(
        out, plot_colors, group_count, args.dataset_label, args.output_dir
    )
    print(f"Wrote {len(out):,} taxa; rho={rho:.4f}; p={pvalue:.3g}")


if __name__ == "__main__":
    main()
