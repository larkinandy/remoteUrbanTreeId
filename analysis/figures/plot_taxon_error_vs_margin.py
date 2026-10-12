"""Plot taxon-level classification error against discriminator prototype margin.

The defaults use the final pruned model's top-five test ensemble.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib
matplotlib.__version__ = "3.x"
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


COLORS = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#332288"]
K11_TO_K6 = np.asarray([0, 0, 1, 1, 2, 2, 3, 4, 2, 5, 1], dtype=np.int64)
# Analytical k=11 labels are reordered for manuscript display so groups that
# collapse together are sequential and share related color families.
K11_TO_MANUSCRIPT = np.asarray([0, 1, 2, 3, 5, 6, 8, 9, 7, 10, 4], dtype=np.int64)
K11_MANUSCRIPT_COLORS = [
    "#005A8D", "#56B4E9", "#A96500", "#E69F00", "#F2C14E",
    "#006D50", "#009E73", "#66C2A5", "#D55E00", "#CC79A7", "#332288",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--predictions", type=Path, default=Path("outputs/reviewer_revision/final_pruned_k11_test_top5_ensemble/ensemble_predictions.npz"))
    p.add_argument("--core-partition", type=Path, default=Path("outputs/reviewer_revision/revised_groupnorm_interaction_fixed_k11_consensus_b500/fixed_k11_consensus_partition.npz"))
    p.add_argument("--expanded-partition", type=Path, default=Path("config/final_k11_partition.npz"))
    p.add_argument("--rare-embeddings", type=Path, default=Path("outputs/reviewer_revision/revised_groupnorm_interaction_k11_rare_extension/rare_taxon_embeddings.npz"))
    p.add_argument("--shard-root", type=Path, default=Path(r"H:\TreeCenteredModelInputs\tree_centered_complete_sharded100k_prism_daily3_clean"))
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--dataset-label", default="final pruned height-screened test ensemble")
    p.add_argument("--group-space", choices=("k6", "k11"), default="k6")
    p.add_argument(
        "--exclude-height-warning",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Exclude records carrying the retained 2 m vegetation-height QA warning.",
    )
    return p.parse_args()


def normalized_rows(x: np.ndarray) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def as_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(False)
    return series.fillna(False).astype(str).str.strip().str.lower().isin({"1", "true", "t", "yes", "y"})


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
    warning = as_bool(joined["qa_flag_low_vegetated_height_with_lidar"]).to_numpy(dtype=bool)
    return taxa, warning


def taxon_centroids(core_path: Path, expanded_path: Path, rare_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    core = np.load(core_path, allow_pickle=True)
    expanded = np.load(expanded_path, allow_pickle=True)
    rare = np.load(rare_path, allow_pickle=True)
    taxa = expanded["label_columns"].astype(str)
    assignments = expanded["assignments"].astype(np.int64)
    core_taxa = core["label_columns"].astype(str)
    centroid_by_taxon = {t: v for t, v in zip(core_taxa, core["full_validation_centroids"].astype(np.float64))}
    rare_frame = pd.DataFrame({"taxon": rare["taxon"].astype(str), "row": np.arange(len(rare["taxon"]))})
    embeddings = rare["embeddings"].astype(np.float64)
    for taxon, idx in rare_frame.groupby("taxon", sort=False).indices.items():
        centroid_by_taxon[taxon] = embeddings[np.asarray(idx)].mean(axis=0)
    centroids = np.stack([centroid_by_taxon[t] for t in taxa])
    return taxa, assignments, centroids


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
    if args.group_space == "k6":
        model_probabilities = np.column_stack(
            [probabilities[:, K11_TO_K6 == g].sum(axis=1) for g in range(6)]
        )
    else:
        model_probabilities = probabilities
    predicted = model_probabilities.argmax(axis=1)

    taxa, analytical_groups, centroids = taxon_centroids(args.core_partition, args.expanded_partition, args.rare_embeddings)
    groups = K11_TO_K6[analytical_groups] if args.group_space == "k6" else analytical_groups
    display_groups = groups if args.group_space == "k6" else K11_TO_MANUSCRIPT[groups]
    group_count = 6 if args.group_space == "k6" else 11
    colors = COLORS if args.group_space == "k6" else K11_MANUSCRIPT_COLORS
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
    out["frequency_category"] = np.where(np.arange(len(out)) < 280, "core", "rare extension")
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

    rx = out.prototype_margin.rank(method="average").to_numpy(dtype=float).copy()
    ry = out.misclassification_error.rank(method="average").to_numpy(dtype=float).copy()
    rx -= rx.mean()
    ry -= ry.mean()
    rho = float(np.sum(rx * ry) / np.sqrt(np.sum(rx * rx) * np.sum(ry * ry)))
    t_stat = rho * np.sqrt((len(out) - 2) / max(1.0 - rho * rho, 1e-15))
    # Normal approximation is effectively identical here at several hundred taxa.
    pvalue = float(math.erfc(abs(t_stat) / math.sqrt(2.0)))
    core = out[out.frequency_category.eq("core")]
    rare = out[out.frequency_category.eq("rare extension")]
    subsets = (("core", core, "o", 22, 0), ("rare extension", rare, "+", 34, 1.35))
    fig, (core_ax, rare_ax) = plt.subplots(
        1, 2, figsize=(17.2, 7.6), dpi=220, sharex=True, sharey=True,
        gridspec_kw={"wspace": 0.12},
    )
    for ax in (core_ax, rare_ax):
        ax.set_facecolor("#F7F9FA")
    density_cmap = LinearSegmentedColormap.from_list("density", ["#F7F9FA", "#D4D4D4", "#8D8D8D"])
    hexbins = []
    for ax, (_, subset, marker, size, marker_width) in zip((core_ax, rare_ax), subsets):
        hb = ax.hexbin(
            subset.prototype_margin, subset.misclassification_error,
            gridsize=17, mincnt=1, cmap=density_cmap, linewidths=0, alpha=.45,
        )
        hexbins.append(hb)
        for g, color in enumerate(colors, start=1):
            d = subset[subset.group == g]
            ax.scatter(
                d.prototype_margin, d.misclassification_error,
                marker=marker, s=size, linewidths=marker_width,
                color=color, alpha=.92, zorder=3,
            )
            if len(d) >= 3 and d.prototype_margin.nunique() >= 2:
                x = d.prototype_margin.to_numpy()
                y = d.misclassification_error.to_numpy()
                slope = np.sum((x - x.mean()) * (y - y.mean())) / np.sum((x - x.mean()) ** 2)
                intercept = y.mean() - slope * x.mean()
                xx = np.linspace(d.prototype_margin.quantile(.03), d.prototype_margin.quantile(.97), 80)
                ax.plot(
                    xx, intercept + slope * xx, color=color, linewidth=2.2,
                    alpha=.98, label=f"Group {g}", zorder=4,
                )
    for ax in (core_ax, rare_ax):
        ax.set_xlabel("Prototype margin\n(nearest competing group − assigned group)", fontsize=13)
        ax.set_ylim(-.025, min(1.025, max(1.0, out.misclassification_error.max() + .04)))
        ax.yaxis.set_major_formatter(lambda x, pos: f"{x:.0%}")
        ax.grid(True, color="#D9E0E5", linewidth=.85, alpha=.85)
        ax.tick_params(labelsize=11, colors="#354354")
    core_ax.set_ylabel("Taxon misclassification error", fontsize=14)
    core_ax.set_title(f"A  Core discriminator taxa (n = {len(core):,})", loc="left", fontsize=14, fontweight="bold", color="#202B38", pad=12)
    rare_ax.set_title(f"B  Rare-extension taxa (n = {len(rare):,})", loc="left", fontsize=14, fontweight="bold", color="#202B38", pad=12)
    rare_ax.legend(
        ncol=2 if group_count == 6 else 3, frameon=True, facecolor="white",
        edgecolor="#B7C1C9", fontsize=9.3 if group_count == 11 else 10.5,
        loc="upper right",
    )
    fig.suptitle(
        "Taxon error declines with separation from competing groups",
        x=0.075, ha="left", fontsize=20, fontweight="bold", color="#202B38", y=.985,
    )
    subtitle = (f"{len(out):,} taxa · {args.dataset_label} · Spearman ρ = {rho:.3f}, P = {pvalue:.1e} · "
                f"medians: margin {out.prototype_margin.median():.3f}, error {out.misclassification_error.median():.1%}")
    fig.text(.075, .925, subtitle, fontsize=11.8, color="#607184", va="bottom")
    density_max = max(float(collection.get_array().max()) for collection in hexbins)
    for collection in hexbins:
        collection.set_clim(1.0, density_max)
    cbar = fig.colorbar(hexbins[-1], ax=[core_ax, rare_ax], pad=.015, fraction=.025)
    cbar.set_label("Taxa per hexagon", fontsize=11, color="#607184")
    cbar.ax.tick_params(labelsize=10, colors="#607184")
    fig.tight_layout()
    for suffix in ("png", "svg", "pdf"):
        fig.savefig(args.output_dir / f"taxon_error_vs_prototype_margin.{suffix}", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Wrote {len(out):,} taxa; rho={rho:.4f}; p={pvalue:.3g}")


if __name__ == "__main__":
    main()
