#!/usr/bin/env python3
"""Summarize final-ensemble gate weights across prespecified test strata."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import kruskal

from sensitivity_utils import boolean_series, record_keys


HEIGHT_ORDER = ["<2 m", "2–5 m", "5–10 m", ">=10 m"]
MODALITY_LABELS = {
    "tree_centered_naip": "NAIP",
    "tree_centered_chm": "CHM",
    "naip_chm_interaction": "NAIP–CHM interaction",
    "sentinel_phenology": "Sentinel-2 phenology",
    "raw_sentinel": "Raw Sentinel-2",
    "satellite_embedding": "Satellite embedding",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--predictions",
        type=Path,
        default=Path("outputs/reviewer_revision/final_pruned_k11_test_top5_ensemble/ensemble_predictions.npz"),
    )
    parser.add_argument(
        "--sensitivity-variables",
        type=Path,
        default=Path("outputs/reviewer_revision/final_pruned_sensitivity_table_with_taxon_provenance/final_test_sensitivity_variables.csv.gz"),
    )
    parser.add_argument(
        "--shard-root",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/reviewer_revision/final_gate_weight_stratified_analysis"),
    )
    parser.add_argument("--minimum-city-height-support", type=int, default=100)
    parser.add_argument(
        "--plot", action="store_true",
        help="Also render the optional summary figure; table generation does not require Matplotlib.",
    )
    return parser.parse_args()


def make_keys(city: np.ndarray, tree_id: np.ndarray, crop_index: np.ndarray) -> np.ndarray:
    """Backward-compatible wrapper around the shared record-key implementation."""
    return record_keys(city, tree_id, crop_index)


def extract_height(frame: pd.DataFrame, shard_root: Path, cache_path: Path | None = None) -> np.ndarray:
    """Extract CHM height and save a record-keyed cache when requested."""
    positions: dict[str, dict[tuple[int, int], int]] = {}
    for index, row in frame[["city_token", "tree_id", "crop_index"]].iterrows():
        positions.setdefault(str(row.city_token), {})[(int(row.tree_id), int(row.crop_index))] = int(index)
    heights = np.full(len(frame), np.nan, dtype=np.float32)
    matched = 0
    for city, wanted in positions.items():
        city_dir = shard_root / city
        for path in sorted(city_dir.glob("*_tree_centered_complete_inputs.npz")):
            with np.load(path, allow_pickle=True) as z:
                columns = np.asarray(z["tree_centered_naip_chm_structure_columns"], dtype=str)
                hits = np.flatnonzero(columns == "structure_tree_chm_p95_m")
                if len(hits) != 1:
                    raise ValueError(f"Could not uniquely locate vegetation maximum height in {path}")
                values = np.asarray(z["tree_centered_naip_chm_structure"][:, int(hits[0])], dtype=np.float32)
                for local, (tree, crop) in enumerate(zip(z["tree_id"], z["crop_index"])):
                    destination = wanted.get((int(tree), int(crop)))
                    if destination is not None:
                        heights[destination] = values[local]
                        matched += 1
        print(f"Height extraction: {city}; matched={matched:,}/{len(frame):,}", flush=True)
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache_path, record_key=frame.record_key.to_numpy(str), height_m=heights)
    if matched != len(frame) or np.isnan(heights).any():
        raise ValueError(f"Height extraction incomplete: matched={matched:,}; missing={int(np.isnan(heights).sum()):,}")
    return heights


def summarize(long: pd.DataFrame, strata: list[str]) -> pd.DataFrame:
    grouped = long.groupby(strata + ["modality", "modality_label"], observed=True, sort=False)["gate_weight"]
    result = grouped.agg(support="size", mean="mean", standard_deviation="std", median="median").reset_index()
    quantiles = grouped.quantile([0.25, 0.75]).unstack(-1).reset_index().rename(columns={0.25: "q25", 0.75: "q75"})
    result = result.merge(quantiles, on=strata + ["modality", "modality_label"], validate="one_to_one")
    result["interquartile_range"] = result.q75 - result.q25
    return result


def omnibus(long: pd.DataFrame, stratum: str) -> pd.DataFrame:
    rows = []
    for (modality, label), sub in long.groupby(["modality", "modality_label"], sort=False):
        arrays = [group.gate_weight.to_numpy() for _, group in sub.groupby(stratum, observed=True) if len(group)]
        statistic, p_value = kruskal(*arrays)
        n = sum(map(len, arrays))
        k = len(arrays)
        epsilon_squared = max(0.0, (float(statistic) - k + 1) / max(n - k, 1))
        means = sub.groupby(stratum, observed=True).gate_weight.mean()
        rows.append({
            "stratum": stratum,
            "modality": modality,
            "modality_label": label,
            "groups": k,
            "records": n,
            "kruskal_h": statistic,
            "p_value": p_value,
            "epsilon_squared": epsilon_squared,
            "maximum_minus_minimum_mean": float(means.max() - means.min()),
        })
    return pd.DataFrame(rows)


def plot_summary(height: pd.DataFrame, provenance: pd.DataFrame, output: Path) -> None:
    modalities = height.modality.drop_duplicates().tolist()
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8), constrained_layout=True)
    markers = ["o", "s", "^", "D", "v", "P"]
    for modality, marker in zip(modalities, markers):
        sub = height.loc[height.modality.eq(modality)].set_index("height_class").reindex(HEIGHT_ORDER)
        axes[0].errorbar(
            np.arange(len(HEIGHT_ORDER)), sub["median"],
            yerr=np.vstack([sub["median"] - sub["q25"], sub["q75"] - sub["median"]]),
            color="0.15", marker=marker, linestyle="-", linewidth=0.9, markersize=5,
            capsize=2, alpha=0.82, label=sub.modality_label.dropna().iloc[0],
        )
        subp = provenance.loc[provenance.modality.eq(modality)].copy()
        order = ["Core taxa", "Rare taxa"]
        subp = subp.set_index("taxon_provenance").reindex(order)
        offset = (modalities.index(modality) - (len(modalities) - 1) / 2) * 0.035
        axes[1].errorbar(
            np.arange(2) + offset, subp["median"],
            yerr=np.vstack([subp["median"] - subp["q25"], subp["q75"] - subp["median"]]),
            color="0.15", marker=marker, linestyle="none", markersize=5, capsize=2, alpha=0.82,
        )
    axes[0].set_xticks(range(len(HEIGHT_ORDER)), HEIGHT_ORDER)
    axes[1].set_xticks(range(2), ["Core taxa", "Rare taxa"])
    axes[0].set_title("A  Vegetation-height class", loc="left", fontweight="bold")
    axes[1].set_title("B  Taxon provenance", loc="left", fontweight="bold")
    for axis in axes:
        axis.set_ylabel("Gate weight (median and IQR)")
        axis.set_ylim(0, 0.75)
        axis.grid(axis="y", color="0.88", linewidth=0.7)
    axes[0].legend(frameon=False, fontsize=8, ncol=2)
    fig.savefig(output.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with np.load(args.predictions, allow_pickle=True) as z:
        frame = pd.DataFrame({
            "record_key": make_keys(z["city_token"], z["tree_id"], z["crop_index"]),
            "city_token": np.asarray(z["city_token"], dtype=str),
            "tree_id": np.asarray(z["tree_id"], dtype=np.int64),
            "crop_index": np.asarray(z["crop_index"], dtype=np.int64),
        })
        gates = np.asarray(z["gate_weights"], dtype=np.float64)
        modalities = np.asarray(z["modality_names"], dtype=str)
    sensitivity = pd.read_csv(args.sensitivity_variables)
    if sensitivity.record_key.duplicated().any() or frame.record_key.duplicated().any():
        raise ValueError("Duplicate record keys prevent one-to-one stratification")
    frame = frame.merge(
        sensitivity[["record_key", "qa_flag_low_vegetated_height_with_lidar", "frequency_category"]],
        on="record_key", how="left", validate="one_to_one",
    )
    if frame[["qa_flag_low_vegetated_height_with_lidar", "frequency_category"]].isna().any().any():
        raise ValueError("Missing sensitivity metadata after record-key join")
    height_cache = args.output_dir / "tree_chm_p95_height_by_record.npz"
    if height_cache.exists():
        with np.load(height_cache, allow_pickle=False) as cached:
            cached_keys = np.asarray(cached["record_key"], dtype=str)
            heights = np.asarray(cached["height_m"], dtype=np.float32)
        if not np.array_equal(cached_keys, frame.record_key.to_numpy(str)):
            raise ValueError("Cached height record keys do not match the current prediction order")
        if len(heights) != len(frame) or np.isnan(heights).any():
            raise ValueError("Cached height values are incomplete")
        frame["vegetation_height_m"] = heights
    else:
        frame["vegetation_height_m"] = extract_height(frame, args.shard_root, height_cache)
    frame["height_class"] = pd.cut(
        frame.vegetation_height_m,
        bins=[-np.inf, 2.0, 5.0, 10.0, np.inf],
        labels=HEIGHT_ORDER,
        right=False,
        ordered=True,
    )
    frame["taxon_provenance"] = np.where(frame.frequency_category.eq("core_ge500"), "Core taxa", "Rare taxa")
    frame["qa_screened"] = ~boolean_series(frame.qa_flag_low_vegetated_height_with_lidar)
    for index, modality in enumerate(modalities):
        frame[f"gate_{modality}"] = gates[:, index]
    screened = frame.loc[frame.qa_screened].copy()
    gate_columns = [f"gate_{name}" for name in modalities]
    long = screened.melt(
        id_vars=["record_key", "city_token", "height_class", "taxon_provenance"],
        value_vars=gate_columns,
        var_name="modality", value_name="gate_weight",
    )
    long["modality"] = long.modality.str.removeprefix("gate_")
    long["modality_label"] = long.modality.map(MODALITY_LABELS)

    overall = summarize(long.assign(stratum="All QA-screened records"), ["stratum"])
    by_height = summarize(long, ["height_class"])
    by_city = summarize(long, ["city_token"])
    by_city_height_all = summarize(long, ["city_token", "height_class"])
    by_city_height = by_city_height_all.loc[by_city_height_all.support.ge(args.minimum_city_height_support)].copy()
    by_provenance = summarize(long, ["taxon_provenance"])
    overall.to_csv(args.output_dir / "gate_weights_overall.csv", index=False)
    by_height.to_csv(args.output_dir / "gate_weights_by_height_class.csv", index=False)
    by_city.to_csv(args.output_dir / "gate_weights_by_city.csv", index=False)
    by_city_height_all.to_csv(args.output_dir / "gate_weights_by_city_and_height_all.csv", index=False)
    by_city_height.to_csv(args.output_dir / "gate_weights_by_city_and_height_support_ge100.csv", index=False)
    by_provenance.to_csv(args.output_dir / "gate_weights_by_taxon_provenance.csv", index=False)
    tests = pd.concat([
        omnibus(long, "height_class"), omnibus(long, "city_token"), omnibus(long, "taxon_provenance")
    ], ignore_index=True)
    tests.to_csv(args.output_dir / "gate_weight_heterogeneity_tests.csv", index=False)
    frame[[
        "record_key", "city_token", "vegetation_height_m", "height_class",
        "taxon_provenance", "qa_screened", *gate_columns,
    ]].to_csv(args.output_dir / "gate_weight_stratification_records.csv.gz", index=False, compression="gzip")
    if args.plot:
        plot_summary(by_height, by_provenance, args.output_dir / "gate_weight_height_and_provenance")
    metadata = {
        "source_predictions": str(args.predictions),
        "source_sensitivity_variables": str(args.sensitivity_variables),
        "test_records_total": int(len(frame)),
        "qa_screened_records": int(len(screened)),
        "qa_warning_records_excluded": int((~frame.qa_screened).sum()),
        "height_variable": "structure_tree_chm_p95_m",
        "height_classes": HEIGHT_ORDER,
        "minimum_city_height_support": args.minimum_city_height_support,
        "modalities": modalities.tolist(),
        "interpretation_warning": "Gate weights describe learned sample-dependent allocation and are not causal feature importance.",
    }
    (args.output_dir / "analysis_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))
    print("\nHeight-class summaries:\n", by_height.to_string(index=False))
    print("\nTaxon-provenance summaries:\n", by_provenance.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
