#!/usr/bin/env python3
"""Summarize final-ensemble gate weights across prespecified test strata.

The analysis joins sample-dependent gate weights to QA, vegetation-height,
city, and taxon-provenance strata by stable record key. It writes compact
statistical tables and an optional publication figure. Gate weights describe
model allocation behavior and must not be interpreted as causal importance.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kruskal

from sensitivity_utils import (
    boolean_series,
    file_sha256,
    load_configuration,
    merge_one_to_one,
    record_keys,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "analysis" / "sensitivity" / "sensitivity_config.json"
MODALITY_LABELS = {
    "tree_centered_naip": "NAIP",
    "tree_centered_chm": "CHM",
    "naip_chm_interaction": "NAIP-CHM interaction",
    "sentinel_phenology": "Sentinel-2 phenology",
    "raw_sentinel": "Raw Sentinel-2",
    "satellite_embedding": "Satellite embedding",
}


def parse_args() -> argparse.Namespace:
    """Parse explicit input artifacts, configuration, and output options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True, help="Final ensemble NPZ containing gate weights.")
    parser.add_argument("--sensitivity-variables", type=Path, required=True, help="Record-keyed QA and provenance CSV.")
    parser.add_argument("--shard-root", type=Path, required=True, help="Root containing complete city shard NPZ files.")
    parser.add_argument("--output-dir", type=Path, required=True, help="External directory for generated tables, cache, and figures.")
    parser.add_argument("--sensitivity-config", type=Path, default=DEFAULT_CONFIG, help="Frozen sensitivity and gate-analysis configuration JSON.")
    parser.add_argument("--minimum-city-height-support", type=int, help="Override minimum records per city-height summary cell.")
    parser.add_argument("--plot", action="store_true", help="Render SVG, PDF, and 600-dpi PNG summaries in addition to tables.")
    return parser.parse_args()


def load_gate_frame(predictions: Path, sensitivity_variables: Path) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Load gate predictions and join the required record-level metadata."""
    with np.load(predictions, allow_pickle=True) as artifact:
        frame = pd.DataFrame({
            "record_key": record_keys(artifact["city_token"], artifact["tree_id"], artifact["crop_index"]),
            "city_token": np.asarray(artifact["city_token"], dtype=str),
            "tree_id": np.asarray(artifact["tree_id"], dtype=np.int64),
            "crop_index": np.asarray(artifact["crop_index"], dtype=np.int64),
        })
        gates = np.asarray(artifact["gate_weights"], dtype=np.float64)
        modalities = np.asarray(artifact["modality_names"], dtype=str)
    if gates.shape != (len(frame), len(modalities)):
        raise ValueError("Gate-weight dimensions do not match records and modality names")
    sensitivity = pd.read_csv(
        sensitivity_variables,
        usecols=["record_key", "qa_flag_low_vegetated_height_with_lidar", "frequency_category"],
    )
    frame = merge_one_to_one(frame, sensitivity, "sensitivity variables")
    required = ["qa_flag_low_vegetated_height_with_lidar", "frequency_category"]
    if frame[required].isna().any().any():
        raise ValueError("Missing sensitivity metadata after record-key join")
    return frame, gates, modalities


def extract_height(frame: pd.DataFrame, shard_root: Path, cache_path: Path) -> np.ndarray:
    """Extract CHM height by record key and save a validated external cache."""
    positions: dict[str, dict[tuple[int, int], int]] = {}
    for index, row in frame[["city_token", "tree_id", "crop_index"]].iterrows():
        city_positions = positions.setdefault(str(row.city_token), {})
        city_positions[(int(row.tree_id), int(row.crop_index))] = int(index)
    heights = np.full(len(frame), np.nan, dtype=np.float32)
    matched = 0
    for city, wanted in positions.items():
        for path in sorted((shard_root / city).glob("*_tree_centered_complete_inputs.npz")):
            with np.load(path, allow_pickle=True) as artifact:
                columns = np.asarray(artifact["tree_centered_naip_chm_structure_columns"], dtype=str)
                hits = np.flatnonzero(columns == "structure_tree_chm_p95_m")
                if len(hits) != 1:
                    raise ValueError(f"Could not uniquely locate vegetation height in {path}")
                values = np.asarray(
                    artifact["tree_centered_naip_chm_structure"][:, int(hits[0])],
                    dtype=np.float32,
                )
                for local, (tree, crop) in enumerate(zip(artifact["tree_id"], artifact["crop_index"])):
                    destination = wanted.get((int(tree), int(crop)))
                    if destination is not None:
                        heights[destination] = values[local]
                        matched += 1
        print(f"Height extraction: {city}; matched={matched:,}/{len(frame):,}", flush=True)
    if matched != len(frame) or np.isnan(heights).any():
        missing = int(np.isnan(heights).sum())
        raise ValueError(f"Height extraction incomplete: matched={matched:,}; missing={missing:,}")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, record_key=frame.record_key.to_numpy(str), height_m=heights)
    return heights


def load_or_extract_height(frame: pd.DataFrame, shard_root: Path, cache_path: Path) -> np.ndarray:
    """Reuse an exact keyed height cache or extract heights from complete shards."""
    if not cache_path.exists():
        return extract_height(frame, shard_root, cache_path)
    with np.load(cache_path, allow_pickle=False) as cached:
        cached_keys = np.asarray(cached["record_key"], dtype=str)
        heights = np.asarray(cached["height_m"], dtype=np.float32)
    if not np.array_equal(cached_keys, frame.record_key.to_numpy(str)):
        raise ValueError("Cached height record keys do not match prediction order")
    if len(heights) != len(frame) or np.isnan(heights).any():
        raise ValueError("Cached height values are incomplete")
    return heights


def prepare_long_table(
    frame: pd.DataFrame,
    gates: np.ndarray,
    modalities: np.ndarray,
    configuration: dict,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Assign configured strata and reshape gate weights into tidy long form."""
    gate_config = configuration["gate_analysis"]
    boundaries = [float(value) for value in gate_config["height_class_boundaries_m"]]
    height_labels = [str(value) for value in gate_config["height_class_labels"]]
    if len(height_labels) != len(boundaries) + 1:
        raise ValueError("Height labels must contain one more value than boundaries")
    frame["height_class"] = pd.cut(
        frame.vegetation_height_m,
        bins=[-np.inf, *boundaries, np.inf],
        labels=height_labels,
        right=False,
        ordered=True,
    )
    provenance = configuration["taxon_provenance"]
    frame["taxon_provenance"] = np.where(
        frame.frequency_category.eq(provenance["core_value"]),
        provenance["core_label"],
        provenance["rare_label"],
    )
    qa_column = str(configuration["qa_warning_excluded"])
    frame["qa_screened"] = ~boolean_series(frame[qa_column])
    for index, modality in enumerate(modalities):
        frame[f"gate_{modality}"] = gates[:, index]
    gate_columns = [f"gate_{name}" for name in modalities]
    screened = frame.loc[frame.qa_screened].copy()
    long = screened.melt(
        id_vars=["record_key", "city_token", "height_class", "taxon_provenance"],
        value_vars=gate_columns,
        var_name="modality",
        value_name="gate_weight",
    )
    long["modality"] = long.modality.str.removeprefix("gate_")
    long["modality_label"] = long.modality.map(MODALITY_LABELS).fillna(long.modality)
    return screened, long, height_labels


def summarize(long: pd.DataFrame, strata: list[str]) -> pd.DataFrame:
    """Calculate support, center, and spread for every modality-stratum cell."""
    keys = strata + ["modality", "modality_label"]
    grouped = long.groupby(keys, observed=True, sort=False)["gate_weight"]
    result = grouped.agg(support="size", mean="mean", standard_deviation="std", median="median").reset_index()
    quantiles = grouped.quantile([0.25, 0.75]).unstack(-1).reset_index().rename(columns={0.25: "q25", 0.75: "q75"})
    result = result.merge(quantiles, on=keys, validate="one_to_one")
    result["interquartile_range"] = result.q75 - result.q25
    return result


def omnibus(long: pd.DataFrame, stratum: str) -> pd.DataFrame:
    """Run descriptive Kruskal-Wallis tests across one stratification variable."""
    rows: list[dict[str, object]] = []
    for (modality, label), subset in long.groupby(["modality", "modality_label"], sort=False):
        arrays = [group.gate_weight.to_numpy() for _, group in subset.groupby(stratum, observed=True) if len(group)]
        statistic, p_value = kruskal(*arrays)
        records = sum(map(len, arrays))
        groups = len(arrays)
        denominator = max(records - groups, 1)
        means = subset.groupby(stratum, observed=True).gate_weight.mean()
        rows.append({
            "stratum": stratum,
            "modality": modality,
            "modality_label": label,
            "groups": groups,
            "records": records,
            "kruskal_h": statistic,
            "p_value": p_value,
            "epsilon_squared": max(0.0, (float(statistic) - groups + 1) / denominator),
            "maximum_minus_minimum_mean": float(means.max() - means.min()),
        })
    return pd.DataFrame(rows)


def write_tables(output_dir: Path, long: pd.DataFrame, minimum_support: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Write all aggregate gate tables and return figure-facing summaries."""
    overall = summarize(long.assign(stratum="All QA-screened records"), ["stratum"])
    by_height = summarize(long, ["height_class"])
    by_city = summarize(long, ["city_token"])
    by_city_height_all = summarize(long, ["city_token", "height_class"])
    by_city_height = by_city_height_all.loc[by_city_height_all.support.ge(minimum_support)].copy()
    by_provenance = summarize(long, ["taxon_provenance"])
    overall.to_csv(output_dir / "gate_weights_overall.csv", index=False)
    by_height.to_csv(output_dir / "gate_weights_by_height_class.csv", index=False)
    by_city.to_csv(output_dir / "gate_weights_by_city.csv", index=False)
    by_city_height_all.to_csv(output_dir / "gate_weights_by_city_and_height_all.csv", index=False)
    by_city_height.to_csv(output_dir / f"gate_weights_by_city_and_height_support_ge{minimum_support}.csv", index=False)
    by_provenance.to_csv(output_dir / "gate_weights_by_taxon_provenance.csv", index=False)
    tests = pd.concat(
        [omnibus(long, "height_class"), omnibus(long, "city_token"), omnibus(long, "taxon_provenance")],
        ignore_index=True,
    )
    tests.to_csv(output_dir / "gate_weight_heterogeneity_tests.csv", index=False)
    return by_height, by_provenance


def plot_summary(
    height: pd.DataFrame,
    provenance: pd.DataFrame,
    height_order: list[str],
    output: Path,
    png_dpi: int,
) -> None:
    """Render the optional two-panel summary in standard publication formats."""
    import matplotlib.pyplot as plt

    modalities = height.modality.drop_duplicates().tolist()
    figure, axes = plt.subplots(1, 2, figsize=(12.0, 4.8), constrained_layout=True)
    markers = ["o", "s", "^", "D", "v", "P"]
    for modality, marker in zip(modalities, markers):
        subset = height.loc[height.modality.eq(modality)].set_index("height_class").reindex(height_order)
        axes[0].errorbar(
            np.arange(len(height_order)), subset["median"],
            yerr=np.vstack([subset["median"] - subset["q25"], subset["q75"] - subset["median"]]),
            color="0.15", marker=marker, linestyle="-", linewidth=0.9,
            markersize=5, capsize=2, alpha=0.82,
            label=subset.modality_label.dropna().iloc[0],
        )
        provenance_order = ["Core taxa", "Rare taxa"]
        provenance_subset = provenance.loc[provenance.modality.eq(modality)].set_index("taxon_provenance").reindex(provenance_order)
        offset = (modalities.index(modality) - (len(modalities) - 1) / 2) * 0.035
        axes[1].errorbar(
            np.arange(2) + offset, provenance_subset["median"],
            yerr=np.vstack([provenance_subset["median"] - provenance_subset["q25"], provenance_subset["q75"] - provenance_subset["median"]]),
            color="0.15", marker=marker, linestyle="none", markersize=5,
            capsize=2, alpha=0.82,
        )
    axes[0].set_xticks(range(len(height_order)), height_order)
    axes[1].set_xticks(range(2), ["Core taxa", "Rare taxa"])
    axes[0].set_title("A  Vegetation-height class", loc="left", fontweight="bold")
    axes[1].set_title("B  Taxon provenance", loc="left", fontweight="bold")
    for axis in axes:
        axis.set_ylabel("Gate weight (median and IQR)")
        axis.set_ylim(0, 0.75)
        axis.grid(axis="y", color="0.88", linewidth=0.7)
    axes[0].legend(frameon=False, fontsize=8, ncol=2)
    figure.savefig(output.with_suffix(".png"), dpi=png_dpi, bbox_inches="tight")
    figure.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(output.with_suffix(".svg"), bbox_inches="tight")
    plt.close(figure)


def main() -> int:
    """Run the complete gate-weight stratification workflow."""
    args = parse_args()
    configuration = load_configuration(args.sensitivity_config)
    configured_support = int(configuration["gate_analysis"]["minimum_city_height_support"])
    minimum_support = configured_support if args.minimum_city_height_support is None else args.minimum_city_height_support
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame, gates, modalities = load_gate_frame(args.predictions, args.sensitivity_variables)
    height_cache = args.output_dir / "tree_chm_p95_height_by_record.npz"
    frame["vegetation_height_m"] = load_or_extract_height(frame, args.shard_root, height_cache)
    screened, long, height_order = prepare_long_table(frame, gates, modalities, configuration)
    by_height, by_provenance = write_tables(args.output_dir, long, minimum_support)
    gate_columns = [f"gate_{name}" for name in modalities]
    record_columns = [
        "record_key", "city_token", "vegetation_height_m", "height_class",
        "taxon_provenance", "qa_screened", *gate_columns,
    ]
    frame[record_columns].to_csv(
        args.output_dir / "gate_weight_stratification_records.csv.gz",
        index=False,
        compression="gzip",
    )
    if args.plot:
        plot_summary(
            by_height, by_provenance, height_order,
            args.output_dir / "gate_weight_height_and_provenance",
            int(configuration["figures"]["png_dpi"]),
        )
    metadata = {
        "test_records_total": int(len(frame)),
        "qa_screened_records": int(len(screened)),
        "qa_warning_records_excluded": int((~frame.qa_screened).sum()),
        "height_variable": "structure_tree_chm_p95_m",
        "height_classes": height_order,
        "minimum_city_height_support": minimum_support,
        "modalities": modalities.tolist(),
        "inputs": {
            "predictions": {"file_name": args.predictions.name, "sha256": file_sha256(args.predictions)},
            "sensitivity_variables": {"file_name": args.sensitivity_variables.name, "sha256": file_sha256(args.sensitivity_variables)},
            "sensitivity_config": {"file_name": args.sensitivity_config.name, "sha256": file_sha256(args.sensitivity_config)},
            "complete_shards": {"artifact_role": "complete city shards"},
        },
        "interpretation_warning": "Gate weights describe learned sample-dependent allocation and are not causal feature importance.",
    }
    (args.output_dir / "analysis_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))
    print("\nHeight-class summaries:\n", by_height.to_string(index=False))
    print("\nTaxon-provenance summaries:\n", by_provenance.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
