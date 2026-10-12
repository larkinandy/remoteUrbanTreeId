#!/usr/bin/env python3
"""Reproduce the final QA-screened independent-test sensitivity table."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from sensitivity_utils import (
    boolean_series,
    file_sha256,
    load_configuration,
    load_predictions,
    merge_one_to_one,
    metric_set,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "analysis" / "sensitivity" / "sensitivity_config.json"
DEFAULT_GROUPING = ROOT / "config" / "final_grouping.json"


def parse_args() -> argparse.Namespace:
    """Parse explicit source artifacts used to construct every final stratum."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--sensitivity-variables", type=Path, required=True)
    parser.add_argument("--mixed-pixel-proxies", type=Path, required=True)
    parser.add_argument("--filter-variables", type=Path, required=True)
    parser.add_argument("--naip-year-variables", type=Path, required=True)
    parser.add_argument("--complete-shard-root", type=Path, required=True)
    parser.add_argument("--grouping-config", type=Path, default=DEFAULT_GROUPING)
    parser.add_argument("--sensitivity-config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def load_interpolation(shard_root: Path, wanted_keys: set[str]) -> pd.DataFrame:
    """Read record-keyed Sentinel-2 interpolation fractions from complete shards."""
    rows: list[tuple[str, float]] = []
    for shard in sorted(shard_root.glob("*/*_tree_centered_complete_inputs.npz")):
        with np.load(shard, allow_pickle=True) as artifact:
            columns = np.asarray(artifact["sentinel_quality_columns"], dtype=str)
            hits = np.flatnonzero(columns == "sentinel_interpolated_fraction")
            if len(hits) != 1:
                raise ValueError(f"Cannot locate Sentinel interpolation fraction in {shard}")
            values = np.asarray(artifact["sentinel_quality"][:, int(hits[0])], dtype=float)
            for tree, crop, value in zip(artifact["tree_id"], artifact["crop_index"], values):
                key = f"{shard.parent.name}|{int(tree)}|{int(crop)}"
                if key in wanted_keys:
                    rows.append((key, float(value)))
    if not rows:
        raise FileNotFoundError(f"No complete shards found below {shard_root}")
    result = pd.DataFrame(rows, columns=["record_key", "sentinel_interpolation_fraction"])
    if result.record_key.duplicated().any():
        raise ValueError("Complete-shard record keys are not unique")
    if set(result.record_key) != wanted_keys:
        raise ValueError(f"Missing interpolation values for {len(wanted_keys - set(result.record_key))} records")
    return result


def prepare_analysis_frame(arguments: argparse.Namespace, configuration: dict) -> pd.DataFrame:
    """Join all finalized record-level variables to canonical test predictions."""
    frame = load_predictions(arguments.predictions, arguments.grouping_config)
    qa_column = str(configuration["qa_warning_excluded"])
    sensitivity = pd.read_csv(
        arguments.sensitivity_variables,
        usecols=["record_key", qa_column, "match_distance_m",
                 "duplicate_coordinate_percent", "frequency_category"],
    )
    frame = merge_one_to_one(frame, sensitivity, "sensitivity variables")
    mixed = pd.read_csv(
        arguments.mixed_pixel_proxies,
        usecols=["record_key", "naip_vegetation_fraction"],
    )
    frame = merge_one_to_one(frame, mixed, "mixed-pixel proxies")
    filters = pd.read_csv(
        arguments.filter_variables,
        usecols=["record_key", "inventory_minus_lidar_years"],
    )
    frame = merge_one_to_one(frame, filters, "filter variables")
    naip = pd.read_csv(
        arguments.naip_year_variables,
        usecols=["record_key", "naip_minus_inventory_years"],
    )
    frame = merge_one_to_one(frame, naip, "NAIP year variables")
    interpolation = load_interpolation(arguments.complete_shard_root, set(frame.record_key))
    frame = merge_one_to_one(frame, interpolation, "Sentinel-2 interpolation variables")
    required = [
        qa_column, "match_distance_m", "duplicate_coordinate_percent", "frequency_category",
        "naip_vegetation_fraction",
        "naip_minus_inventory_years",
        "sentinel_interpolation_fraction",
    ]
    missing = frame[required].isna().sum()
    if missing.any():
        raise ValueError(f"Missing finalized sensitivity values:\n{missing[missing.gt(0)]}")
    frame["qa_warning"] = boolean_series(frame[qa_column])
    return frame


def define_strata(frame: pd.DataFrame, configuration: dict) -> list[tuple[str, str, pd.Series]]:
    """Create the reference population and both levels of each frozen contrast."""
    thresholds = configuration["thresholds"]
    provenance = configuration["taxon_provenance"]
    distance = float(thresholds["crown_match_distance_m"])
    duplicate = float(thresholds["duplicate_coordinate_percent"])
    vegetation = float(thresholds["naip_vegetation_fraction"])
    interpolation = float(thresholds["sentinel_interpolation_fraction"])
    lidar = float(thresholds["lidar_inventory_absolute_year_difference"])
    naip = float(thresholds["naip_inventory_absolute_year_difference"])
    all_records = pd.Series(True, index=frame.index)
    return [
        ("Reference model", "All independent test records", all_records),
        ("Vegetation height", "Passed center vegetation-height QA (2 m threshold)", ~frame.qa_warning),
        ("Vegetation height", "Low center vegetation-height warning (2 m threshold)", frame.qa_warning),
        ("Crown-match distance", f"Distance >= {distance:g} m", frame.match_distance_m.ge(distance)),
        ("Crown-match distance", f"Distance < {distance:g} m", frame.match_distance_m.lt(distance)),
        ("Duplicate coordinates", f"Duplicates >= {duplicate:g}%", frame.duplicate_coordinate_percent.ge(duplicate)),
        ("Duplicate coordinates", f"Duplicates < {duplicate:g}%", frame.duplicate_coordinate_percent.lt(duplicate)),
        ("Taxon provenance", provenance["core_label"], frame.frequency_category.eq(provenance["core_value"])),
        ("Taxon provenance", provenance["rare_label"], frame.frequency_category.ne(provenance["core_value"])),
        ("Sentinel-2 vegetation", f">= {vegetation:.0%} within cell", frame.naip_vegetation_fraction.ge(vegetation)),
        ("Sentinel-2 vegetation", f"< {vegetation:.0%} within cell", frame.naip_vegetation_fraction.lt(vegetation)),
        ("Sentinel-2 interpolation", f">= {interpolation:.0%}", frame.sentinel_interpolation_fraction.ge(interpolation)),
        ("Sentinel-2 interpolation", f"< {interpolation:.0%}", frame.sentinel_interpolation_fraction.lt(interpolation)),
        ("LiDAR year mismatch", f">= {lidar:g} years", frame.inventory_minus_lidar_years.notna() & frame.inventory_minus_lidar_years.abs().ge(lidar)),
        ("LiDAR year mismatch", f"< {lidar:g} years", frame.inventory_minus_lidar_years.notna() & frame.inventory_minus_lidar_years.abs().lt(lidar)),
        ("NAIP year mismatch", f">= {naip:g} years", frame.naip_minus_inventory_years.abs().ge(naip)),
        ("NAIP year mismatch", f"< {naip:g} years", frame.naip_minus_inventory_years.abs().lt(naip)),
    ]


def calculate_table(frame: pd.DataFrame, strata: list[tuple[str, str, pd.Series]]) -> pd.DataFrame:
    """Calculate both model label spaces for every frozen binary stratum."""
    rows: list[dict[str, object]] = []
    for analysis, criterion, selector in strata:
        subset = frame.loc[selector]
        for label_space, class_count in (("k11", 11), ("collapsed_k6", 6)):
            metrics = metric_set(
                subset[f"label_{label_space}"].to_numpy(),
                subset[f"prediction_{label_space}"].to_numpy(),
                class_count,
            )
            rows.append({
                "analysis": analysis,
                "criterion": criterion,
                "label_space": label_space,
                **metrics,
            })
    return pd.DataFrame(rows)


def main() -> int:
    """Generate the complete table and a machine-readable audit."""
    arguments = parse_args()
    for path in (
        arguments.predictions, arguments.sensitivity_variables, arguments.mixed_pixel_proxies,
        arguments.filter_variables, arguments.naip_year_variables,
        arguments.complete_shard_root,
        arguments.grouping_config, arguments.sensitivity_config,
    ):
        if not path.exists():
            raise FileNotFoundError(path)
    configuration = load_configuration(arguments.sensitivity_config)
    frame = prepare_analysis_frame(arguments, configuration)
    table = calculate_table(frame, define_strata(frame, configuration))
    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(arguments.output_dir / "final_sensitivity_table_all_label_spaces.csv", index=False)
    table.loc[table.label_space.eq(configuration["reporting"]["primary_label_space"])].to_csv(
        arguments.output_dir / "final_sensitivity_table_collapsed_k6.csv", index=False
    )
    inputs = {
        name: {"path": str(path), "sha256": file_sha256(path)}
        for name, path in {
            "predictions": arguments.predictions,
            "sensitivity_variables": arguments.sensitivity_variables,
            "mixed_pixel_proxies": arguments.mixed_pixel_proxies,
            "filter_variables": arguments.filter_variables,
            "naip_year_variables": arguments.naip_year_variables,
            "grouping_config": arguments.grouping_config,
            "sensitivity_config": arguments.sensitivity_config,
        }.items()
    }
    inputs["complete_shard_root"] = {
        "path": str(arguments.complete_shard_root),
        "shard_count": len(list(arguments.complete_shard_root.glob(
            "*/*_tree_centered_complete_inputs.npz"
        ))),
    }
    audit = {
        "independent_test_records": int(len(frame)),
        "qa_screened_test_records": int((~frame.qa_warning).sum()),
        "qa_warning_test_records": int(frame.qa_warning.sum()),
        "record_keys_unique": bool(frame.record_key.is_unique),
        "canonical_grouping_validation_passed": True,
        "strata_rows_per_label_space": int(len(table) / 2),
        "missing_lidar_year_records": int(frame.inventory_minus_lidar_years.isna().sum()),
        "inputs": inputs,
    }
    (arguments.output_dir / "analysis_metadata.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8"
    )
    print(table.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
