#!/usr/bin/env python3
"""Evaluate final-model performance by CHM-supported Sentinel-cell vegetation.

The proxy is the fraction of valid CHM pixels in the aligned Sentinel-2 cell
with height at or above the configured vegetation-height threshold. It is a
neighborhood occupancy measure, not a focal-crown coverage estimate. Inputs
are explicit so the script can be reproduced outside the original workstation.
"""

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
    """Parse explicit data sources and optional configuration overrides."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True, help="Final ensemble prediction NPZ.")
    parser.add_argument("--sensitivity-variables", type=Path, required=True, help="Record-keyed QA-variable CSV.")
    parser.add_argument("--mixed-pixel-proxies", type=Path, required=True, help="Record-keyed CHM proxy CSV.")
    parser.add_argument("--output-dir", type=Path, required=True, help="External directory for generated tables and records.")
    parser.add_argument("--grouping-config", type=Path, default=DEFAULT_GROUPING, help="Canonical k11-to-collapsed_k6 mapping JSON.")
    parser.add_argument("--sensitivity-config", type=Path, default=DEFAULT_CONFIG, help="Frozen sensitivity thresholds JSON.")
    parser.add_argument("--threshold", type=float, help="Override the configured tall-vegetation fraction threshold.")
    parser.add_argument("--minimum-chm-coverage", type=float, help="Override the configured minimum valid CHM fraction.")
    return parser.parse_args()


def calculate_stratified_metrics(
    frame: pd.DataFrame,
    threshold: float,
    minimum_coverage: float,
) -> pd.DataFrame:
    """Calculate performance in low and high CHM-occupancy strata."""
    rows: list[dict[str, object]] = []
    criteria = (f"<{threshold:.0%}", f">={threshold:.0%}")
    for label_space, class_count in (("collapsed_k6", 6), ("k11", 11)):
        for criterion in criteria:
            subset = frame.loc[frame.chm_proxy_stratum.eq(criterion)]
            row = {
                "label_space": label_space,
                "criterion": criterion,
                "minimum_chm_valid_coverage": minimum_coverage,
                "tall_vegetation_fraction_threshold": threshold,
            }
            row.update(
                metric_set(
                    subset[f"label_{label_space}"].to_numpy(),
                    subset[f"prediction_{label_space}"].to_numpy(),
                    class_count,
                )
            )
            rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    """Calculate CHM-supported neighborhood-vegetation metrics and audits."""
    args = parse_args()
    configuration = load_configuration(args.sensitivity_config)
    thresholds = configuration["thresholds"]
    threshold = float(
        thresholds["chm_tall_vegetation_fraction"] if args.threshold is None else args.threshold
    )
    minimum_coverage = float(
        thresholds["chm_valid_fraction"]
        if args.minimum_chm_coverage is None
        else args.minimum_chm_coverage
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    pred = load_predictions(args.predictions, args.grouping_config)
    if not pred.record_key.is_unique:
        raise ValueError("Prediction record keys are not unique")

    qa = pd.read_csv(
        args.sensitivity_variables,
        usecols=["record_key", "qa_flag_low_vegetated_height_with_lidar"],
    )
    proxy = pd.read_csv(
        args.mixed_pixel_proxies,
        usecols=[
            "record_key",
            "chm_valid_fraction_in_cell",
            "chm_tall_vegetation_fraction_of_valid_area",
        ],
    )
    frame = merge_one_to_one(pred, qa, "sensitivity variables")
    frame = merge_one_to_one(frame, proxy, "mixed-pixel proxies")
    if frame.qa_flag_low_vegetated_height_with_lidar.isna().any():
        raise ValueError("Some prediction records lack QA flags")

    frame["qa_screened"] = ~boolean_series(frame.qa_flag_low_vegetated_height_with_lidar)
    frame["chm_proxy_eligible"] = (
        frame.qa_screened
        & frame.chm_tall_vegetation_fraction_of_valid_area.notna()
        & frame.chm_valid_fraction_in_cell.ge(minimum_coverage)
    )
    frame["chm_proxy_stratum"] = pd.NA
    eligible = frame.chm_proxy_eligible
    frame.loc[eligible & frame.chm_tall_vegetation_fraction_of_valid_area.lt(threshold), "chm_proxy_stratum"] = (
        f"<{threshold:.0%}"
    )
    frame.loc[eligible & frame.chm_tall_vegetation_fraction_of_valid_area.ge(threshold), "chm_proxy_stratum"] = (
        f">={threshold:.0%}"
    )

    results = calculate_stratified_metrics(frame, threshold, minimum_coverage)
    results.to_csv(args.output_dir / "chm_mixed_pixel_stratified_performance.csv", index=False)

    deltas = []
    for label_space in ("collapsed_k6", "k11"):
        part = results.set_index(["label_space", "criterion"]).loc[label_space]
        high = part.loc[f">={threshold:.0%}"]
        low = part.loc[f"<{threshold:.0%}"]
        deltas.append(
            {
                "label_space": label_space,
                "contrast": f">={threshold:.0%} minus <{threshold:.0%}",
                "accuracy_difference": high.accuracy - low.accuracy,
                "macro_f1_difference": high.macro_f1 - low.macro_f1,
                "weighted_f1_difference": high.weighted_f1 - low.weighted_f1,
            }
        )
    pd.DataFrame(deltas).to_csv(args.output_dir / "chm_mixed_pixel_performance_differences.csv", index=False)

    distribution = frame.loc[frame.qa_screened, [
        "chm_valid_fraction_in_cell", "chm_tall_vegetation_fraction_of_valid_area"
    ]].describe(percentiles=[0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]).T
    distribution.to_csv(args.output_dir / "chm_proxy_distribution_qa_screened.csv")

    audit = {
        "prediction_records": int(len(frame)),
        "qa_screened_records": int(frame.qa_screened.sum()),
        "qa_excluded_records": int((~frame.qa_screened).sum()),
        "eligible_records": int(frame.chm_proxy_eligible.sum()),
        "excluded_for_missing_or_low_chm_coverage": int(
            frame.qa_screened.sum() - frame.chm_proxy_eligible.sum()
        ),
        "minimum_chm_valid_coverage": minimum_coverage,
        "tall_vegetation_fraction_threshold": threshold,
        "proxy_definition": "Fraction of valid CHM pixels in the aligned 10 x 10 m Sentinel-2 cell with height >= 2 m.",
        "interpretation_warning": "This is neighborhood-scale tall-vegetation occupancy, not focal-tree crown coverage.",
        "inputs": {
            "predictions": {"file_name": args.predictions.name, "sha256": file_sha256(args.predictions)},
            "sensitivity_variables": {"file_name": args.sensitivity_variables.name, "sha256": file_sha256(args.sensitivity_variables)},
            "mixed_pixel_proxies": {"file_name": args.mixed_pixel_proxies.name, "sha256": file_sha256(args.mixed_pixel_proxies)},
            "grouping_config": {"file_name": args.grouping_config.name, "sha256": file_sha256(args.grouping_config)},
            "sensitivity_config": {"file_name": args.sensitivity_config.name, "sha256": file_sha256(args.sensitivity_config)},
        },
    }
    (args.output_dir / "analysis_metadata.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    frame[[
        "record_key", "qa_screened", "chm_valid_fraction_in_cell",
        "chm_tall_vegetation_fraction_of_valid_area", "chm_proxy_eligible",
        "chm_proxy_stratum", "label_k11", "prediction_k11",
        "label_collapsed_k6", "prediction_collapsed_k6"
    ]].to_csv(args.output_dir / "record_level_chm_strata.csv.gz", index=False, compression="gzip")

    print(results.to_string(index=False))
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
