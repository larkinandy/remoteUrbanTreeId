#!/usr/bin/env python
"""Evaluate final-model performance by CHM-supported Sentinel-cell vegetation.

The proxy is the fraction of valid CHM pixels in the aligned Sentinel-2 cell
with height >= 2 m. It is not a focal-crown coverage estimate.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--predictions",
        type=Path,
        default=Path("outputs/reviewer_revision/final_pruned_k11_test_top5_ensemble/ensemble_predictions.npz"),
    )
    p.add_argument(
        "--sensitivity-variables",
        type=Path,
        default=Path("outputs/reviewer_revision/final_pruned_sensitivity_table_with_taxon_provenance/final_test_sensitivity_variables.csv.gz"),
    )
    p.add_argument(
        "--mixed-pixel-proxies",
        type=Path,
        default=Path("outputs/reviewer_revision/sentinel_mixed_pixel_proxies_reportable_v1/record_level_reportable_mixed_pixel_proxies.csv.gz"),
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/reviewer_revision/chm_mixed_pixel_stratified_sensitivity"),
    )
    p.add_argument("--threshold", type=float, default=0.25)
    p.add_argument("--minimum-chm-coverage", type=float, default=0.50)
    return p.parse_args()


def metrics(y: np.ndarray, pred: np.ndarray) -> dict[str, float | int]:
    return {
        "records": int(len(y)),
        "accuracy": float(accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y, pred, average="weighted", zero_division=0)),
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    z = np.load(args.predictions, allow_pickle=True)
    pred = pd.DataFrame(
        {
            "record_key": [
                f"{c}|{int(t)}|{int(i)}"
                for c, t, i in zip(z["city_token"], z["tree_id"], z["crop_index"])
            ],
            "label_k11": z["labels"].astype(int),
            "prediction_k11": z["predictions"].astype(int),
            "label_k6": z["collapsed_labels"].astype(int),
            "prediction_k6": z["collapsed_predictions"].astype(int),
        }
    )
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
    frame = pred.merge(qa, on="record_key", how="left", validate="one_to_one")
    frame = frame.merge(proxy, on="record_key", how="left", validate="one_to_one")
    if frame.qa_flag_low_vegetated_height_with_lidar.isna().any():
        raise ValueError("Some prediction records lack QA flags")

    frame["qa_screened"] = ~frame.qa_flag_low_vegetated_height_with_lidar.astype(bool)
    frame["chm_proxy_eligible"] = (
        frame.qa_screened
        & frame.chm_tall_vegetation_fraction_of_valid_area.notna()
        & frame.chm_valid_fraction_in_cell.ge(args.minimum_chm_coverage)
    )
    frame["chm_proxy_stratum"] = pd.NA
    eligible = frame.chm_proxy_eligible
    frame.loc[eligible & frame.chm_tall_vegetation_fraction_of_valid_area.lt(args.threshold), "chm_proxy_stratum"] = (
        f"<{args.threshold:.0%}"
    )
    frame.loc[eligible & frame.chm_tall_vegetation_fraction_of_valid_area.ge(args.threshold), "chm_proxy_stratum"] = (
        f">={args.threshold:.0%}"
    )

    rows = []
    for label_space in ("k6", "k11"):
        for stratum in (f"<{args.threshold:.0%}", f">={args.threshold:.0%}"):
            part = frame[frame.chm_proxy_stratum.eq(stratum)]
            row = {
                "label_space": label_space,
                "criterion": stratum,
                "minimum_chm_valid_coverage": args.minimum_chm_coverage,
                "tall_vegetation_fraction_threshold": args.threshold,
            }
            row.update(metrics(part[f"label_{label_space}"].to_numpy(), part[f"prediction_{label_space}"].to_numpy()))
            rows.append(row)
    results = pd.DataFrame(rows)
    results.to_csv(args.output_dir / "chm_mixed_pixel_stratified_performance.csv", index=False)

    deltas = []
    for label_space in ("k6", "k11"):
        part = results.set_index(["label_space", "criterion"]).loc[label_space]
        high = part.loc[f">={args.threshold:.0%}"]
        low = part.loc[f"<{args.threshold:.0%}"]
        deltas.append(
            {
                "label_space": label_space,
                "contrast": f">={args.threshold:.0%} minus <{args.threshold:.0%}",
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
        "minimum_chm_valid_coverage": args.minimum_chm_coverage,
        "tall_vegetation_fraction_threshold": args.threshold,
        "proxy_definition": "Fraction of valid CHM pixels in the aligned 10 x 10 m Sentinel-2 cell with height >= 2 m.",
        "interpretation_warning": "This is neighborhood-scale tall-vegetation occupancy, not focal-tree crown coverage.",
    }
    (args.output_dir / "analysis_metadata.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    frame[[
        "record_key", "qa_screened", "chm_valid_fraction_in_cell",
        "chm_tall_vegetation_fraction_of_valid_area", "chm_proxy_eligible",
        "chm_proxy_stratum", "label_k11", "prediction_k11", "label_k6", "prediction_k6"
    ]].to_csv(args.output_dir / "record_level_chm_strata.csv.gz", index=False, compression="gzip")

    print(results.to_string(index=False))
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
