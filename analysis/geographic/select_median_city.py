#!/usr/bin/env python3
"""Select an average city using only final primary-model test performance."""

from __future__ import annotations

import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from geographic_utils import load_predictions, load_qa_flags, metric_set, screen_qa


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "outputs" / "reviewer_revision" / "primary_model_median_city_selection"


def parse_args() -> argparse.Namespace:
    """Parse primary predictions and finalized QA metadata."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--qa-metadata", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--minimum-test-records", type=int, default=1000)
    parser.add_argument("--minimum-group-records", type=int, default=10)
    return parser.parse_args()


def main() -> int:
    """Rank eligible cities using primary-model collapsed_k6 performance only."""
    arguments = parse_args()
    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    frame = load_predictions(arguments.predictions, "primary").rename(columns={
        "collapsed_k6_label_primary": "label",
        "collapsed_k6_prediction_primary": "prediction",
    })
    frame = screen_qa(
        frame.merge(load_qa_flags(arguments.qa_metadata), on="record_key", validate="one_to_one")
    )
    rows = []
    for city, group in frame.groupby("city", sort=True):
        metrics, _, _ = metric_set(
            group.label.to_numpy(), group.prediction.to_numpy(), 6,
            arguments.minimum_group_records,
        )
        rows.append({
            "city": city,
            "qa_screened_test_records": len(group),
            "groups_ge_minimum": metrics["groups_ge_minimum"],
            "macro_f1_groups_ge_minimum": metrics["macro_f1_groups_ge_minimum"],
            "accuracy": metrics["accuracy"],
        })
    ranking = pd.DataFrame(rows)
    ranking = ranking.loc[
        ranking.qa_screened_test_records.ge(arguments.minimum_test_records)
    ].sort_values(
        ["macro_f1_groups_ge_minimum", "city"]
    ).reset_index(drop=True)
    ranking["rank"] = np.arange(1, len(ranking) + 1)
    median = float(ranking.macro_f1_groups_ge_minimum.median())
    ranking["distance_from_median"] = (ranking.macro_f1_groups_ge_minimum - median).abs()
    selection = ranking.sort_values(
        ["distance_from_median", "qa_screened_test_records", "city"],
        ascending=[True, False, True],
    ).iloc[0]
    ranking.to_csv(arguments.output_dir / "eligible_city_primary_model_ranking.csv", index=False)
    result = {
        "selection_source": "final primary national model; leave-one-city-out and one-city-trained results not used",
        "minimum_qa_screened_test_records": arguments.minimum_test_records,
        "macro_f1_rule": (
            "unweighted mean of collapsed_k6 group F1 values for groups with at least "
            f"{arguments.minimum_group_records} city test records"
        ),
        "eligible_cities": int(len(ranking)),
        "median_city_macro_f1": median,
        "tie_breaker": "larger QA-screened test support, then alphabetical city token",
        "selected_city": str(selection.city),
        "selected_city_rank": int(selection["rank"]),
        "selected_city_macro_f1": float(selection.macro_f1_groups_ge_minimum),
        "selected_city_accuracy": float(selection.accuracy),
        "selected_city_qa_screened_test_records": int(selection.qa_screened_test_records),
        "selected_city_groups_ge_minimum": int(selection.groups_ge_minimum),
    }
    (arguments.output_dir / "selection.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(ranking.to_string(index=False))
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
