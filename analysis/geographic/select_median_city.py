#!/usr/bin/env python3
"""Select an average city using only final primary-model test performance."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score


PREDICTIONS = Path("outputs/reviewer_revision/final_pruned_k11_test_top5_ensemble/ensemble_predictions.npz")
METADATA = Path("outputs/reviewer_revision/final_pruned_sensitivity_table_with_taxon_provenance/final_test_sensitivity_variables.csv.gz")
OUTPUT = Path("outputs/reviewer_revision/primary_model_median_city_selection")


def main() -> int:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with np.load(PREDICTIONS, allow_pickle=True) as z:
        record_key = np.asarray([
            f"{city}|{int(tree)}|{int(crop)}"
            for city, tree, crop in zip(z["city_token"], z["tree_id"], z["crop_index"])
        ])
        frame = pd.DataFrame({
            "record_key": record_key,
            "city": np.asarray(z["city_token"], dtype=str),
            "label": np.asarray(z["collapsed_labels"], dtype=np.int64),
            "prediction": np.asarray(z["collapsed_predictions"], dtype=np.int64),
        })
    metadata = pd.read_csv(METADATA, usecols=["record_key", "qa_flag_low_vegetated_height_with_lidar"])
    frame = frame.merge(metadata, on="record_key", validate="one_to_one")
    frame = frame.loc[~frame.qa_flag_low_vegetated_height_with_lidar.astype(bool)].copy()
    rows = []
    for city, group in frame.groupby("city", sort=True):
        support = group.label.value_counts()
        labels = sorted(support.loc[support.ge(10)].index.tolist())
        rows.append({
            "city": city,
            "qa_screened_test_records": len(group),
            "groups_with_at_least_10_records": len(labels),
            "macro_f1_groups_ge10": f1_score(
                group.label, group.prediction, labels=labels, average="macro", zero_division=0
            ),
            "accuracy": float((group.label == group.prediction).mean()),
        })
    ranking = pd.DataFrame(rows)
    ranking = ranking.loc[ranking.qa_screened_test_records.ge(1000)].sort_values(
        ["macro_f1_groups_ge10", "city"]
    ).reset_index(drop=True)
    ranking["rank"] = np.arange(1, len(ranking) + 1)
    median = float(ranking.macro_f1_groups_ge10.median())
    ranking["distance_from_median"] = (ranking.macro_f1_groups_ge10 - median).abs()
    selection = ranking.sort_values(
        ["distance_from_median", "qa_screened_test_records", "city"],
        ascending=[True, False, True],
    ).iloc[0]
    ranking.to_csv(OUTPUT / "eligible_city_primary_model_ranking.csv", index=False)
    result = {
        "selection_source": "final primary national model; leave-one-city-out and one-city-trained results not used",
        "minimum_qa_screened_test_records": 1000,
        "macro_f1_rule": "unweighted mean of collapsed-k6 group F1 values for groups with at least 10 city test records",
        "eligible_cities": int(len(ranking)),
        "median_city_macro_f1": median,
        "tie_breaker": "larger QA-screened test support, then alphabetical city token",
        "selected_city": str(selection.city),
        "selected_city_rank": int(selection["rank"]),
        "selected_city_macro_f1": float(selection.macro_f1_groups_ge10),
        "selected_city_accuracy": float(selection.accuracy),
        "selected_city_qa_screened_test_records": int(selection.qa_screened_test_records),
        "selected_city_groups_with_at_least_10_records": int(selection.groups_with_at_least_10_records),
    }
    (OUTPUT / "selection.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(ranking.to_string(index=False))
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
