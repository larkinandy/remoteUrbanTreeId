#!/usr/bin/env python3
"""Compare primary, leave-one-city-out, and St. Louis-trained models by city."""

from __future__ import annotations

import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from geographic_utils import load_predictions, load_qa_flags, screen_qa


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "outputs" / "reviewer_revision" / "completed_city_three_regime_comparison"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary-predictions", type=Path, required=True)
    parser.add_argument("--loco-root", type=Path, required=True)
    parser.add_argument("--one-city-transfer-predictions", type=Path, required=True)
    parser.add_argument("--qa-metadata", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def metrics(labels: np.ndarray, predictions: np.ndarray, class_count: int) -> dict[str, float | int | str]:
    support = np.bincount(labels, minlength=class_count)
    represented = np.flatnonzero(support > 0)
    reliable = np.flatnonzero(support >= 10)
    return {
        "records": int(len(labels)),
        "represented_groups": int(len(represented)),
        "groups_with_at_least_10_records": int(len(reliable)),
        "groups_in_restricted_macro_f1": ",".join(str(int(value) + 1) for value in reliable),
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1_groups_ge10": float(f1_score(labels, predictions, labels=reliable, average="macro", zero_division=0)),
        "macro_f1_unrestricted": float(f1_score(labels, predictions, labels=represented, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(labels, predictions, labels=represented, average="weighted", zero_division=0)),
    }


def main() -> int:
    arguments = parse_args()
    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    primary = load_predictions(arguments.primary_predictions, "primary")
    stlouis = load_predictions(arguments.one_city_transfer_predictions, "stlouis")
    qa = load_qa_flags(arguments.qa_metadata)
    rows = []
    audits = []
    for directory in sorted(arguments.loco_root.glob("loco_*_test_top5_ensemble")):
        path = directory / "ensemble_predictions.npz"
        if not path.exists():
            continue
        city = directory.name.removeprefix("loco_").removesuffix("_test_top5_ensemble")
        loco = load_predictions(path, "loco")
        if set(loco.city) != {city}:
            raise ValueError(f"Unexpected cities in {path}: {sorted(set(loco.city))}")
        p = primary.loc[primary.city.eq(city)].copy()
        s = stlouis.loc[stlouis.city.eq(city)].copy()
        joined = p.merge(loco.drop(columns="city"), on="record_key", validate="one_to_one")
        joined = joined.merge(s.drop(columns="city"), on="record_key", validate="one_to_one")
        joined = joined.merge(qa, on="record_key", validate="one_to_one")
        audits.append({
            "city": city,
            "primary_records": len(p),
            "loco_records": len(loco),
            "stlouis_records": len(s),
            "joined_records": len(joined),
            "record_sets_exact": len(p) == len(loco) == len(s) == len(joined),
            "k11_labels_exact": bool(
                np.array_equal(joined.k11_label_primary, joined.k11_label_loco)
                and np.array_equal(joined.k11_label_primary, joined.k11_label_stlouis)
            ),
            "k6_labels_exact": bool(
                np.array_equal(joined.collapsed_k6_label_primary, joined.collapsed_k6_label_loco)
                and np.array_equal(joined.collapsed_k6_label_primary, joined.collapsed_k6_label_stlouis)
            ),
        })
        joined = screen_qa(joined)
        for regime in ("primary", "loco", "stlouis"):
            for space, class_count in (("k11", 11), ("collapsed_k6", 6)):
                result = metrics(
                    joined[f"{space}_label_{regime}"].to_numpy(),
                    joined[f"{space}_prediction_{regime}"].to_numpy(),
                    class_count,
                )
                rows.append({"city": city, "regime": regime, "label_space": space, **result})
    audit = pd.DataFrame(audits)
    if not audit[["record_sets_exact", "k11_labels_exact", "k6_labels_exact"]].all().all():
        raise ValueError(f"Comparability audit failed:\n{audit.to_string(index=False)}")
    results = pd.DataFrame(rows)
    results.to_csv(arguments.output_dir / "three_regime_city_metrics_long.csv", index=False)
    audit.to_csv(arguments.output_dir / "three_regime_comparability_audit.csv", index=False)
    k6 = results.loc[results.label_space.eq("collapsed_k6")].copy()
    wide = k6.pivot(index="city", columns="regime", values=[
        "records", "represented_groups", "groups_with_at_least_10_records", "accuracy",
        "macro_f1_groups_ge10", "macro_f1_unrestricted", "weighted_f1",
    ])
    wide.columns = [f"{metric}_{regime}" for metric, regime in wide.columns]
    wide = wide.reset_index()
    for regime in ("loco", "stlouis"):
        for metric in ("accuracy", "macro_f1_groups_ge10", "weighted_f1"):
            wide[f"delta_{metric}_{regime}_vs_primary"] = wide[f"{metric}_{regime}"] - wide[f"{metric}_primary"]
    wide.to_csv(arguments.output_dir / "three_regime_city_metrics_collapsed_k6_wide.csv", index=False)
    regime_summary = k6.groupby("regime").agg(
        cities=("city", "nunique"),
        mean_accuracy=("accuracy", "mean"), median_accuracy=("accuracy", "median"),
        mean_macro_f1=("macro_f1_groups_ge10", "mean"), median_macro_f1=("macro_f1_groups_ge10", "median"),
        mean_weighted_f1=("weighted_f1", "mean"), median_weighted_f1=("weighted_f1", "median"),
    ).reset_index()
    regime_summary.to_csv(arguments.output_dir / "three_regime_summary_across_completed_cities.csv", index=False)
    metadata = {
        "completed_cities": sorted(audit.city.tolist()),
        "city_count": len(audit),
        "qa_screen": "excluded qa_flag_low_vegetated_height_with_lidar records",
        "primary_macro_f1": "collapsed_k6 macro-F1 across groups with at least 10 city test records",
        "weighted_f1": "calculated across represented collapsed_k6 groups",
        "record_and_label_comparability_checks_passed": True,
    }
    (arguments.output_dir / "analysis_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(wide.to_string(index=False))
    print("\nAcross-city summary:\n", regime_summary.to_string(index=False))
    print("\nAudit:\n", audit.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
