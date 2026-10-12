#!/usr/bin/env python3
"""Audit and summarize the seven-source one-city geographic-transfer experiment."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score


ROOT = Path(__file__).resolve().parents[2]
NEW_ROOT = Path(r"H:\k11_one_city_seven_source")
STL_ROOT = Path(r"H:\k11_one_city_stlouis")
PRIMARY = ROOT / "outputs" / "reviewer_revision" / "final_pruned_k11_test_top5_ensemble" / "ensemble_predictions.npz"
QA = ROOT / "outputs" / "reviewer_revision" / "final_pruned_sensitivity_table_with_taxon_provenance" / "final_test_sensitivity_variables.csv.gz"
OUTPUT = ROOT / "outputs" / "reviewer_revision" / "seven_one_city_analysis"
SOURCES = ("denver", "portland", "ranchocucamonga", "sanfrancisco", "sanjose", "seattle", "stlouis")


def load_predictions(path: Path, suffix: str) -> pd.DataFrame:
    with np.load(path, allow_pickle=True) as z:
        keys = np.asarray([
            f"{city}|{int(tree)}|{int(crop)}"
            for city, tree, crop in zip(z["city_token"], z["tree_id"], z["crop_index"])
        ])
        return pd.DataFrame({
            "record_key": keys,
            "city": np.asarray(z["city_token"], dtype=str),
            f"k11_label_{suffix}": np.asarray(z["labels"], dtype=np.int64),
            f"k11_prediction_{suffix}": np.asarray(z["predictions"], dtype=np.int64),
            f"k6_label_{suffix}": np.asarray(z["collapsed_labels"], dtype=np.int64),
            f"k6_prediction_{suffix}": np.asarray(z["collapsed_predictions"], dtype=np.int64),
        })


def metric_set(labels: np.ndarray, predictions: np.ndarray, class_count: int) -> dict[str, float | int | str]:
    support = np.bincount(labels, minlength=class_count)
    represented = np.flatnonzero(support > 0)
    reliable = np.flatnonzero(support >= 10)
    return {
        "records": int(len(labels)),
        "represented_groups": int(len(represented)),
        "groups_ge10": int(len(reliable)),
        "groups_in_macro_f1": ",".join(str(int(value) + 1) for value in reliable),
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1_groups_ge10": float(f1_score(labels, predictions, labels=reliable, average="macro", zero_division=0)),
        "macro_f1_represented": float(f1_score(labels, predictions, labels=represented, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(labels, predictions, labels=represented, average="weighted", zero_division=0)),
    }


def source_paths(source: str) -> tuple[Path, Path]:
    if source == "stlouis":
        return (
            STL_ROOT / "results" / "stlouis_within_city_test_top5" / "ensemble_predictions.npz",
            STL_ROOT / "results" / "stlouis_trained_other_cities_test_top5" / "ensemble_predictions.npz",
        )
    return (
        NEW_ROOT / "results" / f"{source}_within_city_test_top5" / "ensemble_predictions.npz",
        NEW_ROOT / "results" / f"{source}_to_22_cities_test_top5" / "ensemble_predictions.npz",
    )


def main() -> int:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    primary = load_predictions(PRIMARY, "primary")
    eligible_counts = primary["city"].value_counts()
    targets = sorted(eligible_counts.loc[eligible_counts.ge(1000)].index.tolist())
    qa = pd.read_csv(QA, usecols=["record_key", "qa_flag_low_vegetated_height_with_lidar"])

    metric_rows: list[dict] = []
    audits: list[dict] = []
    source_frames: dict[str, pd.DataFrame] = {}
    for source in SOURCES:
        within_path, cross_path = source_paths(source)
        for path in (within_path, cross_path):
            if not path.exists():
                raise FileNotFoundError(path)
        within = load_predictions(within_path, "source")
        cross = load_predictions(cross_path, "source")
        cross = cross.loc[cross["city"].isin(targets) & cross["city"].ne(source)].copy()
        predictions = pd.concat([within, cross], ignore_index=True)
        expected = primary.loc[primary["city"].isin(targets)].copy()
        joined = expected.merge(predictions.drop(columns="city"), on="record_key", how="inner", validate="one_to_one")
        joined = joined.merge(qa, on="record_key", validate="one_to_one")
        expected_by_city = expected.groupby("city").size().sort_index()
        actual_by_city = predictions.groupby("city").size().sort_index()
        audits.append({
            "source_city": source,
            "expected_records": len(expected),
            "prediction_records": len(predictions),
            "joined_records": len(joined),
            "target_city_set_exact": set(predictions.city) == set(targets),
            "records_per_city_exact": expected_by_city.equals(actual_by_city),
            "k11_labels_exact": bool(np.array_equal(joined.k11_label_primary, joined.k11_label_source)),
            "k6_labels_exact": bool(np.array_equal(joined.k6_label_primary, joined.k6_label_source)),
        })
        joined = joined.loc[~joined.qa_flag_low_vegetated_height_with_lidar.astype(bool)].copy()
        source_frames[source] = joined
        for target, frame in joined.groupby("city", sort=True):
            relation = "within_city" if target == source else "cross_city"
            for space, class_count in (("k11", 11), ("k6", 6)):
                for regime in ("source", "primary"):
                    values = metric_set(
                        frame[f"{space}_label_{regime}"].to_numpy(),
                        frame[f"{space}_prediction_{regime}"].to_numpy(),
                        class_count,
                    )
                    metric_rows.append({
                        "source_city": source, "target_city": target, "relation": relation,
                        "regime": "one_city" if regime == "source" else "primary",
                        "label_space": space, **values,
                    })

    audit = pd.DataFrame(audits)
    checks = ["target_city_set_exact", "records_per_city_exact", "k11_labels_exact", "k6_labels_exact"]
    if not audit[checks].all().all():
        raise RuntimeError(f"Comparability audit failed:\n{audit.to_string(index=False)}")
    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(OUTPUT / "source_target_metrics_long.csv", index=False)
    audit.to_csv(OUTPUT / "comparability_audit.csv", index=False)

    summary_rows: list[dict] = []
    for (source, regime, space), frame in metrics.groupby(["source_city", "regime", "label_space"], sort=True):
        within = frame.loc[frame.relation.eq("within_city")].iloc[0]
        cross = frame.loc[frame.relation.eq("cross_city")]
        pooled_frame = source_frames[source].loc[source_frames[source].city.ne(source)]
        suffix = "source" if regime == "one_city" else "primary"
        pooled = metric_set(
            pooled_frame[f"{space}_label_{suffix}"].to_numpy(),
            pooled_frame[f"{space}_prediction_{suffix}"].to_numpy(),
            11 if space == "k11" else 6,
        )
        summary_rows.append({
            "source_city": source, "regime": regime, "label_space": space,
            "within_accuracy": within.accuracy,
            "within_macro_f1": within.macro_f1_groups_ge10,
            "within_weighted_f1": within.weighted_f1,
            "cross_city_count": int(len(cross)),
            "cross_city_mean_accuracy": float(cross.accuracy.mean()),
            "cross_city_median_accuracy": float(cross.accuracy.median()),
            "cross_city_mean_macro_f1": float(cross.macro_f1_groups_ge10.mean()),
            "cross_city_median_macro_f1": float(cross.macro_f1_groups_ge10.median()),
            "cross_city_mean_weighted_f1": float(cross.weighted_f1.mean()),
            "cross_city_median_weighted_f1": float(cross.weighted_f1.median()),
            "cross_pooled_records": pooled["records"],
            "cross_pooled_accuracy": pooled["accuracy"],
            "cross_pooled_macro_f1": pooled["macro_f1_groups_ge10"],
            "cross_pooled_weighted_f1": pooled["weighted_f1"],
        })
    summaries = pd.DataFrame(summary_rows)
    summaries.to_csv(OUTPUT / "source_model_summaries.csv", index=False)

    k6 = summaries.loc[summaries.label_space.eq("k6")].copy()
    overall = k6.groupby("regime").agg(
        source_models=("source_city", "nunique"),
        mean_within_accuracy=("within_accuracy", "mean"),
        mean_within_macro_f1=("within_macro_f1", "mean"),
        mean_within_weighted_f1=("within_weighted_f1", "mean"),
        mean_source_balanced_cross_accuracy=("cross_city_mean_accuracy", "mean"),
        median_source_balanced_cross_accuracy=("cross_city_mean_accuracy", "median"),
        mean_source_balanced_cross_macro_f1=("cross_city_mean_macro_f1", "mean"),
        median_source_balanced_cross_macro_f1=("cross_city_mean_macro_f1", "median"),
        mean_source_balanced_cross_weighted_f1=("cross_city_mean_weighted_f1", "mean"),
        median_source_balanced_cross_weighted_f1=("cross_city_mean_weighted_f1", "median"),
    ).reset_index()
    overall.to_csv(OUTPUT / "overall_k6_summary.csv", index=False)

    one = k6.loc[k6.regime.eq("one_city")].copy()
    prim = k6.loc[k6.regime.eq("primary")].copy()
    comparison = one.merge(prim, on=["source_city", "label_space"], suffixes=("_one_city", "_primary"))
    for metric in ("cross_city_mean_accuracy", "cross_city_mean_macro_f1", "cross_city_mean_weighted_f1"):
        comparison[f"delta_{metric}"] = comparison[f"{metric}_one_city"] - comparison[f"{metric}_primary"]
    comparison.to_csv(OUTPUT / "one_city_vs_primary_k6.csv", index=False)

    metadata = {
        "source_cities": list(SOURCES),
        "target_cities": targets,
        "source_models": len(SOURCES),
        "target_cities_per_source": len(targets),
        "cross_city_targets_per_source": len(targets) - 1,
        "qa_screen": "excluded qa_flag_low_vegetated_height_with_lidar records",
        "macro_f1_rule": "calculated over target-city groups with at least 10 screened test records",
        "primary_summary": "source-balanced: target cities averaged within source, then source models averaged equally",
        "comparability_checks_passed": True,
    }
    (OUTPUT / "analysis_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print("AUDIT\n", audit.to_string(index=False))
    print("\nK6 SOURCE SUMMARIES\n", k6.to_string(index=False))
    print("\nOVERALL K6\n", overall.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
