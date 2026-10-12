"""Create publication summaries from prepared geographic metric tables."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def write_one_city_reports(
    summaries: pd.DataFrame,
    output_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Write source-balanced collapsed_k6 summaries without reopening predictions."""
    collapsed_k6 = summaries.loc[summaries.label_space.eq("collapsed_k6")].copy()
    overall = collapsed_k6.groupby("regime").agg(
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
    one_city = collapsed_k6.loc[collapsed_k6.regime.eq("one_city")]
    primary = collapsed_k6.loc[collapsed_k6.regime.eq("primary")]
    comparison = one_city.merge(
        primary, on=["source_city", "label_space"], suffixes=("_one_city", "_primary")
    )
    for metric in (
        "cross_city_mean_accuracy", "cross_city_mean_macro_f1", "cross_city_mean_weighted_f1"
    ):
        comparison[f"delta_{metric}"] = (
            comparison[f"{metric}_one_city"] - comparison[f"{metric}_primary"]
        )
    overall.to_csv(output_dir / "overall_collapsed_k6_summary.csv", index=False)
    comparison.to_csv(output_dir / "one_city_vs_primary_collapsed_k6.csv", index=False)
    return overall, comparison
