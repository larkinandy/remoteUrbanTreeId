#!/usr/bin/env python3
"""Compare default, leave-one-city-out, and one-city training on matched cities."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "outputs" / "manuscript_analysis" / "geographic_training_regime_comparison"
METRICS = ("accuracy", "macro_f1", "weighted_f1")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--loco-summary",
        type=Path,
        required=True,
        help="Final leave-one-city-out city-summary table.",
    )
    parser.add_argument(
        "--one-city-long",
        type=Path,
        required=True,
        help="Final source-by-target single-city performance table.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def source_balanced_summary(frame: pd.DataFrame) -> dict[str, float]:
    per_source = frame.groupby("source_city", as_index=False)[list(METRICS)].mean()
    return {
        **{f"mean_{metric}": float(per_source[metric].mean()) for metric in METRICS},
        **{f"median_{metric}": float(frame[metric].median()) for metric in METRICS},
    }


def difference_statistics(values: pd.Series, seed: int = 2026) -> dict[str, float | int]:
    """Return descriptive statistics and a percentile bootstrap CI for a paired loss."""
    array = values.dropna().to_numpy(dtype=float)
    if not len(array):
        raise ValueError("Cannot summarize an empty difference vector.")
    rng = np.random.default_rng(seed)
    bootstrap_means = rng.choice(array, size=(10_000, len(array)), replace=True).mean(axis=1)
    return {
        "units": int(len(array)),
        "mean": float(array.mean()),
        "sd": float(array.std(ddof=1)) if len(array) > 1 else 0.0,
        "median": float(np.median(array)),
        "q1": float(np.quantile(array, 0.25)),
        "q3": float(np.quantile(array, 0.75)),
        "minimum": float(array.min()),
        "maximum": float(array.max()),
        "mean_ci95_low": float(np.quantile(bootstrap_means, 0.025)),
        "mean_ci95_high": float(np.quantile(bootstrap_means, 0.975)),
    }


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    loco_raw = pd.read_csv(args.loco_summary)
    loco = loco_raw.rename(columns={
        "city": "target_city",
        "collapsed_k6_records": "records",
        "collapsed_k6_accuracy": "accuracy",
        "collapsed_k6_macro_f1_groups_ge_minimum": "macro_f1",
        "collapsed_k6_weighted_f1": "weighted_f1",
    })[["target_city", "records", *METRICS]]

    long = pd.read_csv(args.one_city_long)
    cross = long.loc[
        long["label_space"].eq("collapsed_k6") & long["relation"].eq("cross_city")
    ].rename(columns={"macro_f1_groups_ge_minimum": "macro_f1"})
    cross = cross[["source_city", "target_city", "regime", "records", *METRICS]]
    matched_cities = set(loco["target_city"])
    cross = cross.loc[cross["target_city"].isin(matched_cities)].copy()

    expected = len(cross["source_city"].unique()) * (len(matched_cities) - 1)
    for regime in ("primary", "one_city"):
        count = len(cross.loc[cross["regime"].eq(regime)])
        if count != expected:
            raise ValueError(f"Expected {expected} {regime} source-target evaluations; found {count}")

    sources = sorted(cross["source_city"].unique())
    loco_repeated = pd.concat([
        loco.loc[~loco["target_city"].eq(source)].assign(source_city=source, regime="loco")
        for source in sources
    ], ignore_index=True)

    regime_frames = {
        "Default multimodal": cross.loc[cross["regime"].eq("primary")],
        "Leave one city out": loco_repeated,
        "One-city trained": cross.loc[cross["regime"].eq("one_city")],
    }
    training_scope = {
        "Default multimodal": "All training cities",
        "Leave one city out": "All cities except target city",
        "One-city trained": "One source city; evaluated on other cities",
    }
    rows = []
    source_balanced_rows = []
    for regime, frame in regime_frames.items():
        summary = source_balanced_summary(frame)
        per_source = frame.groupby("source_city", as_index=False)[list(METRICS)].mean()
        per_source.insert(0, "model", regime)
        source_balanced_rows.append(per_source)
        rows.append({
            "model": regime,
            "training_scope": training_scope[regime],
            "source_models": 1 if regime == "Default multimodal" else (len(sources) if regime == "One-city trained" else len(matched_cities)),
            "target_cities": len(matched_cities),
            "source_target_evaluations": len(frame),
            "unique_test_records": int(loco["records"].sum()),
            **summary,
        })
    summary_table = pd.DataFrame(rows)
    pd.concat(source_balanced_rows, ignore_index=True).to_csv(
        args.output_dir / "source_balanced_regime_performance.csv", index=False
    )

    target_city_rows = []
    for regime, frame in regime_frames.items():
        per_target = frame.groupby("target_city", as_index=False)[list(METRICS)].mean()
        per_target.insert(0, "model", regime)
        target_city_rows.append(per_target)
    pd.concat(target_city_rows, ignore_index=True).to_csv(
        args.output_dir / "target_city_regime_performance.csv", index=False
    )
    reference = summary_table.loc[summary_table["model"].eq("Default multimodal")].iloc[0]
    for metric in METRICS:
        summary_table[f"delta_mean_{metric}_vs_default"] = (
            summary_table[f"mean_{metric}"] - float(reference[f"mean_{metric}"])
        )
    summary_table.to_csv(args.output_dir / "default_loco_one_city_summary.csv", index=False)
    manuscript = summary_table[[
        "model", "training_scope", "source_models", "target_cities", "unique_test_records",
        "mean_accuracy", "mean_macro_f1", "mean_weighted_f1",
        "delta_mean_accuracy_vs_default", "delta_mean_macro_f1_vs_default",
        "delta_mean_weighted_f1_vs_default",
    ]].copy()
    manuscript.to_csv(args.output_dir / "default_loco_one_city_summary_manuscript.csv", index=False)

    primary_by_city = (
        cross.loc[cross["regime"].eq("primary")]
        .groupby("target_city", as_index=False)[["records", *METRICS]].first()
        .rename(columns={metric: f"default_{metric}" for metric in METRICS})
    )
    comparison = loco.merge(primary_by_city, on=["target_city", "records"], validate="one_to_one")
    for metric in METRICS:
        comparison[f"delta_{metric}_loco_vs_default"] = comparison[metric] - comparison[f"default_{metric}"]
    comparison.to_csv(args.output_dir / "loco_vs_default_by_city.csv", index=False)

    difference_rows = []
    for metric in METRICS:
        loco_loss = comparison[f"default_{metric}"] - comparison[metric]
        difference_rows.append({
            "sensitivity_model": "Leave one city out",
            "summary_unit": "target city",
            "metric": metric,
            **difference_statistics(loco_loss),
        })

    one_city_pairs = cross.pivot(
        index=["source_city", "target_city"], columns="regime", values=list(METRICS)
    )
    for metric in METRICS:
        paired_loss = one_city_pairs[(metric, "primary")] - one_city_pairs[(metric, "one_city")]
        source_mean_loss = paired_loss.groupby(level="source_city").mean()
        difference_rows.append({
            "sensitivity_model": "One-city trained",
            "summary_unit": "source-model mean across 21 target cities",
            "metric": metric,
            **difference_statistics(source_mean_loss),
        })
    difference_summary = pd.DataFrame(difference_rows)
    difference_summary.to_csv(args.output_dir / "default_minus_sensitivity_difference_summary.csv", index=False)

    metadata = {
        "matched_cities": sorted(matched_cities),
        "city_count": len(matched_cities),
        "unique_test_records": int(loco["records"].sum()),
        "qa_screen": "Records with the low vegetation-height QA flag were excluded.",
        "macro_f1_rule": "Groups required at least 10 screened records within each target city.",
        "summary_weighting": "City metrics were averaged within source model, then source models were weighted equally.",
        "loco_vs_default": {
            f"mean_delta_{metric}": float(comparison[f"delta_{metric}_loco_vs_default"].mean())
            for metric in METRICS
        } | {
            f"cities_loco_better_{metric}": int((comparison[f"delta_{metric}_loco_vs_default"] > 0).sum())
            for metric in METRICS
        },
    }
    (args.output_dir / "analysis_summary.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(summary_table.to_string(index=False))
    print("\nLOCO versus default:\n", comparison.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
