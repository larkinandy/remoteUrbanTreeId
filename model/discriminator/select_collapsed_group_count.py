#!/usr/bin/env python3
"""Apply the prespecified stopping rule to nested variance-aware partitions.

The input table must contain validation-only diagnostics for consecutive values
of k. This script never reads record-level test data. It reports the selected k
for lambda = 0.5, 1, and 2 rather than tuning lambda to obtain a preferred k.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = {
    "k", "accuracy", "macro_f1", "minimum_group_f1",
    "minimum_separation", "mean_separation", "maximum_record_share",
    "maximum_taxon_share",
}
DEFAULT_LAMBDAS = (0.5, 1.0, 2.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-metrics", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--minimum-k", type=int, default=5)
    parser.add_argument("--maximum-k", type=int, default=10)
    parser.add_argument("--maximum-record-share", type=float, default=0.5)
    parser.add_argument("--maximum-taxon-share", type=float, default=0.5)
    parser.add_argument("--lambda", dest="lambdas", type=float, action="append")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def first_failed_split(frame: pd.DataFrame, score_column: str) -> int:
    eligible = frame.sort_values("k")
    selected = int(eligible.iloc[0]["k"])
    for row in eligible.iloc[1:].itertuples(index=False):
        if float(getattr(row, score_column)) <= 0:
            break
        selected = int(row.k)
    return selected


def main() -> int:
    args = parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.overwrite:
        raise SystemExit(f"Output directory is not empty; pass --overwrite: {args.output_dir}")
    frame = pd.read_csv(args.candidate_metrics)
    missing = sorted(REQUIRED_COLUMNS.difference(frame.columns))
    if missing:
        raise SystemExit(f"Candidate table is missing columns: {missing}")
    frame = frame.loc[frame["k"].between(args.minimum_k, args.maximum_k)].sort_values("k").copy()
    expected = list(range(args.minimum_k, args.maximum_k + 1))
    if frame["k"].astype(int).tolist() != expected:
        raise SystemExit(f"Expected one consecutive row for each k in {expected}")
    if (frame["maximum_record_share"] > args.maximum_record_share).any():
        raise SystemExit("At least one candidate exceeds the maximum record-share constraint")
    if (frame["maximum_taxon_share"] > args.maximum_taxon_share).any():
        raise SystemExit("At least one candidate exceeds the maximum taxon-share constraint")
    positive = (frame[["accuracy", "macro_f1", "minimum_group_f1", "minimum_separation", "mean_separation"]] > 0).all(axis=1)
    if not positive.all():
        raise SystemExit("Recoverability and separation inputs must be positive")

    frame["recoverability"] = np.cbrt(frame["accuracy"] * frame["macro_f1"] * frame["minimum_group_f1"])
    frame["separation"] = np.sqrt(frame["minimum_separation"] * frame["mean_separation"])
    lambdas = tuple(args.lambdas or DEFAULT_LAMBDAS)
    selections: dict[str, int] = {}
    for value in lambdas:
        column = f"j_lambda_{value:g}".replace(".", "p")
        frame[column] = np.nan
        frame.loc[frame.index[1:], column] = (
            np.log(frame["recoverability"].to_numpy()[1:] / frame["recoverability"].to_numpy()[:-1])
            + value * np.log(frame["separation"].to_numpy()[1:] / frame["separation"].to_numpy()[:-1])
        )
        selections[f"lambda_{value:g}"] = first_failed_split(frame, column)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output_dir / "collapsed_group_count_scores.csv", index=False)
    summary = {
        "test_data_used": False,
        "selection_rule": "start at minimum_k and stop at first J_k(lambda) <= 0",
        "maximum_record_share": args.maximum_record_share,
        "maximum_taxon_share": args.maximum_taxon_share,
        "selected_k": selections,
    }
    (args.output_dir / "collapsed_group_count_selection.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
