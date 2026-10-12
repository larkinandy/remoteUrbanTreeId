#!/usr/bin/env python3
"""Summarize QA-screened final predictions by taxon provenance and city.

The script verifies the canonical probability-sum collapse, joins finalized
test metadata by stable record key, removes the configured QA warning, and
reports identically named metrics for ``k11`` and ``collapsed_k6``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from sensitivity_utils import (
    load_configuration,
    load_predictions,
    file_sha256,
    merge_one_to_one,
    metric_set,
    screen_qa,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_GROUPING = ROOT / "config" / "final_grouping.json"
DEFAULT_CONFIG = ROOT / "analysis" / "sensitivity" / "sensitivity_config.json"


def parse_args() -> argparse.Namespace:
    """Parse finalized prediction, metadata, configuration, and output paths."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True, help="Final top-checkpoint ensemble NPZ.")
    parser.add_argument("--sensitivity-variables", type=Path, required=True, help="Record-keyed QA and taxon-provenance CSV.")
    parser.add_argument("--grouping-config", type=Path, default=DEFAULT_GROUPING, help="Canonical k11-to-collapsed_k6 mapping JSON.")
    parser.add_argument("--sensitivity-config", type=Path, default=DEFAULT_CONFIG, help="Frozen sensitivity-analysis configuration JSON.")
    parser.add_argument("--output-dir", type=Path, required=True, help="External directory for generated summaries.")
    return parser.parse_args()


def summarize(frame: pd.DataFrame, grouping_column: str) -> pd.DataFrame:
    """Calculate canonical metrics for both label spaces within each group."""
    rows: list[dict[str, object]] = []
    for group_name, group in frame.groupby(grouping_column, sort=True):
        row: dict[str, object] = {grouping_column: group_name, "records": len(group)}
        for label_space, class_count in (("k11", 11), ("collapsed_k6", 6)):
            values = metric_set(
                group[f"label_{label_space}"].to_numpy(),
                group[f"prediction_{label_space}"].to_numpy(),
                class_count,
            )
            row.update({f"{label_space}_{key}": value for key, value in values.items()})
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> int:
    """Create provenance, city, confusion, and portable audit summaries."""
    arguments = parse_args()
    configuration = load_configuration(arguments.sensitivity_config)
    qa_column = str(configuration["qa_warning_excluded"])
    metadata = pd.read_csv(
        arguments.sensitivity_variables,
        usecols=["record_key", qa_column, "frequency_category"],
    )
    frame = merge_one_to_one(
        load_predictions(arguments.predictions, arguments.grouping_config),
        metadata,
        "sensitivity variables",
    )
    total_records = len(frame)
    frame = screen_qa(frame, qa_column)
    provenance = configuration["taxon_provenance"]
    frame["taxon_provenance"] = np.where(
        frame.frequency_category.eq(provenance["core_value"]),
        provenance["core_label"],
        provenance["rare_label"],
    )
    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    by_provenance = summarize(frame, "taxon_provenance")
    by_city = summarize(frame, "city_token")
    by_provenance.to_csv(arguments.output_dir / "performance_by_taxon_provenance.csv", index=False)
    by_city.sort_values("collapsed_k6_macro_f1", ascending=False).to_csv(
        arguments.output_dir / "performance_by_city.csv", index=False
    )
    confusion = np.bincount(
        frame.label_collapsed_k6.to_numpy() * 6 + frame.prediction_collapsed_k6.to_numpy(),
        minlength=36,
    ).reshape(6, 6)
    pd.DataFrame(
        confusion,
        index=[f"true_{index + 1}" for index in range(6)],
        columns=[f"pred_{index + 1}" for index in range(6)],
    ).to_csv(arguments.output_dir / "collapsed_k6_confusion_counts.csv")
    audit = {
        "test_records_total": int(total_records),
        "qa_screened_records": int(len(frame)),
        "qa_warning_records_excluded": int(total_records - len(frame)),
        "canonical_grouping_validation_passed": True,
        "inputs": {
            "predictions": {
                "file_name": arguments.predictions.name,
                "sha256": file_sha256(arguments.predictions),
            },
            "sensitivity_variables": {
                "file_name": arguments.sensitivity_variables.name,
                "sha256": file_sha256(arguments.sensitivity_variables),
            },
            "grouping_config": {
                "file_name": arguments.grouping_config.name,
                "sha256": file_sha256(arguments.grouping_config),
            },
            "sensitivity_config": {
                "file_name": arguments.sensitivity_config.name,
                "sha256": file_sha256(arguments.sensitivity_config),
            },
        },
    }
    (arguments.output_dir / "analysis_metadata.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8"
    )
    print(by_provenance.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
