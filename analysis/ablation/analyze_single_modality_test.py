#!/usr/bin/env python3
"""Create final independent-test summaries for frozen single-modality models."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support

from workflow import collapsed_k6_probabilities, metric_summary, record_keys

ROOT = Path(__file__).resolve().parents[2]


def provenance_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return f"external://{path.name}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--single-root", type=Path, default=Path("outputs/reviewer_revision/k11_single_modality_test_inference"))
    parser.add_argument("--reference", type=Path, default=Path("outputs/reviewer_revision/final_pruned_k11_test_top5_ensemble/ensemble_predictions.npz"))
    parser.add_argument(
        "--collapse",
        type=Path,
        default=ROOT / "config" / "final_k11_to_k6_collapse.npz",
        help="Frozen balanced 11-to-6 manuscript mapping.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/reviewer_revision/k11_single_modality_test_analysis"))
    return parser.parse_args()


def score(labels: np.ndarray, predictions: np.ndarray) -> dict[str, float | int]:
    return {"records": int(len(labels)), **metric_summary(labels, predictions)}


def group_table(labels: np.ndarray, predictions: np.ndarray, names: np.ndarray, modality: str, space: str) -> pd.DataFrame:
    precision, recall, f1, support = precision_recall_fscore_support(
        labels, predictions, labels=np.arange(len(names)), zero_division=0
    )
    return pd.DataFrame({
        "modality": modality,
        "label_space": space,
        "group": np.arange(1, len(names) + 1),
        "group_name": names,
        "support": support.astype(int),
        "precision": precision,
        "recall": recall,
        "f1": f1,
    })


def evaluate(
    path: Path, modality: str, collapse: np.ndarray | None = None
) -> tuple[list[dict], list[pd.DataFrame], dict]:
    with np.load(path, allow_pickle=True) as z:
        labels = np.asarray(z["labels"], dtype=np.int64)
        probabilities = np.asarray(z["probabilities"], dtype=np.float64)
        if collapse is None:
            collapsed_labels = np.asarray(z["collapsed_labels"], dtype=np.int64)
            collapsed_probabilities = np.asarray(z["collapsed_probabilities"], dtype=np.float64)
        else:
            collapsed_labels = collapse[labels]
            collapsed_probabilities = collapsed_k6_probabilities(probabilities, collapse)
        collapsed_predictions = collapsed_probabilities.argmax(axis=1)
        audit = {
            "keys": record_keys(z),
            "labels": labels,
            "collapsed_labels": collapsed_labels,
            "groups": np.asarray(z["group_names"], dtype=str),
            "collapsed_groups": np.asarray(z["collapsed_group_names"], dtype=str),
        }
        rows = [
            {"modality": modality, "label_space": "k11", **score(audit["labels"], np.asarray(z["predictions"], dtype=np.int64))},
            {"modality": modality, "label_space": "collapsed_k6", **score(audit["collapsed_labels"], collapsed_predictions)},
        ]
        tables = [
            group_table(audit["labels"], np.asarray(z["predictions"], dtype=np.int64), audit["groups"], modality, "k11"),
            group_table(audit["collapsed_labels"], collapsed_predictions, audit["collapsed_groups"], modality, "collapsed_k6"),
        ]
    return rows, tables, audit


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with np.load(args.collapse, allow_pickle=True) as artifact:
        collapse = np.asarray(artifact["initial_group_assignment"], dtype=np.int64)
    summary_rows, group_tables, reference = evaluate(
        args.reference, "final_pruned_multimodal_top5_ensemble", collapse
    )
    if len(collapse) != len(reference["groups"]):
        raise ValueError("Frozen collapse map does not match the reference class count")
    audits = []
    for directory in sorted(path for path in args.single_root.iterdir() if path.is_dir()):
        prediction = directory / "ensemble_predictions.npz"
        if not prediction.exists():
            continue
        rows, tables, current = evaluate(prediction, directory.name, collapse)
        summary_rows.extend(rows)
        group_tables.extend(tables)
        audit = {
            "modality": directory.name,
            "records": len(current["keys"]),
            "record_keys_exactly_match_reference": bool(np.array_equal(current["keys"], reference["keys"])),
            "labels_exactly_match_reference": bool(np.array_equal(current["labels"], reference["labels"])),
            "collapsed_labels_exactly_match_reference": bool(np.array_equal(current["collapsed_labels"], reference["collapsed_labels"])),
            "group_order_exactly_matches_reference": bool(np.array_equal(current["groups"], reference["groups"])),
            "collapsed_group_order_exactly_matches_reference": bool(np.array_equal(current["collapsed_groups"], reference["collapsed_groups"])),
            "unique_record_keys": int(len(np.unique(current["keys"]))),
        }
        if not all(value for key, value in audit.items() if key.endswith("reference")):
            raise ValueError(f"Test-set comparability failure for {directory.name}: {audit}")
        audits.append(audit)

    summary = pd.DataFrame(summary_rows)
    baseline = summary.loc[summary.modality.eq("final_pruned_multimodal_top5_ensemble")].set_index("label_space")
    for metric in ("accuracy", "macro_f1", "weighted_f1"):
        summary[f"delta_{metric}_vs_final_pruned"] = summary.apply(
            lambda row: row[metric] - baseline.loc[row.label_space, metric], axis=1
        )
    summary["rank_by_macro_f1"] = summary.groupby("label_space")["macro_f1"].rank(method="min", ascending=False).astype(int)
    summary.to_csv(args.output_dir / "single_modality_test_summary.csv", index=False)
    pd.concat(group_tables, ignore_index=True).to_csv(args.output_dir / "single_modality_test_per_group.csv", index=False)
    pd.DataFrame(audits).to_csv(args.output_dir / "single_modality_test_comparability_audit.csv", index=False)
    metadata = {
        "evaluation_split": "independent test",
        "single_modality_checkpoint_count": len(audits),
        "test_records": int(len(reference["keys"])),
        "reference": provenance_path(args.reference),
        "collapse_artifact": provenance_path(args.collapse),
        "collapse_map_zero_based": collapse.tolist(),
        "all_comparability_checks_passed": True,
    }
    (args.output_dir / "analysis_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(summary.sort_values(["label_space", "macro_f1"], ascending=[True, False]).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
