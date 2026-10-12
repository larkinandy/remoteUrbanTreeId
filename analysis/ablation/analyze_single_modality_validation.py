#!/usr/bin/env python
"""Consolidate development-set results from validation-only modality pilots.

These outputs support model development and are not independent-test metrics.
"""

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
    """Return a repository-relative path or an external artifact basename."""
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return f"external://{path.name}"


def args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--single-root", type=Path, required=True)
    p.add_argument(
        "--reference",
        type=Path,
        default=Path("outputs/reviewer_revision/k11_loo_without_structure_validation_top5_ensemble/ensemble_predictions.npz"),
    )
    p.add_argument(
        "--collapse",
        type=Path,
        default=ROOT / "config" / "final_k11_to_k6_collapse.npz",
        help="Frozen balanced 11-to-6 manuscript mapping.",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/reviewer_revision/k11_single_modality_analysis"),
    )
    return p.parse_args()


def metrics(y: np.ndarray, pred: np.ndarray) -> dict[str, float | int]:
    return {"records": int(len(y)), **metric_summary(y, pred)}


def per_group(y: np.ndarray, pred: np.ndarray, groups: int, modality: str, space: str) -> pd.DataFrame:
    precision, recall, f1, support = precision_recall_fscore_support(
        y, pred, labels=np.arange(groups), zero_division=0
    )
    return pd.DataFrame(
        {
            "modality": modality,
            "label_space": space,
            "group": np.arange(1, groups + 1),
            "support": support.astype(int),
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
    )


def main() -> None:
    a = args()
    a.output_dir.mkdir(parents=True, exist_ok=True)
    with np.load(a.collapse, allow_pickle=True) as z:
        collapse = np.asarray(z["initial_group_assignment"], dtype=np.int64)
    with np.load(a.reference, allow_pickle=True) as z:
        ref_keys = record_keys(z)
        ref_labels = np.asarray(z["labels"], dtype=np.int64)
        ref_probs = np.asarray(z["probabilities"], dtype=np.float64)
        ref_group_names = np.asarray(z["group_names"], dtype=str)

    def evaluate(name: str, labels: np.ndarray, probs: np.ndarray) -> tuple[list[dict], list[pd.DataFrame]]:
        pred11 = probs.argmax(axis=1)
        probs6 = collapsed_k6_probabilities(probs, collapse)
        labels6 = collapse[labels]
        pred6 = probs6.argmax(axis=1)
        rows = []
        tables = []
        for space, y, pred, groups in (("k11", labels, pred11, 11), ("collapsed_k6", labels6, pred6, 6)):
            rows.append({"modality": name, "label_space": space, **metrics(y, pred)})
            tables.append(per_group(y, pred, groups, name, space))
        return rows, tables

    summary_rows, group_tables = evaluate("final_pruned_multimodal_top5_ensemble", ref_labels, ref_probs)
    audit_rows = []
    training_rows = []
    selected_dirs: dict[str, Path] = {}
    for directory in sorted(a.single_root.glob("k11_single_*")):
        prediction_path = directory / "validation_predictions.npz"
        config_path = directory / "run_config.json"
        history_path = directory / "history.csv"
        if not prediction_path.exists() or not config_path.exists() or not history_path.exists():
            continue
        with np.load(prediction_path, allow_pickle=True) as z:
            modalities = np.asarray(z["modality_names"], dtype=str)
            if len(modalities) != 1:
                raise ValueError(f"Expected one modality in {directory}: {modalities.tolist()}")
            modality = str(modalities[0])
            current_keys = record_keys(z)
            current_labels = np.asarray(z["labels"], dtype=np.int64)
            current_probs = np.asarray(z["probabilities"], dtype=np.float64)
            current_groups = np.asarray(z["group_names"], dtype=str)
        if modality in selected_dirs:
            raise ValueError(f"Duplicate completed result for modality {modality}")
        selected_dirs[modality] = directory
        key_exact = np.array_equal(current_keys, ref_keys)
        label_exact = np.array_equal(current_labels, ref_labels)
        groups_exact = np.array_equal(current_groups, ref_group_names)
        if not (key_exact and label_exact and groups_exact):
            raise ValueError(f"Reference mismatch for {directory}")
        rows, tables = evaluate(modality, current_labels, current_probs)
        summary_rows.extend(rows)
        group_tables.extend(tables)

        config = json.loads(config_path.read_text(encoding="utf-8"))
        history = pd.read_csv(history_path)
        best_index = int(history["val_selection_score"].idxmax())
        best = history.loc[best_index]
        training_rows.append(
            {
                "modality": modality,
                "run_directory": provenance_path(directory),
                "best_epoch_config": int(config["best_epoch"]),
                "best_epoch_history": int(best["epoch"]),
                "epochs_completed": int(history["epoch"].max()),
                "best_validation_score": float(best["val_selection_score"]),
                "best_validation_accuracy": float(best["val_accuracy"]),
                "best_validation_macro_f1": float(best["val_macro_f1"]),
                "best_validation_weighted_f1": float(best["val_weighted_f1"]),
            }
        )
        audit_rows.append(
            {
                "modality": modality,
                "records": len(current_keys),
                "record_keys_exactly_match_reference": key_exact,
                "labels_exactly_match_reference": label_exact,
                "group_order_exactly_matches_reference": groups_exact,
                "unique_record_keys": len(np.unique(current_keys)),
                "validation_only_pilot": config.get("validation_only_pilot"),
                "test_evaluation_performed": config.get("test_evaluation_performed"),
                "partition_file": Path(config.get("partition_path", "unknown")).name,
                "seed": config.get("args", {}).get("seed"),
                "batch_size": config.get("args", {}).get("batch_size"),
            }
        )

    summary = pd.DataFrame(summary_rows)
    baseline = summary[summary.modality.eq("final_pruned_multimodal_top5_ensemble")].set_index("label_space")
    for metric in ("accuracy", "macro_f1", "weighted_f1"):
        summary[f"delta_{metric}_vs_final_pruned"] = summary.apply(
            lambda row: row[metric] - baseline.loc[row.label_space, metric], axis=1
        )
    summary["rank_by_macro_f1"] = summary.groupby("label_space")["macro_f1"].rank(method="min", ascending=False).astype(int)
    summary.to_csv(a.output_dir / "single_modality_validation_summary.csv", index=False)
    pd.concat(group_tables, ignore_index=True).to_csv(a.output_dir / "single_modality_validation_per_group.csv", index=False)
    pd.DataFrame(training_rows).sort_values("best_validation_score", ascending=False).to_csv(
        a.output_dir / "single_modality_training_summary.csv", index=False
    )
    audit = pd.DataFrame(audit_rows)
    audit.to_csv(a.output_dir / "single_modality_comparability_audit.csv", index=False)
    metadata = {
        "completed_modalities": sorted(selected_dirs),
        "completed_count": len(selected_dirs),
        "reference": provenance_path(a.reference),
        "collapse_artifact": provenance_path(a.collapse),
        "collapse": collapse.tolist(),
        "evaluation_split": "validation",
        "reference_is_top5_ensemble": True,
        "single_modality_predictions_are_best_checkpoint": True,
        "test_evaluation_available": bool(audit.test_evaluation_performed.fillna(False).any()),
    }
    (a.output_dir / "analysis_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(summary.sort_values(["label_space", "macro_f1"], ascending=[True, False]).to_string(index=False))
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
