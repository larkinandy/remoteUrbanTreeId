#!/usr/bin/env python3
"""Evaluate frozen discriminator groupings on independent test embeddings."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support


def parse_args() -> argparse.Namespace:
    config_dir = Path(__file__).resolve().parents[2] / "config"
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument(
        "--validation-embeddings",
        type=Path,
        required=True,
        help="Validation embeddings used to construct the frozen partition and prototypes.",
    )
    p.add_argument(
        "--test-embeddings",
        type=Path,
        required=True,
        help="Independent test embeddings; never used to select group count or assignments.",
    )
    p.add_argument(
        "--k11-partition",
        type=Path,
        default=config_dir / "core_k11_partition.npz",
    )
    p.add_argument(
        "--k11-prototypes",
        type=Path,
        default=config_dir / "core_k11_group_prototypes.npz",
    )
    p.add_argument(
        "--collapse",
        type=Path,
        default=config_dir / "final_k11_to_k6_collapse.npz",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )
    return p.parse_args()


def normalize_rows(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def keys(z: np.lib.npyio.NpzFile) -> np.ndarray:
    return np.asarray(
        [f"{c}|{int(t)}|{int(i)}" for c, t, i in zip(z["city_token"], z["tree_id"], z["crop_index"])],
        dtype=str,
    )


def metric_set(y: np.ndarray, pred: np.ndarray) -> dict[str, float | int]:
    return {
        "records": int(len(y)),
        "accuracy": float(accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y, pred, average="weighted", zero_division=0)),
        "minimum_group_f1": float(
            precision_recall_fscore_support(y, pred, labels=np.arange(int(max(y.max(), pred.max())) + 1), zero_division=0)[2].min()
        ),
    }


def group_table(y: np.ndarray, pred: np.ndarray, label_space: str, method: str, groups: int) -> pd.DataFrame:
    precision, recall, f1, support = precision_recall_fscore_support(
        y, pred, labels=np.arange(groups), zero_division=0
    )
    predicted = np.bincount(pred, minlength=groups)
    return pd.DataFrame(
        {
            "label_space": label_space,
            "prediction_method": method,
            "group": np.arange(1, groups + 1),
            "support": support.astype(int),
            "predicted": predicted.astype(int),
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
    )


def weighted_group_prototypes(
    taxon_centroids: np.ndarray,
    assignments: np.ndarray,
    validation_counts: np.ndarray,
    groups: int,
) -> np.ndarray:
    centroids = normalize_rows(taxon_centroids)
    weights = np.sqrt(np.asarray(validation_counts, dtype=np.float64))
    out = []
    for group in range(groups):
        mask = assignments == group
        value = np.average(centroids[mask], axis=0, weights=weights[mask])
        out.append(value / max(np.linalg.norm(value), 1e-12))
    return np.asarray(out)


def nearest_prototype(embeddings: np.ndarray, prototypes: np.ndarray) -> np.ndarray:
    """Return cosine-nearest prototypes without relying on a large BLAS call."""
    scores = np.empty((len(embeddings), len(prototypes)), dtype=np.float64)
    for group, prototype in enumerate(prototypes):
        scores[:, group] = np.sum(embeddings * prototype, axis=1)
    return np.argmax(scores, axis=1).astype(np.int64)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    with np.load(args.validation_embeddings, allow_pickle=False) as val, np.load(args.test_embeddings, allow_pickle=False) as test:
        val_labels = np.asarray(val["label_columns"], dtype=str)
        test_labels = np.asarray(test["label_columns"], dtype=str)
        if not np.array_equal(val_labels, test_labels):
            raise ValueError("Validation and test taxon label columns differ")
        val_keys = keys(val)
        test_keys = keys(test)
        test_embedding = normalize_rows(test["embeddings"])
        test_taxon_index = np.asarray(test["labels"], dtype=np.int64)
        test_city = np.asarray(test["city_token"], dtype=str)
        test_tree = np.asarray(test["tree_id"], dtype=np.int64)
        test_crop = np.asarray(test["crop_index"], dtype=np.int64)

    with np.load(args.k11_partition, allow_pickle=False) as part:
        if not np.array_equal(np.asarray(part["label_columns"], dtype=str), val_labels):
            raise ValueError("k=11 partition taxon order differs from embedding labels")
        assignment_k11 = np.asarray(part["assignments"], dtype=np.int64)
        validation_counts = np.asarray(part["validation_counts"], dtype=np.int64)
        taxon_centroids = np.asarray(part["full_validation_centroids"], dtype=np.float64)

    with np.load(args.k11_prototypes, allow_pickle=False) as proto:
        prototypes_k11 = normalize_rows(proto["group_centroids"])
        if not np.array_equal(np.asarray(proto["assignments"], dtype=np.int64), assignment_k11):
            raise ValueError("Stored k=11 prototypes do not match the frozen partition")

    with np.load(args.collapse, allow_pickle=False) as collapsed:
        k11_to_k6 = np.asarray(collapsed["initial_group_assignment"], dtype=np.int64)
    if len(k11_to_k6) != 11 or set(k11_to_k6.tolist()) != set(range(6)):
        raise ValueError("Expected a complete zero-based 11-to-6 mapping")

    assignment_k6 = k11_to_k6[assignment_k11]
    prototypes_k6 = weighted_group_prototypes(taxon_centroids, assignment_k6, validation_counts, 6)
    true_k11 = assignment_k11[test_taxon_index]
    true_k6 = assignment_k6[test_taxon_index]
    pred_k11 = nearest_prototype(test_embedding, prototypes_k11)
    pred_k6_direct = nearest_prototype(test_embedding, prototypes_k6)
    pred_k6_hierarchical = k11_to_k6[pred_k11]

    evaluations = [
        ("k11", "direct_nearest_validation_prototype", true_k11, pred_k11, 11),
        ("k6", "direct_nearest_collapsed_validation_prototype", true_k6, pred_k6_direct, 6),
        ("k6", "collapse_of_nearest_k11_prediction", true_k6, pred_k6_hierarchical, 6),
    ]
    summary_rows = []
    per_group = []
    for label_space, method, y, pred, groups in evaluations:
        summary_rows.append({"label_space": label_space, "prediction_method": method, **metric_set(y, pred)})
        per_group.append(group_table(y, pred, label_space, method, groups))
        cm = confusion_matrix(y, pred, labels=np.arange(groups))
        pd.DataFrame(cm, index=np.arange(1, groups + 1), columns=np.arange(1, groups + 1)).rename_axis(
            "true_group"
        ).to_csv(args.output_dir / f"confusion_{label_space}_{method}.csv")

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(args.output_dir / "independent_test_summary.csv", index=False)
    pd.concat(per_group, ignore_index=True).to_csv(args.output_dir / "independent_test_per_group_metrics.csv", index=False)

    records = pd.DataFrame(
        {
            "record_key": test_keys,
            "city_token": test_city,
            "tree_id": test_tree,
            "crop_index": test_crop,
            "taxon": val_labels[test_taxon_index],
            "true_k11": true_k11 + 1,
            "predicted_k11": pred_k11 + 1,
            "true_k6": true_k6 + 1,
            "predicted_k6_direct": pred_k6_direct + 1,
            "predicted_k6_hierarchical": pred_k6_hierarchical + 1,
        }
    )
    records.to_csv(args.output_dir / "independent_test_predictions.csv.gz", index=False, compression="gzip")

    overlap = np.intersect1d(val_keys, test_keys)
    audit = {
        "validation_records": int(len(val_keys)),
        "test_records": int(len(test_keys)),
        "validation_unique_keys": int(len(np.unique(val_keys))),
        "test_unique_keys": int(len(np.unique(test_keys))),
        "validation_test_overlap": int(len(overlap)),
        "core_taxa": int(len(val_labels)),
        "k11_groups": 11,
        "k6_groups": 6,
        "test_data_used_for_group_selection": False,
        "prototype_source": "validation taxon centroids only; sqrt(validation support)-weighted group means",
        "source_sha256": {
            "validation_embeddings": sha256(args.validation_embeddings),
            "test_embeddings": sha256(args.test_embeddings),
            "k11_partition": sha256(args.k11_partition),
            "k11_prototypes": sha256(args.k11_prototypes),
            "collapse": sha256(args.collapse),
        },
    }
    (args.output_dir / "independent_test_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    np.savez_compressed(
        args.output_dir / "frozen_grouping_materials.npz",
        k11_to_k6=k11_to_k6,
        taxon_assignment_k11=assignment_k11,
        taxon_assignment_k6=assignment_k6,
        prototypes_k11=prototypes_k11,
        prototypes_k6=prototypes_k6,
        label_columns=val_labels,
    )
    print(summary.to_string(index=False))
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
