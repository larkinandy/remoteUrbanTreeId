#!/usr/bin/env python3
"""Build and compare k=5--12 taxon groups from discriminator embeddings.

The discriminator embeds every tree on the unit hypersphere. This script
computes one normalized centroid per fine taxon, clusters those centroids with
sample-count-weighted k-means, and evaluates the resulting collapsed classes
with an Oracle nearest-group-prototype classifier. The selected partition is
the one with the strongest centroid separability objective, not necessarily
the partition with the fewest classes or highest raw accuracy.

The emitted ``kNN_partition.npz`` files implement the exact contract consumed
by ``train_clean_tree_id_centered_k6_classifier.py``.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_RUN_DIR = Path(
    r"H:\TreeCenteredModelInputs\taxon_discrimination_clean\clean_abq_atl_taxon_discriminator"
)
DEFAULT_OUTPUT_NAME = "global_centroid_taxon_partitions_clean_exploratory"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--validation-embeddings", type=Path, default=None)
    parser.add_argument("--test-embeddings", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--min-groups", type=int, default=5)
    parser.add_argument("--max-groups", type=int, default=12)
    parser.add_argument("--n-init", type=int, default=200)
    parser.add_argument("--max-iterations", type=int, default=200)
    parser.add_argument("--weight-power", type=float, default=0.5)
    parser.add_argument("--min-class-validation-samples", type=int, default=1)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def normalize_rows(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return np.divide(values, norms, out=np.zeros_like(values), where=norms > 1.0e-12)


def load_embeddings(path: Path) -> dict[str, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=True) as data:
        required = {"embeddings", "label_columns"}
        missing = sorted(required.difference(data.files))
        if missing:
            raise RuntimeError(f"{path} is missing {missing}")
        label_key = "labels" if "labels" in data.files else "label_index"
        if label_key not in data.files:
            raise RuntimeError(f"{path} has neither labels nor label_index")
        return {
            "embeddings": normalize_rows(np.asarray(data["embeddings"], dtype=np.float32)),
            "labels": np.asarray(data[label_key], dtype=np.int64),
            "label_columns": np.asarray(data["label_columns"]).astype(str),
        }


def validate_splits(validation: dict[str, np.ndarray], test: dict[str, np.ndarray]) -> None:
    if not np.array_equal(validation["label_columns"], test["label_columns"]):
        raise RuntimeError("Validation and test label_columns differ.")
    count = len(validation["label_columns"])
    for name, split in (("validation", validation), ("test", test)):
        labels = split["labels"]
        if labels.size and (labels.min() < 0 or labels.max() >= count):
            raise RuntimeError(f"{name} contains label indices outside [0, {count}).")


def taxon_centroids(
    split: dict[str, np.ndarray], min_samples: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    labels = split["labels"]
    embeddings = split["embeddings"]
    class_count = len(split["label_columns"])
    counts = np.bincount(labels, minlength=class_count).astype(np.int64)
    keep = np.flatnonzero(counts >= int(min_samples))
    if not len(keep):
        raise RuntimeError("No taxa satisfy --min-class-validation-samples.")
    centroids = np.stack([embeddings[labels == index].mean(axis=0) for index in keep])
    return keep, normalize_rows(centroids), counts


def fit_weighted_kmeans(
    centroids: np.ndarray,
    weights: np.ndarray,
    groups: int,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray]:
    try:
        from sklearn.cluster import KMeans
    except ModuleNotFoundError as error:
        raise SystemExit("scikit-learn is required: python -m pip install scikit-learn") from error
    model = KMeans(
        n_clusters=int(groups),
        n_init=int(args.n_init),
        max_iter=int(args.max_iterations),
        random_state=int(args.seed),
        algorithm="lloyd",
    )
    assignments = model.fit_predict(centroids, sample_weight=weights)
    # Recompute prototypes explicitly. This matches the saved historical files
    # and keeps their meaning independent of scikit-learn implementation detail.
    prototypes = np.stack(
        [np.average(centroids[assignments == group], axis=0, weights=weights[assignments == group]) for group in range(groups)]
    )
    return assignments.astype(np.int64), normalize_rows(prototypes)


def group_metrics(
    split: dict[str, np.ndarray],
    full_assignments: np.ndarray,
    prototypes: np.ndarray,
) -> tuple[dict[str, float], list[dict[str, Any]], np.ndarray]:
    true = full_assignments[split["labels"]]
    if (true < 0).any():
        keep = true >= 0
        true = true[keep]
        embeddings = split["embeddings"][keep]
    else:
        embeddings = split["embeddings"]
    predicted = np.argmax(embeddings @ prototypes.T, axis=1).astype(np.int64)
    rows: list[dict[str, Any]] = []
    f1_values: list[float] = []
    supports: list[int] = []
    confusion = np.zeros((len(prototypes), len(prototypes)), dtype=np.int64)
    np.add.at(confusion, (true, predicted), 1)
    for group in range(len(prototypes)):
        tp = int(confusion[group, group])
        true_n = int(confusion[group].sum())
        pred_n = int(confusion[:, group].sum())
        fp, fn = pred_n - tp, true_n - tp
        precision = tp / pred_n if pred_n else 0.0
        recall = tp / true_n if true_n else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
        rows.append(
            {
                "group_count": len(prototypes), "group": f"group{group + 1:02d}",
                "true_n": true_n, "pred_n": pred_n, "precision": precision,
                "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn,
            }
        )
        f1_values.append(f1)
        supports.append(true_n)
    total = int(confusion.sum())
    metrics = {
        "accuracy": float(np.trace(confusion) / total) if total else 0.0,
        "macro_f1": float(np.mean(f1_values)),
        "weighted_f1": float(np.average(f1_values, weights=supports)) if sum(supports) else 0.0,
        "min_group_f1": float(np.min(f1_values)),
    }
    return metrics, rows, confusion


def separability_metrics(
    centroids: np.ndarray,
    assignments: np.ndarray,
    prototypes: np.ndarray,
    weights: np.ndarray,
) -> dict[str, float]:
    distances = np.clip(1.0 - centroids @ prototypes.T, 0.0, 2.0)
    own = distances[np.arange(len(centroids)), assignments]
    other_distances = distances.copy()
    other_distances[np.arange(len(centroids)), assignments] = np.inf
    alternative = other_distances.min(axis=1)
    silhouette = np.divide(
        alternative - own,
        np.maximum(alternative, own),
        out=np.zeros_like(own),
        where=np.maximum(alternative, own) > 1.0e-12,
    )
    pairwise = np.clip(1.0 - prototypes @ prototypes.T, 0.0, 2.0)
    upper = pairwise[np.triu_indices(len(prototypes), k=1)]
    weighted_silhouette = float(np.average(silhouette, weights=weights))
    mean_distance = float(upper.mean())
    min_distance = float(upper.min())
    # Recovered exactly from the historical k=4--12 partition summaries.
    objective = weighted_silhouette + 0.25 * mean_distance + 0.25 * min_distance
    return {
        "within_centroid_loss": float(np.average(own, weights=weights)),
        "weighted_silhouette": weighted_silhouette,
        "unweighted_silhouette": float(silhouette.mean()),
        "mean_group_centroid_distance": mean_distance,
        "min_group_centroid_distance": min_distance,
        "max_group_centroid_similarity": float(1.0 - min_distance),
        "objective": objective,
    }


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    if args.min_groups < 2 or args.max_groups < args.min_groups:
        raise SystemExit("Require 2 <= --min-groups <= --max-groups.")
    validation_path = args.validation_embeddings or args.run_dir / "val_embeddings.npz"
    test_path = args.test_embeddings or args.run_dir / "test_embeddings.npz"
    output_dir = args.output_dir or args.run_dir / DEFAULT_OUTPUT_NAME
    if output_dir.exists() and any(output_dir.iterdir()) and not args.force:
        raise SystemExit(f"Output directory is not empty (use --force): {output_dir}")
    validation, test = load_embeddings(validation_path), load_embeddings(test_path)
    validate_splits(validation, test)
    kept_indices, centroids, validation_counts = taxon_centroids(validation, args.min_class_validation_samples)
    test_counts = np.bincount(test["labels"], minlength=len(validation["label_columns"])).astype(np.int64)
    weights = validation_counts[kept_indices].astype(np.float64) ** float(args.weight_power)
    labels = validation["label_columns"]
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_rows: list[dict[str, Any]] = []

    for groups in range(args.min_groups, args.max_groups + 1):
        kept_assignments, prototypes = fit_weighted_kmeans(centroids, weights, groups, args)
        full_assignments = np.full(len(labels), -1, dtype=np.int64)
        full_assignments[kept_indices] = kept_assignments
        val_metrics, _val_rows, _val_confusion = group_metrics(validation, full_assignments, prototypes)
        test_metrics, test_rows, test_confusion = group_metrics(test, full_assignments, prototypes)
        separation = separability_metrics(centroids, kept_assignments, prototypes, weights)
        row = {
            "group_count": groups,
            "search_method": "kmeans",
            **{f"validation_{key}": value for key, value in val_metrics.items()},
            "search_proxy_accuracy": val_metrics["accuracy"],
            "search_proxy_macro_f1": val_metrics["macro_f1"],
            "search_proxy_weighted_f1": val_metrics["weighted_f1"],
            "search_proxy_min_group_f1": val_metrics["min_group_f1"],
            **separation,
            **{f"test_{key}": value for key, value in test_metrics.items()},
        }
        summary_rows.append(row)
        np.savez_compressed(
            output_dir / f"k{groups:02d}_partition.npz",
            assignments=full_assignments,
            group_centroids=prototypes,
            label_columns=labels,
            val_counts=validation_counts,
            test_counts=test_counts,
            prism_normals_enabled=np.asarray([False]),
            prism_normals_weight=np.asarray([0.0]),
        )
        composition: list[dict[str, Any]] = []
        for group in range(groups):
            indices = np.flatnonzero(full_assignments == group)
            ordered = indices[np.argsort(-validation_counts[indices], kind="stable")]
            composition.append(
                {
                    "group_count": groups, "group": f"group{group + 1:02d}",
                    "label_count": len(indices),
                    "validation_samples": int(validation_counts[indices].sum()),
                    "labels": ",".join(labels[ordered]),
                }
            )
        write_csv(output_dir / f"k{groups:02d}_group_composition.csv", composition)
        write_csv(output_dir / f"k{groups:02d}_test_group_metrics.csv", test_rows)
        confusion_rows = [
            {"true_group": f"group{i + 1:02d}", **{f"pred_group{j + 1:02d}": float(test_confusion[i, j] / max(test_confusion[i].sum(), 1) * 100.0) for j in range(groups)}}
            for i in range(groups)
        ]
        write_csv(output_dir / f"k{groups:02d}_test_confusion_percent.csv", confusion_rows)
        print(
            f"k={groups:02d}: oracle_macro_f1={test_metrics['macro_f1']:.6f}; "
            f"weighted_silhouette={separation['weighted_silhouette']:.6f}; objective={separation['objective']:.6f}",
            flush=True,
        )

    summary_rows.sort(key=lambda item: float(item["objective"]), reverse=True)
    write_csv(output_dir / "partition_summary.csv", summary_rows)
    best = summary_rows[0]
    summary = {
        "run_dir": str(args.run_dir),
        "validation_embeddings": str(validation_path),
        "test_embeddings": str(test_path),
        "label_count": int(len(kept_indices)),
        "objective": "separability",
        "best_group_count": int(best["group_count"]),
        "best_objective": float(best["objective"]),
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "note": "Oracle metrics assign each embedding to its nearest normalized group-centroid prototype.",
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Best separability: k={int(best['group_count'])}; objective={float(best['objective']):.10f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
