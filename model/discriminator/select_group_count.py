#!/usr/bin/env python3
"""Exploratory validation-only comparison of initial candidate group counts.

The discriminator embeds every tree on the unit hypersphere. This script
computes one normalized centroid per fine taxon, clusters those centroids with
sample-count-weighted k-means, and evaluates the resulting classes using only
validation embeddings. The selected partition is the one with the strongest
centroid-separability objective, not necessarily the partition with the fewest
classes or highest validation accuracy.

The emitted ``kNN_partition.npz`` files implement the exact contract consumed
by ``model/stage2/train_fixed_partition_classifier.py``.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import numpy as np

from discriminator_utils import prepare_output_directory, sha256, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--validation-embeddings", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--min-groups", type=int, default=5)
    parser.add_argument("--max-groups", type=int, default=12)
    parser.add_argument("--n-init", type=int, default=200)
    parser.add_argument("--max-iterations", type=int, default=200)
    parser.add_argument("--weight-power", type=float, default=0.5)
    parser.add_argument("--min-class-validation-samples", type=int, default=1)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument(
        "--kmeans-backend",
        choices=("numpy", "sklearn"),
        default="numpy",
        help="Use sklearn for exact manuscript-era reproduction or numpy for the platform-safe equivalent.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def normalize_rows(values: np.ndarray) -> np.ndarray:
    """Normalize embedding or prototype rows for cosine comparisons."""
    values = np.asarray(values, dtype=np.float32)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return np.divide(values, norms, out=np.zeros_like(values), where=norms > 1.0e-12)


def load_embeddings(path: Path) -> dict[str, np.ndarray]:
    """Load and validate a discriminator validation-embedding archive."""
    if not path.exists():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as data:
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
    if args.kmeans_backend == "sklearn":
        from sklearn.cluster import KMeans

        model = KMeans(
            n_clusters=groups,
            init="k-means++",
            n_init=int(args.n_init),
            max_iter=int(args.max_iterations),
            random_state=int(args.seed),
            algorithm="lloyd",
        )
        # This workstation's threadpoolctl DLL introspection raises
        # 0xc06d007f before fitting. The check only determines whether to emit
        # a Windows MKL/vcomp memory-leak warning; bypassing it does not alter
        # K-means initialization, Lloyd updates, or results.
        model._check_mkl_vcomp = lambda *unused_args, **unused_kwargs: None
        assignments = model.fit_predict(centroids, sample_weight=weights).astype(np.int64)
        prototypes = np.stack(
            [
                np.average(
                    centroids[assignments == group],
                    axis=0,
                    weights=weights[assignments == group],
                )
                for group in range(groups)
            ]
        )
        return assignments, normalize_rows(prototypes)
    # This dataset contains only a few hundred taxon centroids. A compact
    # NumPy implementation avoids platform-specific BLAS/threadpool failures
    # while retaining weighted k-means++ initialization and Lloyd updates.
    rng = np.random.default_rng(int(args.seed))
    values = np.asarray(centroids, dtype=np.float64)
    sample_weights = np.asarray(weights, dtype=np.float64)
    best_inertia = np.inf
    best_assignments = None
    best_prototypes = None
    for _restart in range(int(args.n_init)):
        chosen: list[int] = []
        first_prob = sample_weights / sample_weights.sum()
        chosen.append(int(rng.choice(len(values), p=first_prob)))
        min_sq = np.sum((values - values[chosen[0]]) ** 2, axis=1)
        while len(chosen) < int(groups):
            probability = sample_weights * min_sq
            probability[np.asarray(chosen, dtype=np.int64)] = 0.0
            if probability.sum() <= 0:
                candidates = np.setdiff1d(np.arange(len(values)), np.asarray(chosen), assume_unique=False)
                chosen.append(int(rng.choice(candidates)))
            else:
                chosen.append(int(rng.choice(len(values), p=probability / probability.sum())))
            new_sq = np.sum((values - values[chosen[-1]]) ** 2, axis=1)
            min_sq = np.minimum(min_sq, new_sq)
        prototypes = values[np.asarray(chosen, dtype=np.int64)].copy()
        assignments = np.full(len(values), -1, dtype=np.int64)
        for _iteration in range(int(args.max_iterations)):
            sq = np.sum((values[:, None, :] - prototypes[None, :, :]) ** 2, axis=2)
            updated_assignments = np.argmin(sq, axis=1).astype(np.int64)
            if np.array_equal(updated_assignments, assignments):
                break
            assignments = updated_assignments
            for group in range(int(groups)):
                mask = assignments == group
                if not mask.any():
                    nearest = sq[np.arange(len(values)), assignments]
                    replacement = int(np.argmax(sample_weights * nearest))
                    assignments[replacement] = group
                    mask = assignments == group
                prototypes[group] = np.average(values[mask], axis=0, weights=sample_weights[mask])
        sq = np.sum((values - prototypes[assignments]) ** 2, axis=1)
        inertia = float(np.sum(sample_weights * sq))
        if inertia < best_inertia:
            best_inertia = inertia
            best_assignments = assignments.copy()
            best_prototypes = prototypes.copy()
    if best_assignments is None or best_prototypes is None:
        raise RuntimeError("Weighted k-means failed to produce a partition")
    # Recompute prototypes explicitly and normalize exactly as consumed by the
    # downstream nearest-cosine-centroid classifier.
    prototypes = np.stack(
        [
            np.average(
                centroids[best_assignments == group],
                axis=0,
                weights=weights[best_assignments == group],
            )
            for group in range(groups)
        ]
    )
    return best_assignments.astype(np.int64), normalize_rows(prototypes)


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
    scores = np.einsum("ij,kj->ik", embeddings, prototypes, optimize=False)
    predicted = np.argmax(scores, axis=1).astype(np.int64)
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
    centroid_similarity = np.einsum("ij,kj->ik", centroids, prototypes, optimize=False)
    distances = np.clip(1.0 - centroid_similarity, 0.0, 2.0)
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
    prototype_similarity = np.einsum("ij,kj->ik", prototypes, prototypes, optimize=False)
    pairwise = np.clip(1.0 - prototype_similarity, 0.0, 2.0)
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
    validation_path = args.validation_embeddings
    output_dir = args.output_dir
    output_dir = prepare_output_directory(output_dir, args.overwrite)
    validation = load_embeddings(validation_path)
    kept_indices, centroids, validation_counts = taxon_centroids(validation, args.min_class_validation_samples)
    weights = validation_counts[kept_indices].astype(np.float64) ** float(args.weight_power)
    labels = validation["label_columns"]
    summary_rows: list[dict[str, Any]] = []

    for groups in range(args.min_groups, args.max_groups + 1):
        kept_assignments, prototypes = fit_weighted_kmeans(centroids, weights, groups, args)
        full_assignments = np.full(len(labels), -1, dtype=np.int64)
        full_assignments[kept_indices] = kept_assignments
        val_metrics, _val_rows, _val_confusion = group_metrics(validation, full_assignments, prototypes)
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
        }
        summary_rows.append(row)
        np.savez_compressed(
            output_dir / f"k{groups:02d}_partition.npz",
            assignments=full_assignments,
            group_centroids=prototypes,
            label_columns=labels,
            val_counts=validation_counts,
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
        print(
            f"k={groups:02d}: validation_macro_f1={val_metrics['macro_f1']:.6f}; "
            f"weighted_silhouette={separation['weighted_silhouette']:.6f}; objective={separation['objective']:.6f}",
            flush=True,
        )

    summary_rows.sort(key=lambda item: float(item["objective"]), reverse=True)
    write_csv(output_dir / "partition_summary.csv", summary_rows)
    best = summary_rows[0]
    summary = {
        "validation_embeddings": str(validation_path),
        "validation_embeddings_sha256": sha256(validation_path),
        "label_count": int(len(kept_indices)),
        "objective": "separability",
        "best_group_count": int(best["group_count"]),
        "best_objective": float(best["objective"]),
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "test_data_used_for_group_selection": False,
        "note": "All selection metrics use validation embeddings only. Independent test evaluation is a separate workflow.",
    }
    write_json(output_dir / "summary.json", summary)
    print(f"Best separability: k={int(best['group_count'])}; objective={float(best['objective']):.10f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
