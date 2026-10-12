#!/usr/bin/env python3
"""Fixed-k bootstrap consensus after k is selected from full validation centroids."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import cut_tree, linkage
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import squareform
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score


def norm(x):
    x = np.asarray(x, dtype=np.float32)
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return np.divide(x, n, out=np.zeros_like(x), where=n > 1e-12)


def fit(centroids, weights, groups, seed):
    return KMeans(n_clusters=groups, init="k-means++", n_init=200, max_iter=200,
                  random_state=seed, algorithm="lloyd").fit_predict(
                      centroids, sample_weight=weights).astype(int)


def align(reference, observed, groups):
    table = np.zeros((groups, groups), dtype=int)
    np.add.at(table, (reference, observed), 1)
    rows, cols = linear_sum_assignment(-table)
    mapping = {int(col): int(row) for row, col in zip(rows, cols)}
    return np.asarray([mapping[int(value)] for value in observed]), table


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--groups", type=int, required=True)
    parser.add_argument("--replicates", type=int, default=500)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--progress-every", type=int, default=50)
    args = parser.parse_args()
    if args.groups < 2:
        raise SystemExit("--groups must be at least 2")
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise SystemExit(f"Refusing to overwrite nonempty directory: {out}")

    started = time.perf_counter()
    with np.load(args.validation, allow_pickle=True) as data:
        embeddings = norm(data["embeddings"])
        labels = data["labels"].astype(int)
        taxa = data["label_columns"].astype(str)
    indices = [np.flatnonzero(labels == index) for index in range(len(taxa))]
    counts = np.asarray([len(index) for index in indices])
    if (counts == 0).any():
        raise RuntimeError("Every taxon must have validation support")
    weights = np.sqrt(counts)
    full_centroids = norm(np.stack([embeddings[index].mean(0) for index in indices]))
    reference = fit(full_centroids, weights, args.groups, args.seed)

    rng = np.random.default_rng(args.seed)
    assignments, aligned_assignments, stability_rows = [], [], []
    for replicate in range(1, args.replicates + 1):
        boot_centroids = norm(np.stack([
            embeddings[index[rng.integers(0, len(index), len(index))]].mean(0)
            for index in indices
        ]))
        observed = fit(boot_centroids, weights, args.groups, args.seed)
        aligned, _ = align(reference, observed, args.groups)
        assignments.append(observed)
        aligned_assignments.append(aligned)
        stability_rows.append({
            "replicate": replicate,
            "ARI_vs_full_validation": adjusted_rand_score(reference, observed),
            "aligned_agreement_vs_full_validation": float(np.mean(reference == aligned)),
        })
        if replicate % args.progress_every == 0 or replicate == args.replicates:
            elapsed = time.perf_counter() - started
            eta = elapsed / replicate * (args.replicates - replicate)
            print(f"PROGRESS {replicate}/{args.replicates} "
                  f"ARI={stability_rows[-1]['ARI_vs_full_validation']:.3f} "
                  f"agreement={stability_rows[-1]['aligned_agreement_vs_full_validation']:.3f} "
                  f"elapsed={elapsed:.1f}s eta={eta:.1f}s", flush=True)
    assignments = np.asarray(assignments)
    aligned_assignments = np.asarray(aligned_assignments)

    coassignment = np.eye(len(taxa), dtype=np.float32)
    for index in range(len(taxa)):
        coassignment[index] = np.mean(assignments[:, index, None] == assignments, axis=0)
    tree = linkage(squareform(np.clip(1 - coassignment, 0, 1), checks=False),
                   method="average", optimal_ordering=True)
    consensus = cut_tree(tree, n_clusters=[args.groups]).ravel().astype(int)
    consensus_aligned, overlap = align(reference, consensus, args.groups)
    retention = np.mean(aligned_assignments == reference[None, :], axis=0)

    stem = f"fixed_k{args.groups}"
    np.save(out / f"{stem}_coassignment_matrix.npy", coassignment)
    np.savez_compressed(out / f"{stem}_bootstrap_assignments.npz",
                        assignments=assignments, aligned_assignments=aligned_assignments,
                        label_columns=taxa, reference_assignments=reference)
    np.savez_compressed(out / f"{stem}_consensus_partition.npz",
                        assignments=consensus, label_columns=taxa,
                        validation_counts=counts, full_validation_centroids=full_centroids)
    pd.DataFrame(stability_rows).to_csv(out / "replicate_stability.csv", index=False)
    pd.DataFrame(overlap,
                 index=[f"reference_G{i}" for i in range(1, args.groups + 1)],
                 columns=[f"consensus_G{i}" for i in range(1, args.groups + 1)]
                 ).to_csv(out / "reference_consensus_overlap.csv")
    pd.DataFrame({"taxon": taxa, "reference_group": reference + 1,
                  "consensus_group": consensus_aligned + 1,
                  "reference_group_retention": retention,
                  "validation_support": counts}).to_csv(out / "taxon_stability.csv", index=False)
    frame = pd.DataFrame(stability_rows)
    summary = {
        "bootstrap_replicates": args.replicates,
        "fixed_k": args.groups,
        "reference_group_sizes": np.bincount(reference, minlength=args.groups).tolist(),
        "consensus_group_sizes": np.bincount(consensus, minlength=args.groups).tolist(),
        "consensus_ARI_vs_full_validation": float(adjusted_rand_score(reference, consensus)),
        "consensus_aligned_agreement_vs_full_validation": float(np.mean(reference == consensus_aligned)),
        "replicate_ARI_mean": float(frame.ARI_vs_full_validation.mean()),
        "replicate_agreement_mean": float(frame.aligned_agreement_vs_full_validation.mean()),
        "taxa_retention_below_0_8": int(np.sum(retention < .8)),
        "elapsed_seconds": time.perf_counter() - started,
        "test_data_loaded": False,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out / "method.json").write_text(json.dumps({
        "k_selection": "fixed before bootstrap from the complete-validation taxon-centroid objective",
        "bootstrap": "within-taxon n-out-of-n resampling of validation records",
        "clustering": f"weighted KMeans fixed k={args.groups}; weights=sqrt(validation support); n_init=200; max_iter=200; seed={args.seed}",
        "coassignment": f"fraction of fixed-k={args.groups} bootstrap replicates co-clustered",
        "consensus": f"{args.groups}-cluster average-linkage cut of 1 - coassignment",
        "test_data_loaded": False,
    }, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
