#!/usr/bin/env python3
"""Extend a frozen core-taxon partition using development-only rare centroids."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def normalize_rows(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return np.divide(values, norms, out=np.zeros_like(values), where=norms > 1e-12)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    config_dir = Path(__file__).resolve().parents[2] / "config"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation-embeddings", type=Path, required=True)
    parser.add_argument("--rare-embeddings", type=Path, required=True)
    parser.add_argument("--core-partition", type=Path, default=config_dir / "core_k11_partition.npz")
    parser.add_argument("--core-prototypes", type=Path, default=config_dir / "core_k11_group_prototypes.npz")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    out = args.output_dir.resolve()
    if out.exists() and any(out.iterdir()) and not args.overwrite:
        raise SystemExit(f"Output directory is not empty: {out}")
    out.mkdir(parents=True, exist_ok=True)

    with np.load(args.validation_embeddings, allow_pickle=False) as data:
        embeddings = normalize_rows(data["embeddings"])
        labels = np.asarray(data["labels" if "labels" in data.files else "label_index"], dtype=int)
        core_taxa = data["label_columns"].astype(str)
    with np.load(args.core_partition, allow_pickle=False) as data:
        partition_taxa = data["label_columns"].astype(str)
        core_assignment = np.asarray(data["assignments"], dtype=int)
    if not np.array_equal(core_taxa, partition_taxa):
        raise RuntimeError("Core partition taxa do not match validation embedding taxa and order")
    group_count = int(core_assignment.max()) + 1
    core_counts = np.bincount(labels, minlength=len(core_taxa)).astype(int)
    core_centroids = normalize_rows(np.stack([
        embeddings[labels == index].mean(axis=0) for index in range(len(core_taxa))
    ]))
    with np.load(args.core_prototypes, allow_pickle=False) as data:
        prototypes = normalize_rows(np.asarray(data["group_centroids"], dtype=np.float32))
        if not np.array_equal(np.asarray(data["assignments"], dtype=int), core_assignment):
            raise RuntimeError("Core prototype assignments differ from the frozen core partition")

    with np.load(args.rare_embeddings, allow_pickle=False) as data:
        rare_embeddings = normalize_rows(data["embeddings"])
        rare_taxon_per_record = data["taxon"].astype(str)
        rare_split = data["rare_split"].astype(str)
    rare_taxa = np.sort(np.unique(rare_taxon_per_record))
    development_counts = np.asarray([
        np.sum((rare_taxon_per_record == taxon) & (rare_split == "development")) for taxon in rare_taxa
    ], dtype=int)
    test_counts = np.asarray([
        np.sum((rare_taxon_per_record == taxon) & (rare_split == "test")) for taxon in rare_taxa
    ], dtype=int)
    if np.any(development_counts == 0):
        raise RuntimeError("At least one rare taxon has no development records")
    rare_centroids = normalize_rows(np.stack([
        rare_embeddings[(rare_taxon_per_record == taxon) & (rare_split == "development")].mean(axis=0)
        for taxon in rare_taxa
    ]))
    similarity = rare_centroids @ prototypes.T
    rare_assignment = np.argmax(similarity, axis=1).astype(int)
    ordered = np.sort(similarity, axis=1)
    margin = ordered[:, -1] - ordered[:, -2]
    runner_up = np.argsort(similarity, axis=1)[:, -2]

    extended_taxa = np.concatenate([core_taxa, rare_taxa])
    extended_assignment = np.concatenate([core_assignment, rare_assignment])
    output_path = out / "final_k11_partition.npz"
    np.savez_compressed(
        output_path, assignments=extended_assignment, label_columns=extended_taxa,
        group_centroids=prototypes,
        core_validation_counts=core_counts, rare_development_counts=development_counts,
        rare_test_counts=test_counts, core_taxon_count=np.asarray([len(core_taxa)]),
        rare_extension_taxon_count=np.asarray([len(rare_taxa)]),
    )
    pd.DataFrame({
        "taxon": rare_taxa, "assigned_group": rare_assignment + 1,
        "nearest_similarity": similarity[np.arange(len(rare_taxa)), rare_assignment],
        "runner_up_group": runner_up + 1,
        "runner_up_similarity": similarity[np.arange(len(rare_taxa)), runner_up],
        "assignment_margin": margin, "development_records": development_counts,
        "held_out_test_records": test_counts,
    }).to_csv(out / "rare_taxon_group_assignments.csv", index=False)
    summary = pd.DataFrame({"assigned_group": np.arange(1, group_count + 1)})
    rare_frame = pd.DataFrame({"group": rare_assignment + 1, "development": development_counts,
                               "test": test_counts, "margin": margin})
    grouped = rare_frame.groupby("group").agg(rare_taxa=("group", "size"),
                                               development_records=("development", "sum"),
                                               held_out_test_records=("test", "sum"),
                                               median_assignment_margin=("margin", "median"),
                                               minimum_assignment_margin=("margin", "min")).reset_index()
    summary.merge(grouped, left_on="assigned_group", right_on="group", how="left").drop(columns="group").to_csv(
        out / "rare_assignment_group_summary.csv", index=False)
    audit = {
        "core_taxa": len(core_taxa), "rare_taxa": len(rare_taxa), "expanded_taxa": len(extended_taxa),
        "group_count": group_count, "rare_development_records_used_for_assignment": int(development_counts.sum()),
        "rare_held_out_test_records_not_used_for_assignment": int(test_counts.sum()),
        "test_records_used_in_prototype_construction": 0, "test_records_used_in_rare_centroids": 0,
        "prototype_method": "normalized sqrt-validation-support-weighted mean of normalized core-taxon validation centroids",
        "rare_assignment_method": "nearest cosine group prototype from development-only rare-taxon centroid",
        "assignment_margin_median": float(np.median(margin)), "assignment_margin_minimum": float(margin.min()),
        "core_partition_sha256": sha256(args.core_partition),
        "core_prototypes_sha256": sha256(args.core_prototypes),
        "validation_embeddings_sha256": sha256(args.validation_embeddings),
        "rare_embeddings_sha256": sha256(args.rare_embeddings),
        "output_partition": str(output_path),
    }
    (out / "assignment_audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2), flush=True)


if __name__ == "__main__":
    main()
