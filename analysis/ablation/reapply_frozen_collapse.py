#!/usr/bin/env python3
"""Recalculate collapsed outputs from saved k=11 ensemble probabilities."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inference-root", type=Path, required=True)
    parser.add_argument(
        "--collapse",
        type=Path,
        default=root / "config" / "final_k11_to_k6_collapse.npz",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def collapse_probabilities(probabilities: np.ndarray, mapping: np.ndarray) -> np.ndarray:
    collapsed = np.zeros((len(probabilities), int(mapping.max()) + 1), dtype=probabilities.dtype)
    for source, destination in enumerate(mapping):
        collapsed[:, destination] += probabilities[:, source]
    return collapsed


def update_run(run_dir: Path, mapping: np.ndarray, mapping_text: str, overwrite: bool) -> None:
    prediction_path = run_dir / "ensemble_predictions.npz"
    summary_path = run_dir / "ensemble_summary.json"
    if not prediction_path.exists() or not summary_path.exists():
        return
    if not overwrite:
        raise FileExistsError(f"Use --overwrite to update {run_dir}")

    with np.load(prediction_path, allow_pickle=True) as source:
        arrays = {name: source[name] for name in source.files}
    probabilities = np.asarray(arrays["probabilities"])
    labels = np.asarray(arrays["labels"], dtype=np.int64)
    if probabilities.shape[1] != len(mapping):
        raise ValueError(f"Class-count mismatch in {prediction_path}")

    collapsed_probabilities = collapse_probabilities(probabilities, mapping)
    collapsed_labels = mapping[labels]
    collapsed_predictions = collapsed_probabilities.argmax(axis=1).astype(np.int64)
    group_names = np.asarray(
        [f"collapsed_group{group + 1:02d}" for group in range(int(mapping.max()) + 1)],
        dtype="<U64",
    )
    arrays.update(
        collapsed_probabilities=collapsed_probabilities,
        collapsed_labels=collapsed_labels,
        collapsed_predictions=collapsed_predictions,
        collapsed_group_names=group_names,
    )
    temporary = prediction_path.with_suffix(".npz.tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    os.replace(temporary, prediction_path)

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary.setdefault("args", {})["collapse_map"] = mapping_text
    summary["collapsed"] = {
        "collapse_map": mapping_text,
        "group_names": group_names.tolist(),
        "accuracy": float(accuracy_score(collapsed_labels, collapsed_predictions)),
        "macro_f1": float(f1_score(collapsed_labels, collapsed_predictions, average="macro", zero_division=0)),
        "weighted_f1": float(
            f1_score(collapsed_labels, collapsed_predictions, average="weighted", zero_division=0)
        ),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    pd.DataFrame([summary]).to_csv(run_dir / "ensemble_summary.csv", index=False)

    precision, recall, f1, support = precision_recall_fscore_support(
        collapsed_labels,
        collapsed_predictions,
        labels=np.arange(len(group_names)),
        zero_division=0,
    )
    pd.DataFrame(
        {
            "group": group_names,
            "support": support.astype(int),
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
    ).to_csv(run_dir / "ensemble_collapsed_per_group_metrics.csv", index=False)
    print(f"updated {run_dir}")


def main() -> int:
    args = parse_args()
    with np.load(args.collapse, allow_pickle=True) as artifact:
        mapping = np.asarray(artifact["initial_group_assignment"], dtype=np.int64)
    mapping_text = ",".join(str(group + 1) for group in mapping)
    for run_dir in sorted(path for path in args.inference_root.iterdir() if path.is_dir()):
        update_run(run_dir, mapping, mapping_text, args.overwrite)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
