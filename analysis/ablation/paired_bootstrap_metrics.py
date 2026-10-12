"""Paired record bootstrap for a full model and an ablation prediction file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--full", type=Path, required=True)
    p.add_argument("--ablation", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--replicates", type=int, default=2000)
    p.add_argument("--seed", type=int, default=20261009)
    p.add_argument(
        "--collapse-map",
        default=None,
        help="Optional comma-separated one-based collapsed group for each source class.",
    )
    return p.parse_args()


def confusion_metrics(y: np.ndarray, pred: np.ndarray, weights: np.ndarray, k: int) -> np.ndarray:
    cm = np.bincount(y * k + pred, weights=weights, minlength=k * k).reshape(k, k)
    tp = np.diag(cm)
    support = cm.sum(axis=1)
    predicted = cm.sum(axis=0)
    denom = 2 * tp + (predicted - tp) + (support - tp)
    f1 = np.divide(2 * tp, denom, out=np.zeros_like(tp), where=denom > 0)
    accuracy = tp.sum() / cm.sum()
    macro = f1[support > 0].mean()
    weighted = np.sum(f1 * support) / support.sum()
    score = .3 * accuracy + .5 * macro + .2 * weighted
    return np.asarray([accuracy, macro, weighted, score])


def main() -> None:
    a = args()
    a.output_dir.mkdir(parents=True, exist_ok=True)
    full = np.load(a.full, allow_pickle=True)
    abl = np.load(a.ablation, allow_pickle=True)
    key_names = ("city_token", "tree_id", "crop_index")
    for name in key_names + ("labels",):
        if not np.array_equal(full[name], abl[name]):
            raise RuntimeError(f"Prediction files are not aligned on {name}")
    y = full["labels"].astype(np.int64)
    if a.collapse_map:
        collapse = np.asarray(
            [int(value.strip()) - 1 for value in a.collapse_map.split(",")], dtype=np.int64
        )
        if len(collapse) != full["probabilities"].shape[1]:
            raise ValueError("Collapse map length does not match the source probability dimension")
        collapsed_k = int(collapse.max()) + 1
        if collapse.min() < 0 or set(collapse.tolist()) != set(range(collapsed_k)):
            raise ValueError("Collapsed groups must be contiguous one-based identifiers")
        y = collapse[y]
        full_probability = np.zeros((len(y), collapsed_k), dtype=np.float64)
        ablation_probability = np.zeros((len(y), collapsed_k), dtype=np.float64)
        for source_group, collapsed_group in enumerate(collapse):
            full_probability[:, collapsed_group] += full["probabilities"][:, source_group]
            ablation_probability[:, collapsed_group] += abl["probabilities"][:, source_group]
        pf = full_probability.argmax(axis=1)
        pa = ablation_probability.argmax(axis=1)
    else:
        pf = full["predictions"].astype(np.int64)
        pa = abl["predictions"].astype(np.int64)
    k = int(max(y.max(), pf.max(), pa.max()) + 1)
    n = len(y)
    names = ["accuracy", "macro_f1", "weighted_f1", "score"]
    unit = np.ones(n, dtype=np.float64)
    observed_full = confusion_metrics(y, pf, unit, k)
    observed_ablation = confusion_metrics(y, pa, unit, k)
    rng = np.random.default_rng(a.seed)
    differences = np.empty((a.replicates, len(names)), dtype=np.float64)
    for b in range(a.replicates):
        # Poisson(1) paired bootstrap is the standard large-n approximation to
        # multinomial resampling and avoids constructing 100,000 indices/run.
        weights = rng.poisson(1.0, n).astype(np.float64)
        differences[b] = confusion_metrics(y, pf, weights, k) - confusion_metrics(y, pa, weights, k)
        if (b + 1) % 250 == 0:
            print(f"bootstrap {b + 1:,}/{a.replicates:,}", flush=True)
    rows = []
    for j, name in enumerate(names):
        low, high = np.quantile(differences[:, j], [.025, .975])
        rows.append({
            "metric": name,
            "full": observed_full[j],
            "ablation": observed_ablation[j],
            "difference_full_minus_ablation": observed_full[j] - observed_ablation[j],
            "bootstrap_mean_difference": differences[:, j].mean(),
            "ci_2.5_percent": low,
            "ci_97.5_percent": high,
            "probability_difference_le_zero": np.mean(differences[:, j] <= 0),
            "significant_95_percent": bool(low > 0 or high < 0),
        })
    summary = pd.DataFrame(rows)
    summary.to_csv(a.output_dir / "paired_bootstrap_summary.csv", index=False)
    np.savez_compressed(a.output_dir / "paired_bootstrap_replicates.npz", differences=differences, metric_names=np.asarray(names))
    with open(a.output_dir / "method.json", "w", encoding="utf-8") as f:
        json.dump({"replicates": a.replicates, "seed": a.seed, "resampling": "paired Poisson(1) record bootstrap", "full": str(a.full), "ablation": str(a.ablation), "collapse_map": a.collapse_map}, f, indent=2)
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
