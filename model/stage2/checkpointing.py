#!/usr/bin/env python3
"""Checkpoint selection, serialization, and portable manifest helpers."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import artifacts


def selection_score(metrics: dict[str, Any], args: argparse.Namespace) -> float:
    """Calculate the configured validation checkpoint-selection score."""
    if args.checkpoint_selection_score == "accuracy":
        return float(metrics["accuracy"])
    if args.checkpoint_selection_score == "weighted_f1":
        return float(metrics["weighted_f1"])
    if args.checkpoint_selection_score == "blended":
        return float(
            args.blended_score_macro_weight * metrics["macro_f1"]
            + args.blended_score_accuracy_weight * metrics["accuracy"]
            + args.blended_score_weighted_weight * metrics["weighted_f1"]
        )
    return float(metrics["macro_f1"])


def checkpoint_payload(
    model: Any,
    group_names: list[str],
    label_to_group: dict[str, int],
    standardizers: dict[str, np.ndarray | None],
    args: argparse.Namespace,
    epoch: int,
    val_metrics: dict[str, Any],
    score: float,
    torch: Any,
) -> dict[str, Any]:
    """Create the complete, provenance-bearing checkpoint payload."""
    tensor_values = {
        name: None if value is None else torch.as_tensor(value)
        for name, value in standardizers.items()
    }
    return {
        "model_state_dict": model.state_dict(),
        "group_names": group_names,
        "label_to_group": label_to_group,
        **tensor_values,
        "epoch": int(epoch),
        "selection_score": float(score),
        "selection_metric": args.checkpoint_selection_score,
        "val_metrics": {key: value for key, value in val_metrics.items() if key != "per_class"},
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "provenance": artifacts.provenance(
            getattr(args, "split_manifest", None),
            getattr(args, "input_schema_sha256", None),
        ),
    }


def save_top_k_checkpoint(
    payload: dict[str, Any],
    run_dir: Path,
    top_k: int,
    saved_top: list[tuple[float, Path]],
    torch: Any,
) -> list[tuple[float, Path]]:
    """Save and rank the top validation checkpoints using portable paths."""
    if top_k <= 0:
        return saved_top
    checkpoint_dir = run_dir / "top_checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    score = float(payload["selection_score"])
    epoch = int(payload["epoch"])
    path = checkpoint_dir / f"checkpoint_epoch_{epoch:03d}_score_{score:.6f}.pt"
    torch.save(payload, path)
    saved_top.append((score, path))
    saved_top = sorted(saved_top, key=lambda item: item[0], reverse=True)
    while len(saved_top) > top_k:
        _old_score, old_path = saved_top.pop(-1)
        if old_path.exists():
            old_path.unlink()
    rows = [
        {"rank": rank + 1, "score": value, "checkpoint": path.name}
        for rank, (value, path) in enumerate(saved_top)
    ]
    pd.DataFrame(rows).to_csv(checkpoint_dir / "top_checkpoints.csv", index=False)
    return saved_top


def resolve_checkpoint_paths(
    model_run_dir: Path,
    explicit: list[Path] | None,
    top_k: int,
) -> list[Path]:
    """Resolve explicit or ranked checkpoint paths after directory relocation."""
    if explicit:
        return [Path(path) for path in explicit]
    manifest_path = model_run_dir / "top_checkpoints" / "top_checkpoints.csv"
    if manifest_path.exists():
        frame = pd.read_csv(manifest_path)
        if "rank" in frame.columns:
            frame = frame.sort_values("rank")
        paths: list[Path] = []
        for value in frame["checkpoint"].head(max(int(top_k), 1)).tolist():
            recorded = Path(value)
            relocated = manifest_path.parent / recorded.name
            paths.append(recorded if recorded.exists() else relocated)
        if paths:
            return paths
    best = model_run_dir / "best_model.pt"
    if best.exists():
        print(f"No top-k checkpoint manifest found; using {best}", flush=True)
        return [best]
    raise FileNotFoundError(f"No checkpoints found under {model_run_dir}")
