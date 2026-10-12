"""Shared configuration, validation, and metric helpers for ablation analyses."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, f1_score


ROOT = Path(__file__).resolve().parents[2]
ABLATION_ROOT = Path(__file__).resolve().parent


def experiment_config() -> dict:
    """Return the version-controlled configuration for final experiments."""
    return json.loads((ABLATION_ROOT / "final_experiment_config.json").read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_split_manifest(path: Path) -> None:
    metadata_path = Path(__file__).resolve().parent / "provenance" / "split_manifest.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    observed = sha256(path)
    if observed != metadata["sha256"]:
        raise ValueError(
            f"Split manifest checksum mismatch for {path}: {observed}; "
            f"expected {metadata['sha256']}"
        )


def canonical_collapse_map() -> str:
    config = json.loads((ROOT / "config" / "final_grouping.json").read_text(encoding="utf-8"))
    return ",".join(str(value) for value in config["collapse_map_one_based"])


def training_complete(run_dir: Path, top_k: int) -> bool:
    """Require the config, history, best model, and complete checkpoint ensemble."""
    required = (run_dir / "run_config.json", run_dir / "history.csv", run_dir / "best_model.pt")
    checkpoints = list((run_dir / "top_checkpoints").glob("*.pt"))
    return all(path.is_file() and path.stat().st_size > 0 for path in required) and len(checkpoints) >= top_k


def inference_complete(run_dir: Path) -> bool:
    """Require all final ensemble products, not merely a summary marker."""
    required = (
        "ensemble_summary.json",
        "ensemble_summary.csv",
        "ensemble_per_group_metrics.csv",
        "ensemble_predictions.npz",
    )
    return all((run_dir / name).is_file() and (run_dir / name).stat().st_size > 0 for name in required)


def metric_summary(labels: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    """Calculate consistently named classification metrics."""
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(labels, predictions, average="weighted", zero_division=0)),
    }


def record_keys(prediction_file: np.lib.npyio.NpzFile) -> np.ndarray:
    """Build the canonical record identifiers used in comparability audits."""
    return np.asarray(
        [
            f"{city}|{int(tree_id)}|{int(crop_index)}"
            for city, tree_id, crop_index in zip(
                prediction_file["city_token"], prediction_file["tree_id"], prediction_file["crop_index"]
            )
        ],
        dtype=str,
    )


def collapsed_k6_probabilities(probabilities: np.ndarray, collapse_zero_based: np.ndarray) -> np.ndarray:
    """Sum k=11 probabilities into the six frozen reporting groups."""
    collapsed_k6 = np.zeros((probabilities.shape[0], 6), dtype=probabilities.dtype)
    for source_group, target_group in enumerate(collapse_zero_based):
        collapsed_k6[:, target_group] += probabilities[:, source_group]
    return collapsed_k6
