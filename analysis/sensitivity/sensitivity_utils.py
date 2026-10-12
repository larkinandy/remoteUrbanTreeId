"""Shared loading, validation, screening, and metric helpers for sensitivity analyses."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score


TRUE_VALUES = frozenset({"1", "true", "t", "yes", "y"})
FALSE_VALUES = frozenset({"0", "false", "f", "no", "n", "", "none", "nan"})


def boolean_series(values: pd.Series) -> pd.Series:
    """Parse common Boolean encodings without interpreting the string ``False`` as true."""
    if pd.api.types.is_bool_dtype(values.dtype):
        return values.fillna(False).eq(True)
    normalized = values.fillna("").astype(str).str.strip().str.lower()
    unknown = sorted(set(normalized) - TRUE_VALUES - FALSE_VALUES)
    if unknown:
        raise ValueError(f"Unrecognized Boolean values: {unknown[:5]}")
    return normalized.isin(TRUE_VALUES)


def record_keys(city: np.ndarray, tree_id: np.ndarray, crop_index: np.ndarray) -> np.ndarray:
    """Construct the study's stable record identifier."""
    return np.asarray([
        f"{city_value}|{int(tree)}|{int(crop)}"
        for city_value, tree, crop in zip(city, tree_id, crop_index)
    ], dtype=str)


def load_configuration(path: Path) -> dict[str, Any]:
    """Load a JSON analysis configuration."""
    return json.loads(path.read_text(encoding="utf-8"))


def load_predictions(path: Path, grouping_config: Path) -> pd.DataFrame:
    """Load predictions and verify their canonical k11-to-collapsed_k6 transformation."""
    grouping = load_configuration(grouping_config)
    mapping = np.asarray(grouping["collapse_map_one_based"], dtype=np.int64) - 1
    with np.load(path, allow_pickle=True) as artifact:
        probabilities = np.asarray(artifact["probabilities"], dtype=np.float64)
        labels = np.asarray(artifact["labels"], dtype=np.int64)
        predictions = np.asarray(artifact["predictions"], dtype=np.int64)
        collapsed_probabilities = np.asarray(artifact["collapsed_probabilities"], dtype=np.float64)
        collapsed_labels = np.asarray(artifact["collapsed_labels"], dtype=np.int64)
        collapsed_predictions = np.asarray(artifact["collapsed_predictions"], dtype=np.int64)
        city = np.asarray(artifact["city_token"], dtype=str)
        keys = record_keys(city, artifact["tree_id"], artifact["crop_index"])
    if probabilities.shape[1] != len(mapping):
        raise ValueError("Prediction k11 width does not match the canonical grouping configuration")
    expected_probabilities = np.column_stack([
        probabilities[:, mapping == group].sum(axis=1)
        for group in range(int(grouping["reported_group_count"]))
    ])
    if not np.allclose(expected_probabilities, collapsed_probabilities, atol=2e-6, rtol=1e-6):
        raise ValueError("Stored collapsed probabilities do not match the canonical probability sum")
    if not np.array_equal(mapping[labels], collapsed_labels):
        raise ValueError("Stored collapsed labels do not match the canonical mapping")
    if not np.array_equal(expected_probabilities.argmax(axis=1), collapsed_predictions):
        raise ValueError("Stored collapsed predictions do not match the canonical probability sum")
    frame = pd.DataFrame({
        "record_key": keys,
        "city_token": city,
        "label_k11": labels,
        "prediction_k11": predictions,
        "label_collapsed_k6": collapsed_labels,
        "prediction_collapsed_k6": collapsed_predictions,
    })
    if not frame.record_key.is_unique:
        raise ValueError("Prediction record keys are not unique")
    return frame


def screen_qa(frame: pd.DataFrame, column: str) -> pd.DataFrame:
    """Exclude records carrying the configured QA warning."""
    if frame[column].isna().any():
        raise ValueError(f"Missing QA values in {column}")
    warning = boolean_series(frame[column])
    return frame.loc[~warning].copy()


def metric_set(labels: np.ndarray, predictions: np.ndarray, class_count: int) -> dict[str, Any]:
    """Calculate accuracy and represented-class F1 metrics for one stratum."""
    labels = np.asarray(labels, dtype=np.int64)
    predictions = np.asarray(predictions, dtype=np.int64)
    if not len(labels):
        raise ValueError("Cannot calculate metrics for an empty stratum")
    support = np.bincount(labels, minlength=class_count)
    represented = np.flatnonzero(support > 0)
    return {
        "records": int(len(labels)),
        "represented_groups": int(len(represented)),
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(
            labels, predictions, labels=represented, average="macro", zero_division=0
        )),
        "weighted_f1": float(f1_score(
            labels, predictions, labels=represented, average="weighted", zero_division=0
        )),
    }


def file_sha256(path: Path) -> str:
    """Return the SHA-256 digest of a file without loading it into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def merge_one_to_one(frame: pd.DataFrame, addition: pd.DataFrame, name: str) -> pd.DataFrame:
    """Join record metadata and reject duplicate keys or missing prediction matches."""
    if addition.record_key.duplicated().any():
        raise ValueError(f"Duplicate record keys in {name}")
    merged = frame.merge(addition, on="record_key", how="left", validate="one_to_one")
    return merged
