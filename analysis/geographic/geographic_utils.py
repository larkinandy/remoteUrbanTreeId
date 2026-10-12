"""Shared loading, screening, grouping, and metric helpers for geographic analyses."""

from __future__ import annotations

import json
import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score


TRUE_VALUES = frozenset({"1", "true", "t", "yes", "y"})
FALSE_VALUES = frozenset({"0", "false", "f", "no", "n", "", "none", "nan"})


def boolean_series(values: pd.Series) -> pd.Series:
    """Parse common boolean encodings without treating the string ``False`` as true."""
    if pd.api.types.is_bool_dtype(values.dtype):
        return values.fillna(False).eq(True)
    normalized = values.fillna("").astype(str).str.strip().str.lower()
    unknown = sorted(set(normalized) - TRUE_VALUES - FALSE_VALUES)
    if unknown:
        raise ValueError(f"Unrecognized boolean values: {unknown[:5]}")
    return normalized.isin(TRUE_VALUES)


def record_keys(city: np.ndarray, tree_id: np.ndarray, crop_index: np.ndarray) -> np.ndarray:
    """Build stable keys shared by prediction and QA artifacts."""
    return np.asarray([
        f"{city_value}|{int(tree)}|{int(crop)}"
        for city_value, tree, crop in zip(city, tree_id, crop_index)
    ])


def load_predictions(path: Path, suffix: str) -> pd.DataFrame:
    """Load k11 and collapsed_k6 labels and predictions from an ensemble artifact."""
    with np.load(path, allow_pickle=True) as artifact:
        city = np.asarray(artifact["city_token"], dtype=str)
        keys = record_keys(city, artifact["tree_id"], artifact["crop_index"])
        return pd.DataFrame({
            "record_key": keys,
            "city": city,
            f"k11_label_{suffix}": np.asarray(artifact["labels"], dtype=np.int64),
            f"k11_prediction_{suffix}": np.asarray(artifact["predictions"], dtype=np.int64),
            f"collapsed_k6_label_{suffix}": np.asarray(artifact["collapsed_labels"], dtype=np.int64),
            f"collapsed_k6_prediction_{suffix}": np.asarray(
                artifact["collapsed_predictions"], dtype=np.int64
            ),
        })


def load_qa_flags(path: Path) -> pd.DataFrame:
    """Load and robustly parse the QA warning used for geographic reporting."""
    qa = pd.read_csv(
        path,
        usecols=["record_key", "qa_flag_low_vegetated_height_with_lidar"],
    )
    qa["qa_flag_low_vegetated_height_with_lidar"] = boolean_series(
        qa["qa_flag_low_vegetated_height_with_lidar"]
    )
    return qa


def screen_qa(frame: pd.DataFrame) -> pd.DataFrame:
    """Exclude records carrying the finalized low-height-with-LiDAR warning."""
    return frame.loc[~frame["qa_flag_low_vegetated_height_with_lidar"]].copy()


def eligible_city_support(
    prediction_path: Path,
    qa_path: Path,
    minimum_test_records: int,
) -> pd.DataFrame:
    """Return cities meeting the minimum after the finalized QA screen."""
    prediction = load_predictions(prediction_path, "reference")
    screened = screen_qa(prediction.merge(load_qa_flags(qa_path), on="record_key", validate="one_to_one"))
    counts = screened["city"].value_counts().rename_axis("city").reset_index(name="qa_screened_test_records")
    return counts.loc[counts.qa_screened_test_records.ge(minimum_test_records)].sort_values("city").reset_index(drop=True)


def metric_set(
    labels: np.ndarray,
    predictions: np.ndarray,
    class_count: int,
    minimum_group_records: int = 10,
) -> tuple[dict[str, Any], list[dict[str, Any]], np.ndarray]:
    """Calculate final geographic metrics, including support-restricted macro-F1."""
    matrix = np.bincount(
        labels * class_count + predictions, minlength=class_count**2
    ).reshape(class_count, class_count)
    support = matrix.sum(axis=1)
    predicted = matrix.sum(axis=0)
    true_positive = np.diag(matrix)
    precision = np.divide(
        true_positive, predicted, out=np.zeros(class_count, dtype=float), where=predicted > 0
    )
    recall = np.divide(
        true_positive, support, out=np.zeros(class_count, dtype=float), where=support > 0
    )
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros(class_count, dtype=float),
        where=(precision + recall) > 0,
    )
    represented = np.flatnonzero(support > 0)
    reliable = np.flatnonzero(support >= minimum_group_records)
    if not len(reliable):
        raise ValueError("No groups meet the minimum support for macro-F1")
    overall: dict[str, Any] = {
        "records": int(len(labels)),
        "represented_groups": int(len(represented)),
        "groups_ge_minimum": int(len(reliable)),
        "groups_in_macro_f1": ",".join(str(int(value) + 1) for value in reliable),
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1_groups_ge_minimum": float(
            f1_score(labels, predictions, labels=reliable, average="macro", zero_division=0)
        ),
        "macro_f1_represented": float(
            f1_score(labels, predictions, labels=represented, average="macro", zero_division=0)
        ),
        "weighted_f1": float(
            f1_score(labels, predictions, labels=represented, average="weighted", zero_division=0)
        ),
    }
    per_group = [
        {
            "group": index + 1,
            "support": int(support[index]),
            "included_in_macro_f1": bool(support[index] >= minimum_group_records),
            "predicted": int(predicted[index]),
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(f1[index]),
        }
        for index in range(class_count)
    ]
    return overall, per_group, matrix


def collapse_map_string(grouping_config: Path) -> str:
    """Read the canonical one-based k11-to-collapsed_k6 mapping."""
    configuration = json.loads(grouping_config.read_text(encoding="utf-8"))
    mapping = [int(value) for value in configuration["collapse_map_one_based"]]
    if len(mapping) != int(configuration["intermediate_group_count"]):
        raise ValueError("Canonical collapse map has an invalid length")
    return ",".join(str(value) for value in mapping)


def validate_split_manifest(path: Path, geographic_config: Path) -> None:
    """Verify that an external split manifest matches the frozen study artifact."""
    configuration = json.loads(geographic_config.read_text(encoding="utf-8"))
    expected = configuration["split_manifest"]
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != expected["sha256"]:
        raise ValueError(
            f"Split-manifest checksum mismatch for {path}: {digest} != {expected['sha256']}"
        )
