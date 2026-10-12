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


def load_geographic_config(path: Path) -> dict[str, Any]:
    """Load the frozen geographic experiment configuration."""
    return json.loads(path.read_text(encoding="utf-8"))


def scientific_training_arguments(configuration: dict[str, Any]) -> list[str]:
    """Translate frozen scientific model settings into trainer arguments."""
    training = configuration["training"]
    weights = training["blended_score_weights"]
    return [
        "--seed", str(training["seed"]),
        "--reassign-capped-eval-to-train", "--exclude-qa-from-model",
        "--no-exclude-qa-warning-from-model", "--use-qa-sample-weights",
        "--low-center-vegetation-weight", str(training["low_center_vegetation_weight"]),
        "--low-vegetated-height-with-lidar-weight", str(training["low_vegetated_height_with_lidar_weight"]),
        "--insufficient-lidar-coverage-weight", str(training["insufficient_lidar_coverage_weight"]),
        "--use-distance-sample-weights",
        "--distance-weight-sigma-m", str(training["distance_weight_sigma_m"]),
        "--min-distance-sample-weight", str(training["minimum_distance_sample_weight"]),
        "--early-stopping-min-delta", str(training["early_stopping_min_delta"]),
        "--checkpoint-selection-score", str(training["checkpoint_selection_score"]),
        "--blended-score-macro-weight", str(weights["macro_f1"]),
        "--blended-score-accuracy-weight", str(weights["accuracy"]),
        "--blended-score-weighted-weight", str(weights["weighted_f1"]),
        "--save-top-k-checkpoints", str(training["save_top_k_checkpoints"]),
        "--image-center-crop-pixels", str(training["image_center_crop_pixels"]),
        "--image-encoder-normalization", "group", "--naip-encoder-normalization", "group",
        "--chm-encoder-normalization", "group", "--interaction-encoder-normalization", "group",
        "--independent-chm-full-extent", "--no-interaction-naip-full-extent",
        "--interaction-naip-crop-pixels", str(training["interaction_naip_crop_pixels"]),
        "--interaction-fusion-mode", "learned_common_grid",
        "--interaction-naip-patch-pixels", str(training["interaction_naip_patch_pixels"]),
        "--interaction-chm-patch-pixels", str(training["interaction_chm_patch_pixels"]),
        "--image-augmentation", "--use-sentinel-phenology", "--use-raw-sentinel",
        "--use-satellite-embedding", "--no-use-prism-normals",
        "--use-naip-chm-interaction-branch", "--exclude-modality", str(training["excluded_modality"]),
        "--phenology-stat-samples", str(training["statistics_samples"]),
        "--raw-sentinel-stat-samples", str(training["statistics_samples"]),
        "--satellite-embedding-stat-samples", str(training["statistics_samples"]),
        "--hidden-dim", str(training["hidden_dimension"]), "--dropout", str(training["dropout"]),
        "--learning-rate", str(training["learning_rate"]), "--weight-decay", str(training["weight_decay"]),
        "--grad-clip", str(training["gradient_clip"]), "--class-weighting", str(training["class_weighting"]),
        "--class-weight-max", str(training["maximum_class_weight"]),
    ]


def update_status_manifest(path: Path, city: str, stage: str, status: str) -> None:
    """Update a compact resumability manifest after each geographic job stage."""
    rows: list[dict[str, str]] = []
    if path.exists():
        rows = pd.read_csv(path, dtype=str).to_dict("records")
    rows = [row for row in rows if not (row["city"] == city and row["stage"] == stage)]
    rows.append({"city": city, "stage": stage, "status": status})
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).sort_values(["city", "stage"]).to_csv(path, index=False)
