"""Array preparation, peak extraction, and metadata joins for crown detection."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from skimage.feature import peak_local_max


INDEX_COLUMNS = [
    "chip_index", "row_index", "reduced_id", "cell_id", "cell_col", "cell_row",
    "cell_epsg", "cell_center_x", "cell_center_y", "tree_count", "covered",
    "year", "percent_vegetation", "percent_center_vegetation",
]


def normalize_chip_shape(array: np.ndarray, band_count: int) -> np.ndarray:
    """Return an NCHW or NHWC chip stack as NHWC."""
    if array.ndim != 4:
        raise ValueError(f"Expected a four-dimensional chip stack; got {array.shape}")
    if array.shape[-1] == band_count:
        return array
    if array.shape[1] == band_count:
        return np.transpose(array, (0, 2, 3, 1))
    raise ValueError(f"Cannot identify {band_count}-band channel dimension in {array.shape}")


def pad_batch(images: np.ndarray, minimum: int = 64, multiple: int = 32) -> tuple[np.ndarray, tuple[int, int]]:
    """Pad NHWC chips for SFANet/VGG pooling while retaining original size."""
    height, width = images.shape[1:3]
    out_height = max(minimum, int(np.ceil(height / multiple) * multiple))
    out_width = max(minimum, int(np.ceil(width / multiple) * multiple))
    return np.pad(images, ((0, 0), (0, out_height - height), (0, out_width - width), (0, 0))), (height, width)


def predict_batches(model: object, chips: np.ndarray, batch_size: int):
    """Yield cropped confidence maps without retaining the full stack in RAM."""
    for start in range(0, len(chips), batch_size):
        batch, original = pad_batch(np.asarray(chips[start : start + batch_size]))
        prediction = model.predict(batch, verbose=False)
        if isinstance(prediction, (list, tuple)):
            prediction = prediction[0]
        yield start, np.asarray(prediction[:, : original[0], : original[1], 0], dtype=np.float32)


def extract_peaks(confidence: np.ndarray, min_distance: int, threshold_abs: float | None, threshold_rel: float | None) -> pd.DataFrame:
    """Extract local maxima and retain their model confidence values."""
    rows: list[dict[str, object]] = []
    for chip_index, image in enumerate(confidence):
        peaks = peak_local_max(
            image, min_distance=min_distance,
            threshold_abs=threshold_abs, threshold_rel=threshold_rel,
        )
        rows.extend(
            {"chip_index": chip_index, "x": int(x), "y": int(y), "confidence": float(image[y, x])}
            for y, x in peaks
        )
    return pd.DataFrame(rows, columns=["chip_index", "x", "y", "confidence"])


def attach_index_metadata(peaks: pd.DataFrame, index_csv: Path, chip_shape: tuple[int, int], pixel_size: float) -> pd.DataFrame:
    """Attach cell metadata and convert peak pixels to projected coordinates."""
    index = pd.read_csv(index_csv, low_memory=False)
    if "chip_index" not in index.columns:
        raise ValueError(f"{index_csv} is missing chip_index")
    columns = [column for column in INDEX_COLUMNS if column in index.columns]
    output = peaks.merge(index[columns], on="chip_index", how="left")
    if {"cell_center_x", "cell_center_y"}.issubset(output.columns):
        height, width = chip_shape
        output["approx_x"] = output["cell_center_x"].astype(float) + (output["x"].astype(float) - (width - 1) / 2) * pixel_size
        output["approx_y"] = output["cell_center_y"].astype(float) - (output["y"].astype(float) - (height - 1) / 2) * pixel_size
    return output
