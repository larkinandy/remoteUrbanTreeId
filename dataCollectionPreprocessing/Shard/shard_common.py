"""Shared path, label, and array helpers for shard preprocessing."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def normalize_token(value: object) -> str:
    """Return the canonical alphanumeric city token."""
    return "".join(ch for ch in str(value or "").strip().lower() if ch.isalnum())


def coordinate_key(
    latitude: np.ndarray | pd.Series,
    longitude: np.ndarray | pd.Series,
    decimals: int,
) -> np.ndarray:
    """Create stable rounded coordinate keys."""
    lat = pd.to_numeric(pd.Series(latitude), errors="coerce").round(int(decimals)).astype(str)
    lon = pd.to_numeric(pd.Series(longitude), errors="coerce").round(int(decimals)).astype(str)
    return (lat + "|" + lon).to_numpy(dtype=str)


def boolean_array(series: pd.Series) -> np.ndarray:
    """Interpret common serialized truth values as a Boolean array."""
    if series.dtype == bool:
        return series.fillna(False).to_numpy(dtype=bool)
    return series.fillna(False).astype(str).str.lower().isin({"1", "true", "yes", "y"}).to_numpy(dtype=bool)


def finite_rows(array: np.ndarray) -> np.ndarray:
    """Return one Boolean indicating finite values for every leading row."""
    values = np.asarray(array)
    if values.ndim <= 1:
        return np.isfinite(values)
    return np.isfinite(values.reshape(values.shape[0], -1)).all(axis=1)


def load_npz(path: Path) -> dict[str, np.ndarray]:
    """Load an NPZ into independent arrays so its file handle can close."""
    with np.load(path, allow_pickle=True) as data:
        return {key: np.asarray(data[key]) for key in data.files}


def sidecar_path(root: Path, city: str, suffix: str) -> Path:
    """Resolve either the city-directory or flat sidecar convention."""
    nested = root / city / f"{city}{suffix}"
    return nested if nested.exists() else root / f"{city}{suffix}"
