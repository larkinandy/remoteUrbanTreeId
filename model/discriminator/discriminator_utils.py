#!/usr/bin/env python3
"""Shared validation and provenance helpers for discriminator workflows."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np


SOURCE_DIRECTORY = Path(__file__).resolve().parent


def normalize_rows(values: np.ndarray, dtype: np.dtype = np.float32) -> np.ndarray:
    """Return row-wise unit vectors while preserving zero rows."""
    array = np.asarray(values, dtype=dtype)
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    return np.divide(array, norms, out=np.zeros_like(array), where=norms > 1e-12)


def sha256(path: Path) -> str:
    """Calculate a lowercase SHA-256 digest without loading the file at once."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_arrays(available: Iterable[str], required: set[str], source: Path) -> None:
    """Raise a clear error when an NPZ input does not meet its schema."""
    missing = sorted(required.difference(available))
    if missing:
        raise ValueError(f"{source} is missing required arrays: {missing}")


def require_generated_output_outside_source(path: Path) -> Path:
    """Reject generated outputs placed inside the tracked source directory."""
    resolved = path.resolve()
    try:
        resolved.relative_to(SOURCE_DIRECTORY)
    except ValueError:
        return resolved
    raise ValueError(f"Generated outputs must be stored outside {SOURCE_DIRECTORY}; received {resolved}")


def prepare_output_directory(path: Path, overwrite: bool) -> Path:
    """Validate and create a generated-output directory."""
    resolved = require_generated_output_outside_source(path)
    if resolved.exists() and any(resolved.iterdir()) and not overwrite:
        raise FileExistsError(f"Output directory is not empty; pass --overwrite: {resolved}")
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write deterministic, human-readable JSON with a terminal newline."""
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
