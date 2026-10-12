"""Output helpers shared by QA and final shard assembly."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def write_npz(path: Path, arrays: dict[str, Any], compress: bool) -> None:
    """Write a compressed or uncompressed NPZ after creating its directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if compress:
        np.savez_compressed(path, **arrays)
    else:
        np.savez(path, **arrays)


def merge_city_rows(
    path: Path,
    new_rows: list[dict[str, object]],
    processed_cities: set[str],
    columns: list[str] | None = None,
) -> None:
    """Replace processed cities in a generated summary without losing others."""
    if path.exists():
        existing = pd.read_csv(path)
        if "city_token" in existing.columns:
            existing = existing[~existing["city_token"].astype(str).str.lower().isin(processed_cities)].copy()
    else:
        existing = pd.DataFrame(columns=columns or [])
    additions = pd.DataFrame(new_rows, columns=columns) if columns else pd.DataFrame(new_rows)
    merged = pd.concat([existing, additions], ignore_index=True, sort=False)
    if "city_token" in merged.columns and len(merged):
        merged = merged.sort_values("city_token").reset_index(drop=True)
    if columns:
        for column in columns:
            if column not in merged.columns:
                merged[column] = pd.Series(dtype="object")
        merged = merged[columns]
    merged.to_csv(path, index=False)
