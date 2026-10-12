"""Shared naming, date, and file-discovery utilities for Sentinel-2 workflows."""

from __future__ import annotations

from datetime import date, datetime, timedelta
from pathlib import Path
import re

import numpy as np
import pandas as pd


CITY_ALIASES = {
    "abq": "albuquerque", "ana": "anaheim", "arl": "arlington",
    "atl": "atlanta", "bal": "baltimore", "buf": "buffalo",
    "cc": "capecoral", "cos": "coloradosprings", "csp": "coloradosprings",
    "dca": "washingtondc", "dc": "washingtondc", "den": "denver",
    "ggv": "gardengrove", "hnb": "huntingtonbeach", "la": "losangeles",
    "lax": "losangeles", "nyc": "newyork", "sf": "sanfrancisco",
}


def normalize_token(value: object) -> str:
    text = str(value or "").lower().replace("alberquerque", "albuquerque")
    return re.sub(r"[^a-z0-9]+", "", text)


def canonical_city(value: object) -> str:
    token = normalize_token(value)
    return CITY_ALIASES.get(token, token)


def slug_token(value: object) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", str(value or "")).strip("_").lower() or "unknown"


def parse_iso_date(value: str, label: str = "date") -> date:
    try:
        return datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError as exc:
        raise ValueError(f"{label} must be YYYY-MM-DD: {value!r}") from exc


def date_windows(start: date, end: date, interval_days: int) -> list[tuple[date, date]]:
    """Partition an inclusive date range into half-open windows."""
    if end < start:
        raise ValueError("End date must not precede start date")
    if interval_days <= 0:
        raise ValueError("Interval days must be positive")
    current = start
    exclusive_end = end + timedelta(days=1)
    windows = []
    while current < exclusive_end:
        window_end = min(current + timedelta(days=interval_days), exclusive_end)
        windows.append((current, window_end))
        current = window_end
    return windows


def finite_int(value: object) -> int | None:
    if pd.isna(value):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(number):
        return None
    return int(number)


def city_paths(root: Path, pattern: str, city: str) -> list[Path]:
    if not root.exists():
        return []
    return [
        path for path in sorted(root.glob(pattern))
        if city in normalize_token(path.name) or city == normalize_token(path.parent.name)
    ]
