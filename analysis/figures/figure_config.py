"""Canonical grouping and display configuration for publication figures."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
FIGURE_ROOT = Path(__file__).resolve().parent


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def canonical_collapse_zero_based() -> np.ndarray:
    """Return the frozen analytical k11-to-collapsed_k6 mapping."""
    grouping = _read_json(ROOT / "config" / "final_grouping.json")
    mapping = np.asarray(grouping["collapse_map_one_based"], dtype=np.int64) - 1
    if len(mapping) != grouping["intermediate_group_count"]:
        raise ValueError("Final grouping configuration has an invalid collapse-map length")
    return mapping


def analytical_to_manuscript_zero_based() -> np.ndarray:
    """Return the display-only relabeling for analytical k11 identifiers."""
    display = _read_json(FIGURE_ROOT / "figure_grouping.json")
    mapping = np.asarray(display["analytical_to_manuscript_one_based"], dtype=np.int64) - 1
    if sorted(mapping.tolist()) != list(range(len(mapping))):
        raise ValueError("Manuscript group order must be a permutation")
    return mapping


def manuscript_collapse_zero_based() -> np.ndarray:
    """Return the canonical collapse expressed in manuscript display order."""
    analytical_to_display = analytical_to_manuscript_zero_based()
    result = np.empty_like(analytical_to_display)
    result[analytical_to_display] = canonical_collapse_zero_based()
    return result


def colors(name: str) -> tuple[str, ...]:
    display = _read_json(FIGURE_ROOT / "figure_grouping.json")
    key = {
        "collapsed_k6": "collapsed_k6_colors",
        "k11_analytical": "k11_analytical_colors",
        "k11_manuscript": "k11_manuscript_colors",
    }[name]
    return tuple(display[key])
