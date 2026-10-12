#!/usr/bin/env python3
"""Summarize frozen crown-center detections and inventory matches."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from tree_crown_common import normalize_city_token


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--crown-root", type=Path, required=True)
    parser.add_argument("--match-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def numeric_summary(frame: pd.DataFrame, column: str, prefix: str) -> dict[str, object]:
    if column not in frame.columns:
        values = pd.Series(dtype=float)
    else:
        values = pd.to_numeric(frame[column], errors="coerce").dropna()
    if values.empty:
        return {f"{prefix}_median": np.nan, f"{prefix}_min": np.nan, f"{prefix}_max": np.nan}
    return {
        f"{prefix}_median": float(values.median()),
        f"{prefix}_min": float(values.min()),
        f"{prefix}_max": float(values.max()),
    }


def summarize_city(crown_path: Path, match_root: Path) -> dict[str, object]:
    city = normalize_city_token(crown_path.name.removesuffix("_tree_centers.csv"))
    crowns = pd.read_csv(crown_path, low_memory=False)
    matches = sorted((match_root / city).glob(f"{city}_tree_to_nearest_detected_crown_*m.csv"))
    row: dict[str, object] = {
        "city_token": city,
        "detected_crowns": int(len(crowns)),
        "crown_file": crown_path.name,
        "match_file": matches[0].name if len(matches) == 1 else "",
    }
    row.update(numeric_summary(crowns, "confidence", "crown_confidence"))
    epsg = crowns["cell_epsg"] if "cell_epsg" in crowns.columns else pd.Series(dtype=float)
    row["crown_epsg_count"] = int(pd.to_numeric(epsg, errors="coerce").nunique())
    if len(matches) == 1:
        joined = pd.read_csv(matches[0], low_memory=False)
        row["matched_inventory_records"] = int(len(joined))
        row.update(numeric_summary(joined, "match_distance_m", "match_distance_m"))
        if "crown_id" in joined:
            counts = joined["crown_id"].value_counts(dropna=True)
            row["unique_matched_crowns"] = int(len(counts))
            row["crowns_matched_to_multiple_records"] = int((counts > 1).sum())
    else:
        row["matched_inventory_records"] = np.nan
        row["unique_matched_crowns"] = np.nan
        row["crowns_matched_to_multiple_records"] = np.nan
    return row


def main() -> int:
    args = parse_args()
    crown_files = sorted(args.crown_root.glob("*_tree_centers.csv"))
    if not crown_files:
        raise SystemExit(f"No crown-center CSVs found under {args.crown_root}")
    summary = pd.DataFrame(summarize_city(path, args.match_root) for path in crown_files)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    summary.sort_values("city_token").to_csv(args.output, index=False)
    print(f"Wrote {len(summary):,} city rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
