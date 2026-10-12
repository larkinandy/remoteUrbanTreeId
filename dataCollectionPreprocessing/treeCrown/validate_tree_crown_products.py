#!/usr/bin/env python3
"""Validate crown-center point detections and inventory matches."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from tree_crown_common import normalize_city_token


CROWN_REQUIRED = {"chip_index", "x", "y", "confidence", "approx_x", "approx_y", "cell_epsg"}
MATCH_REQUIRED = {
    "tree_id", "match_distance_m", "crown_id", "crown_x", "crown_y",
    "crown_epsg", "crown_confidence",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--crown-root", type=Path, required=True)
    parser.add_argument("--match-root", type=Path, required=True)
    parser.add_argument("--frozen-summary", type=Path)
    parser.add_argument("--city-token", action="append", default=[])
    parser.add_argument("--maximum-match-distance", type=float, default=5.0)
    parser.add_argument("--minimum-match-confidence", type=float, default=0.10)
    parser.add_argument("--summary", type=Path, help="Optional portable validation report.")
    return parser.parse_args()


def require_columns(frame: pd.DataFrame, required: set[str]) -> list[str]:
    missing = sorted(required.difference(frame.columns))
    return [f"missing columns: {', '.join(missing)}"] if missing else []


def numeric_values(frame: pd.DataFrame, columns: tuple[str, ...]) -> tuple[pd.DataFrame, list[str]]:
    values = frame.loc[:, list(columns)].apply(pd.to_numeric, errors="coerce")
    invalid = ~np.isfinite(values.to_numpy(dtype=np.float64))
    errors = [f"non-finite numeric values: {int(invalid.sum()):,}"] if invalid.any() else []
    return values, errors


def validate_crowns(path: Path) -> tuple[dict[str, object], list[str]]:
    crowns = pd.read_csv(path, low_memory=False)
    errors = require_columns(crowns, CROWN_REQUIRED)
    row: dict[str, object] = {"detected_crowns": int(len(crowns))}
    if errors:
        return row, errors
    numeric, numeric_errors = numeric_values(
        crowns, ("chip_index", "x", "y", "confidence", "approx_x", "approx_y", "cell_epsg")
    )
    errors.extend(numeric_errors)
    if (numeric["chip_index"] < 0).any() or (numeric[["x", "y"]] < 0).any().any():
        errors.append("negative chip or peak-pixel index")
    epsg = numeric["cell_epsg"]
    if (epsg <= 0).any() or not np.allclose(epsg, np.round(epsg), equal_nan=False):
        errors.append("invalid or non-integral crown EPSG code")
    row["crown_epsg_count"] = int(epsg.nunique())
    if "confidence_is_proxy" in crowns.columns:
        proxy = crowns["confidence_is_proxy"].fillna(False).astype(bool)
        row["proxy_confidence_rows"] = int(proxy.sum())
        if proxy.any():
            errors.append(f"proxy confidence flag present on {int(proxy.sum()):,} detection(s)")
    else:
        row["proxy_confidence_rows"] = 0
    return row, errors


def validate_matches(path: Path, maximum_distance: float, minimum_confidence: float) -> tuple[dict[str, object], list[str]]:
    matches = pd.read_csv(path, low_memory=False)
    errors = require_columns(matches, MATCH_REQUIRED)
    row: dict[str, object] = {"matched_inventory_records": int(len(matches))}
    if errors or matches.empty:
        return row, errors
    numeric, numeric_errors = numeric_values(
        matches, ("match_distance_m", "crown_id", "crown_x", "crown_y", "crown_epsg", "crown_confidence")
    )
    errors.extend(numeric_errors)
    distance = numeric["match_distance_m"]
    if (distance < 0).any() or (distance > maximum_distance + 1e-6).any():
        errors.append(f"match distance outside [0, {maximum_distance:g}] m")
    if (numeric["crown_confidence"] < minimum_confidence - 1e-7).any():
        errors.append(f"matched crown confidence below {minimum_confidence:g}")
    if matches["tree_id"].isna().any():
        errors.append("missing tree_id")
    duplicate_trees = int(matches["tree_id"].duplicated().sum())
    if duplicate_trees:
        errors.append(f"duplicate tree_id values: {duplicate_trees:,}")
    crown_counts = matches["crown_id"].value_counts(dropna=True)
    row.update({
        "unique_matched_crowns": int(len(crown_counts)),
        "crowns_matched_to_multiple_records": int((crown_counts > 1).sum()),
        "maximum_match_distance_m": float(distance.max()),
    })
    return row, errors


def discover_city_products(args: argparse.Namespace) -> list[tuple[str, Path, Path | None]]:
    selected = {normalize_city_token(token) for token in args.city_token if normalize_city_token(token)}
    products: list[tuple[str, Path, Path | None]] = []
    for crown_path in sorted(args.crown_root.glob("*_tree_centers.csv")):
        city = normalize_city_token(crown_path.name.removesuffix("_tree_centers.csv"))
        if selected and city not in selected:
            continue
        candidates = sorted((args.match_root / city).glob(f"{city}_tree_to_nearest_detected_crown_*m.csv"))
        products.append((city, crown_path, candidates[0] if len(candidates) == 1 else None))
    return products


def compare_frozen_summary(report: pd.DataFrame, path: Path) -> list[str]:
    frozen = pd.read_csv(path, low_memory=False).set_index("city_token")
    current = report.set_index("city_token")
    errors: list[str] = []
    for city in current.index.intersection(frozen.index):
        for column in ("detected_crowns", "matched_inventory_records"):
            if column in frozen and int(current.at[city, column]) != int(frozen.at[city, column]):
                errors.append(f"{city}: {column} differs from frozen summary")
    errors.extend(f"{city}: absent from frozen summary" for city in sorted(set(current.index).difference(frozen.index)))
    return errors


def main() -> int:
    args = parse_args()
    products = discover_city_products(args)
    if not products:
        raise SystemExit("No crown-center products selected.")
    rows: list[dict[str, object]] = []
    for city, crown_path, match_path in products:
        row: dict[str, object] = {
            "city_token": city,
            "crown_file": crown_path.name,
            "match_file": match_path.name if match_path else "",
        }
        crown_metrics, errors = validate_crowns(crown_path)
        row.update(crown_metrics)
        if match_path is None:
            errors.append("expected exactly one matching product")
        else:
            match_metrics, match_errors = validate_matches(
                match_path, args.maximum_match_distance, args.minimum_match_confidence
            )
            row.update(match_metrics)
            errors.extend(match_errors)
        row["status"] = "failed" if errors else "valid"
        row["issues"] = "; ".join(errors)
        rows.append(row)
        print(f"{city}: {row['status']}" + (f" - {row['issues']}" if errors else ""), flush=True)
    report = pd.DataFrame(rows).sort_values("city_token")
    summary_errors = compare_frozen_summary(report, args.frozen_summary) if args.frozen_summary else []
    for error in summary_errors:
        print(f"frozen-summary mismatch - {error}", flush=True)
    if args.summary:
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        report.to_csv(args.summary, index=False)
        print(f"Wrote validation report: {args.summary.name}", flush=True)
    failed = int(report["status"].ne("valid").sum()) + len(summary_errors)
    print(f"Validated {len(report):,} cities; failures={failed:,}", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
