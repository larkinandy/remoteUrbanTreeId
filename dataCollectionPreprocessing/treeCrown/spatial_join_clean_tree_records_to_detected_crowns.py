#!/usr/bin/env python3
"""Join clean tree inventory records to nearest detected crown within a radius.

Inputs are the rebuilt standalone products:

* per-city clean inventory metadata keyed by tree_id
* per-city detected crown CSVs with projected approx_x/approx_y coordinates

The output is intentionally pre-Sentinel: it links tree records to detected
crowns only. Sentinel/GEE/PRISM linkage should be assigned from the matched
crown coordinate in a later step.
"""

from __future__ import annotations

import argparse
import csv
import multiprocessing as mp
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from pyproj import Transformer
from scipy.spatial import cKDTree


DEFAULT_TREE_METADATA_ROOT = Path(r"H:\TreeCenteredModelInputs\tree_record_metadata_clean")
DEFAULT_CROWN_ROOT = Path(r"H:\TreeCenteredModelInputs\detected_tree_crowns_clean")
DEFAULT_OUTPUT_ROOT = Path(r"H:\TreeCenteredModelInputs\tree_to_detected_crowns_clean")

CROWN_OUTPUT_COLUMNS = [
    "match_distance_m",
    "crown_id",
    "crown_source_row",
    "crown_lat",
    "crown_lon",
    "crown_x",
    "crown_y",
    "crown_epsg",
    "crown_confidence",
]


@dataclass(frozen=True)
class CityJob:
    city_token: str
    tree_metadata_csv: Path
    crown_csv: Path
    output_csv: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--tree-metadata-root", type=Path, default=DEFAULT_TREE_METADATA_ROOT)
    parser.add_argument("--crown-root", type=Path, default=DEFAULT_CROWN_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--summary-csv", type=Path, default=None)
    parser.add_argument("--city-token", action="append", default=None, help="Optional city token(s). Repeatable.")
    parser.add_argument("--exclude-city-token", action="append", default=[], help="City token(s) to skip. Repeatable.")
    parser.add_argument("--match-radius-m", type=float, default=5.0)
    parser.add_argument("--min-crown-confidence", type=float, default=0.10)
    parser.add_argument("--parallel-workers", type=int, default=1)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def require_columns(frame: pd.DataFrame, columns: list[str], path: Path) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise RuntimeError(f"{path} is missing required column(s): {missing}")


def read_kept_cities(summary_path: Path) -> list[str]:
    if not summary_path.exists():
        raise FileNotFoundError(summary_path)
    summary = pd.read_csv(summary_path, low_memory=False)
    require_columns(summary, ["city_token", "fails_unique_coordinate_threshold"], summary_path)
    keep = summary.loc[
        ~summary["fails_unique_coordinate_threshold"].fillna(False).astype(bool),
        "city_token",
    ]
    return sorted(str(value).strip().lower() for value in keep if str(value).strip())


def discover_jobs(args: argparse.Namespace) -> list[CityJob]:
    summary_path = args.tree_metadata_root / "tree_record_metadata_clean_summary.csv"
    city_tokens = read_kept_cities(summary_path)
    selected = {token.strip().lower() for token in args.city_token or [] if token.strip()}
    excluded = {token.strip().lower() for token in args.exclude_city_token if token.strip()}
    if selected:
        city_tokens = [city for city in city_tokens if city in selected]
    city_tokens = [city for city in city_tokens if city not in excluded]

    jobs: list[CityJob] = []
    for city in city_tokens:
        tree_csv = args.tree_metadata_root / city / f"{city}_tree_record_metadata_clean.csv"
        crown_csv = args.crown_root / f"{city}_tree_centers.csv"
        output_csv = args.output_root / city / f"{city}_tree_to_nearest_detected_crown_{args.match_radius_m:g}m.csv"
        jobs.append(CityJob(city, tree_csv, crown_csv, output_csv))
    return jobs


def load_crowns(path: Path, min_confidence: float) -> pd.DataFrame:
    crowns = pd.read_csv(path, low_memory=False)
    require_columns(crowns, ["approx_x", "approx_y", "confidence", "cell_epsg"], path)
    crowns = crowns.copy()
    crowns["crown_x"] = pd.to_numeric(crowns["approx_x"], errors="coerce")
    crowns["crown_y"] = pd.to_numeric(crowns["approx_y"], errors="coerce")
    crowns["crown_confidence"] = pd.to_numeric(crowns["confidence"], errors="coerce")
    crowns["crown_epsg"] = pd.to_numeric(crowns["cell_epsg"], errors="coerce").astype("Int64")
    valid = (
        crowns["crown_x"].notna()
        & crowns["crown_y"].notna()
        & crowns["crown_epsg"].notna()
        & crowns["crown_confidence"].ge(float(min_confidence))
    )
    crowns = crowns.loc[valid].copy()
    crowns.insert(0, "crown_id", np.arange(1, len(crowns) + 1, dtype=np.int64))
    crowns.insert(1, "crown_source_row", crowns.index.to_numpy(dtype=np.int64) + 2)
    return crowns.reset_index(drop=True)


def transform_tree_coordinates(trees: pd.DataFrame, epsg: int) -> tuple[np.ndarray, np.ndarray]:
    transformer = Transformer.from_crs("EPSG:4326", f"EPSG:{int(epsg)}", always_xy=True)
    x, y = transformer.transform(
        trees["tree_lon"].to_numpy(dtype=float),
        trees["tree_lat"].to_numpy(dtype=float),
    )
    return np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)


def inverse_crown_coordinates(crowns: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    out_lon = np.full(len(crowns), np.nan, dtype=np.float64)
    out_lat = np.full(len(crowns), np.nan, dtype=np.float64)
    for epsg, idx in crowns.groupby("crown_epsg").groups.items():
        epsg_int = int(epsg)
        transformer = Transformer.from_crs(f"EPSG:{epsg_int}", "EPSG:4326", always_xy=True)
        index = np.asarray(list(idx), dtype=np.int64)
        lon, lat = transformer.transform(
            crowns.iloc[index]["crown_x"].to_numpy(dtype=float),
            crowns.iloc[index]["crown_y"].to_numpy(dtype=float),
        )
        out_lon[index] = lon
        out_lat[index] = lat
    return out_lon, out_lat


def empty_match_frame(trees: pd.DataFrame) -> pd.DataFrame:
    """Return a header-only join table that downstream crop scripts can read."""
    columns = list(trees.columns)
    for column in CROWN_OUTPUT_COLUMNS:
        if column not in columns:
            columns.append(column)
    return pd.DataFrame(columns=columns)


def match_city(job: CityJob, args: argparse.Namespace) -> dict[str, object]:
    if job.output_csv.exists() and not args.force:
        existing_rows = sum(1 for _ in open(job.output_csv, "rb")) - 1
        return {
            "city_token": job.city_token,
            "status": "exists",
            "tree_rows": np.nan,
            "crown_rows": np.nan,
            "matched_rows": max(existing_rows, 0),
            "match_rate": np.nan,
            "output_csv": str(job.output_csv),
        }
    if not job.tree_metadata_csv.exists():
        raise FileNotFoundError(job.tree_metadata_csv)
    if not job.crown_csv.exists():
        raise FileNotFoundError(job.crown_csv)

    trees = pd.read_csv(job.tree_metadata_csv, low_memory=False)
    require_columns(trees, ["tree_id", "tree_lat", "tree_lon"], job.tree_metadata_csv)
    trees = trees.copy()
    trees["tree_lat"] = pd.to_numeric(trees["tree_lat"], errors="coerce")
    trees["tree_lon"] = pd.to_numeric(trees["tree_lon"], errors="coerce")
    trees = trees.loc[trees["tree_lat"].between(-90, 90) & trees["tree_lon"].between(-180, 180)].copy()
    trees = trees.reset_index(drop=True)

    crowns = load_crowns(job.crown_csv, args.min_crown_confidence)
    if trees.empty or crowns.empty:
        matched = empty_match_frame(trees)
    else:
        matches: list[pd.DataFrame] = []
        for epsg, crown_group in crowns.groupby("crown_epsg", sort=False):
            epsg_int = int(epsg)
            crown_index = crown_group.index.to_numpy(dtype=np.int64)
            crown_xy = crown_group[["crown_x", "crown_y"]].to_numpy(dtype=np.float64)
            tree_x, tree_y = transform_tree_coordinates(trees, epsg_int)
            tree_xy = np.column_stack([tree_x, tree_y])
            tree = cKDTree(crown_xy)
            distances, positions = tree.query(tree_xy, k=1, distance_upper_bound=float(args.match_radius_m))
            ok = np.isfinite(distances) & (positions < len(crown_group))
            if not ok.any():
                continue
            tree_part = trees.loc[ok].copy()
            tree_part["tree_x"] = tree_x[ok]
            tree_part["tree_y"] = tree_y[ok]
            matched_crown_index = crown_index[positions[ok].astype(np.int64)]
            crown_part = crowns.loc[matched_crown_index].copy().reset_index(drop=True)
            crown_lon, crown_lat = inverse_crown_coordinates(crown_part)
            tree_part = tree_part.reset_index(drop=True)
            tree_part["match_distance_m"] = distances[ok].astype(np.float32)
            tree_part["crown_id"] = crown_part["crown_id"].to_numpy(dtype=np.int64)
            tree_part["crown_source_row"] = crown_part["crown_source_row"].to_numpy(dtype=np.int64)
            tree_part["crown_lat"] = crown_lat
            tree_part["crown_lon"] = crown_lon
            tree_part["crown_x"] = crown_part["crown_x"].to_numpy(dtype=np.float64)
            tree_part["crown_y"] = crown_part["crown_y"].to_numpy(dtype=np.float64)
            tree_part["crown_epsg"] = crown_part["crown_epsg"].to_numpy(dtype=np.int64)
            tree_part["crown_confidence"] = crown_part["crown_confidence"].to_numpy(dtype=np.float32)
            optional_names = {
                "row_index": "crown_row_index",
                "reduced_id": "crown_reduced_id",
                "cell_id": "crown_cell_id",
                "chip_index": "crown_chip_index",
                "x": "crown_peak_x_pixel",
                "y": "crown_peak_y_pixel",
                "percent_vegetation": "crown_percent_vegetation",
                "percent_center_vegetation": "crown_percent_center_vegetation",
            }
            for optional, output_name in optional_names.items():
                if optional in crown_part.columns:
                    tree_part[output_name] = crown_part[optional].to_numpy()
            matches.append(tree_part)
        matched = pd.concat(matches, ignore_index=True) if matches else empty_match_frame(trees)
        if not matched.empty:
            matched = matched.sort_values(["tree_id", "match_distance_m"], kind="stable")
            matched = matched.drop_duplicates(subset=["tree_id"], keep="first").reset_index(drop=True)

    job.output_csv.parent.mkdir(parents=True, exist_ok=True)
    matched.to_csv(job.output_csv, index=False)
    match_rate = len(matched) / len(trees) * 100.0 if len(trees) else 0.0
    return {
        "city_token": job.city_token,
        "status": "completed",
        "tree_rows": int(len(trees)),
        "crown_rows": int(len(crowns)),
        "matched_rows": int(len(matched)),
        "unmatched_rows": int(len(trees) - len(matched)),
        "match_rate": float(match_rate),
        "output_csv": str(job.output_csv),
    }


def worker(payload: tuple[CityJob, argparse.Namespace]) -> dict[str, object]:
    job, args = payload
    try:
        result = match_city(job, args)
        print(
            f"{job.city_token}: {result['status']}; matched={int(result['matched_rows']):,}/"
            f"{0 if pd.isna(result['tree_rows']) else int(result['tree_rows']):,} "
            f"({0.0 if pd.isna(result['match_rate']) else float(result['match_rate']):.2f}%)",
            flush=True,
        )
        return result
    except Exception as error:
        print(f"{job.city_token}: FAILED: {error}", flush=True)
        return {
            "city_token": job.city_token,
            "status": "failed",
            "error": str(error),
            "output_csv": str(job.output_csv),
        }


def main() -> int:
    args = parse_args()
    if args.match_radius_m <= 0:
        raise SystemExit("--match-radius-m must be positive.")
    if args.parallel_workers < 1:
        raise SystemExit("--parallel-workers must be >= 1.")
    args.output_root.mkdir(parents=True, exist_ok=True)
    jobs = discover_jobs(args)
    if not jobs:
        raise SystemExit("No city jobs selected.")
    print(f"Spatial joining {len(jobs):,} city/cities; radius={args.match_radius_m:g}m", flush=True)

    if args.parallel_workers == 1:
        rows = [worker((job, args)) for job in jobs]
    else:
        with mp.get_context("spawn").Pool(processes=int(args.parallel_workers)) as pool:
            rows = list(pool.imap_unordered(worker, [(job, args) for job in jobs]))

    summary = pd.DataFrame(rows).sort_values("city_token", kind="stable")
    summary_path = args.summary_csv or args.output_root / "tree_to_detected_crowns_clean_summary.csv"
    summary.to_csv(summary_path, index=False, quoting=csv.QUOTE_MINIMAL)
    print("\nStatus counts:", flush=True)
    print(summary["status"].value_counts(dropna=False).to_string(), flush=True)
    completed = summary.loc[summary["status"].eq("completed")]
    if not completed.empty:
        total_trees = int(completed["tree_rows"].sum())
        total_matched = int(completed["matched_rows"].sum())
        total_rate = total_matched / total_trees * 100.0 if total_trees else 0.0
        print(
            f"Total matched={total_matched:,}/{total_trees:,} ({total_rate:.2f}%)",
            flush=True,
        )
    print(f"Wrote summary: {summary_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
