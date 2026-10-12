#!/usr/bin/env python3
"""Export annual Google Satellite Embeddings for all crown-centered 10 m cells.

The canonical input is the final tree-centered NAIP metadata. Crown coordinates
are converted to unique 10 m grid cells, and one embedding is requested for each
cell and year. A prepared point CSV may be supplied for reruns, but Sentinel-2
missing-cell inventories are not part of the from-scratch workflow.
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import pandas as pd

from satellite_embedding_utils import normalize_city, unique_crown_cell_points
from earth_engine_export_utils import (
    TaskCapacityGate,
    append_manifest,
    completed_output_exists,
    load_submitted_keys,
    sanitize_name,
)


DATASET_ID = "GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL"
EMBEDDING_BANDS = [f"A{i:02d}" for i in range(64)]
DEFAULT_DRIVE_FOLDER = "GEE_TREE_CENTERED_SATELLITE_EMBEDDING"
DEFAULT_DESCRIPTION_PREFIX = "tree_centered_satellite_embedding"

POINT_PROPERTY_COLUMNS = [
    "reduced_id",
    "row_index",
    "crown_cell_id",
    "crown_cell_epsg",
    "crown_cell_col",
    "crown_cell_row",
    "tree_centered_record_count",
    "longitude",
    "latitude",
]

ee = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--crop-root", type=Path, help="Root containing one final NAIP crop-metadata directory per city.")
    source.add_argument("--points-file", type=Path, help="Prepared crown-cell CSV for a single city.")
    parser.add_argument("--metadata-pattern", default="*_tree_id_centered_nearest_64px_metadata.csv")
    parser.add_argument("--city-token", action="append", default=None)
    parser.add_argument("--exclude-city-token", action="append", default=[])
    parser.add_argument("--completed-dir", type=Path, help="Optional directory containing already-downloaded exports.")
    parser.add_argument("--drive-folder", default=DEFAULT_DRIVE_FOLDER)
    parser.add_argument("--description-prefix", default=DEFAULT_DESCRIPTION_PREFIX)
    parser.add_argument("--years", nargs="+", type=int, default=[2021, 2022, 2023])
    parser.add_argument("--batch-size", type=int, default=5000)
    parser.add_argument("--max-active-tasks", type=int, default=20)
    parser.add_argument("--task-poll-seconds", type=int, default=60)
    parser.add_argument(
        "--task-list-limit",
        type=int,
        default=500,
        help=(
            "Only inspect this many most-recent Earth Engine task records when checking capacity. "
            "The Earth Engine Python API returns the full task list, but limiting local inspection "
            "avoids calling status() across thousands of old completed tasks. Use 0 to inspect all."
        ),
    )
    parser.add_argument("--tile-scale", type=int, default=4)
    parser.add_argument("--manifest", type=Path, required=True, help="Runtime JSONL submission ledger (generated; do not commit).")
    parser.add_argument("--project", default=os.getenv("GEE_PROJECT"))
    parser.add_argument("--limit-batches", type=int)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--authenticate", action="store_true")
    return parser.parse_args()


def normalize_token(value: object) -> str:
    return normalize_city(value)


def initialize_earth_engine(args: argparse.Namespace) -> None:
    global ee
    import ee as earth_engine

    ee = earth_engine
    if args.authenticate:
        ee.Authenticate()
    if args.project:
        ee.Initialize(project=args.project)
    else:
        ee.Initialize()


def discover_sources(args: argparse.Namespace) -> list[dict[str, object]]:
    if args.points_file:
        if not args.points_file.exists():
            raise FileNotFoundError(args.points_file)
        city = normalize_token(args.city_token[0] if args.city_token else args.points_file.parent.name)
        return [{"city": city, "source_file": args.points_file, "prepared": True}]
    if not args.crop_root.exists():
        raise FileNotFoundError(args.crop_root)

    selected = None if args.city_token is None else {normalize_token(v) for v in args.city_token if str(v).strip()}
    excluded = {normalize_token(v) for v in args.exclude_city_token if str(v).strip()}
    paths = sorted(args.crop_root.glob(f"*/{args.metadata_pattern}"))
    sources: list[dict[str, object]] = []
    seen: set[Path] = set()
    for path in paths:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        city = normalize_token(path.parent.name)
        if selected is not None and city not in selected:
            continue
        if city in excluded:
            continue
        sources.append({"city": city, "source_file": path, "prepared": False})
    if not sources:
        raise FileNotFoundError(f"No crop metadata matching {args.metadata_pattern!r} found under {args.crop_root}")
    return sources


def load_points(source: dict[str, object]) -> pd.DataFrame:
    path = Path(source["source_file"])
    frame = pd.read_csv(path, low_memory=False)
    if bool(source["prepared"]):
        missing = sorted({"longitude", "latitude"}.difference(frame.columns))
        if missing:
            raise ValueError(f"{path} missing required columns: {missing}")
        return frame
    return unique_crown_cell_points(frame)


def export_name(description_prefix: str, city: str, year: int, batch_index: int) -> str:
    return sanitize_name(f"{description_prefix}_{city}_{year}_batch_{batch_index:05d}")


def clean_property(value):
    if pd.isna(value):
        return ""
    if hasattr(value, "item"):
        return value.item()
    return value


def chunk_to_feature_collection(chunk: pd.DataFrame, city: str, batch_index: int):
    chunk = chunk.copy()
    chunk["longitude"] = pd.to_numeric(chunk["longitude"], errors="coerce")
    chunk["latitude"] = pd.to_numeric(chunk["latitude"], errors="coerce")
    chunk = chunk.dropna(subset=["longitude", "latitude"])
    chunk = chunk[chunk["longitude"].between(-180, 180) & chunk["latitude"].between(-90, 90)]
    if chunk.empty:
        return None

    features = []
    for source_row_number, row in chunk.iterrows():
        props = {
            "tree_centered_city": city,
            "tree_centered_batch": int(batch_index),
            "source_csv_row": int(source_row_number) + 2,
        }
        for column in POINT_PROPERTY_COLUMNS:
            if column in chunk.columns:
                props[column] = clean_property(row[column])
        point = ee.Geometry.Point([float(row["longitude"]), float(row["latitude"])])
        features.append(ee.Feature(point, props))
    return ee.FeatureCollection(features)


def embedding_image_for_year(year: int, points):
    return (
        ee.ImageCollection(DATASET_ID)
        .filterDate(f"{year}-01-01", f"{year + 1}-01-01")
        .filterBounds(points)
        .select(EMBEDDING_BANDS)
        .mosaic()
    )


def export_batch(points, city: str, year: int, batch_index: int, args: argparse.Namespace):
    description = export_name(args.description_prefix, city, year, batch_index)
    if args.dry_run:
        print(f"DRY RUN: {description} -> Drive/{args.drive_folder}/{description}.csv", flush=True)
        return None
    selectors = [
        "tree_centered_city",
        "tree_centered_batch",
        "source_csv_row",
        *POINT_PROPERTY_COLUMNS,
        "embedding_year",
        *EMBEDDING_BANDS,
    ]
    sampled = embedding_image_for_year(year, points).sampleRegions(
        collection=points,
        properties=[s for s in selectors if s not in EMBEDDING_BANDS and s != "embedding_year"],
        scale=10,
        geometries=False,
        tileScale=args.tile_scale,
    )
    sampled = sampled.map(lambda feature: feature.set("embedding_year", year))
    task = ee.batch.Export.table.toDrive(
        collection=sampled,
        description=description,
        folder=args.drive_folder,
        fileNamePrefix=description,
        fileFormat="CSV",
        selectors=selectors,
    )
    task.start()
    print(f"Started {description}: {task.id}", flush=True)
    return task


def validate_args(args: argparse.Namespace) -> None:
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.max_active_tasks <= 0:
        raise ValueError("--max-active-tasks must be positive")
    if args.task_poll_seconds <= 0:
        raise ValueError("--task-poll-seconds must be positive")
    if args.task_list_limit < 0:
        raise ValueError("--task-list-limit must be non-negative")
    if args.tile_scale <= 0:
        raise ValueError("--tile-scale must be positive")
    if args.limit_batches is not None and args.limit_batches <= 0:
        raise ValueError("--limit-batches must be positive")


def run_city(
    source: dict[str, object],
    args: argparse.Namespace,
    submitted_keys: set[str],
    task_gate: TaskCapacityGate | None,
) -> tuple[int, int, int]:
    city = str(source["city"])
    source_file = Path(source["source_file"])
    point_frame = load_points(source)
    use_columns = [column for column in POINT_PROPERTY_COLUMNS if column in point_frame.columns]

    submitted_now = 0
    skipped_completed = 0
    planned = 0
    for batch_index, start in enumerate(range(0, len(point_frame), args.batch_size)):
        chunk = point_frame.iloc[start : start + args.batch_size][use_columns]
        if args.limit_batches is not None and batch_index >= args.limit_batches:
            break
        print(f"{city}: batch {batch_index:05d}; rows={len(chunk):,}", flush=True)
        points = None
        for year in args.years:
            key = f"{city}|{year}|{batch_index}"
            prefix = export_name(args.description_prefix, city, year, batch_index)
            if completed_output_exists(args.completed_dir, prefix):
                print(f"{city}: skipping local completed CSV: {prefix}.csv", flush=True)
                skipped_completed += 1
                continue
            if key in submitted_keys:
                print(f"{city}: skipping already-submitted export: {key}", flush=True)
                continue
            if points is None and not args.dry_run:
                print(f"{city}: building Earth Engine point collection for batch {batch_index:05d}", flush=True)
                points = chunk_to_feature_collection(chunk, city, batch_index)
                print(f"{city}: point collection ready for batch {batch_index:05d}", flush=True)
            if points is None and not args.dry_run:
                continue
            planned += 1
            print(f"{city}: preparing export year={year}; batch={batch_index:05d}; prefix={prefix}", flush=True)
            if task_gate is not None:
                task_gate.acquire()
            task = export_batch(points, city, year, batch_index, args)
            append_manifest(
                args.manifest,
                {
                    "key": key,
                    "city": city,
                    "source_name": source_file.name,
                    "year": int(year),
                    "batch_index": int(batch_index),
                    "batch_size": int(args.batch_size),
                    "drive_folder": args.drive_folder,
                    "task_id": task.id if task else None,
                    "dry_run": bool(args.dry_run),
                    "submitted_at_unix": int(time.time()),
                },
            )
            submitted_keys.add(key)
            submitted_now += 1
    return planned, submitted_now, skipped_completed


def main() -> int:
    args = parse_args()
    validate_args(args)
    if not args.dry_run:
        initialize_earth_engine(args)
    sources = discover_sources(args)
    submitted_keys = load_submitted_keys(args.manifest)
    task_gate = None if args.dry_run else TaskCapacityGate(ee, args.max_active_tasks, args.task_poll_seconds, args.task_list_limit)
    print(f"Discovered {len(sources):,} crown-centered embedding source(s)", flush=True)
    print(f"Years: {', '.join(str(year) for year in args.years)}", flush=True)
    print(f"Drive folder: {args.drive_folder}", flush=True)
    print(f"Completed output check: {args.completed_dir or 'disabled'}", flush=True)
    print(f"Manifest: {args.manifest}", flush=True)

    total_planned = 0
    total_submitted = 0
    total_skipped = 0
    for index, source in enumerate(sources, start=1):
        print(f"[{index:,}/{len(sources):,}] {source['city']}: {source['source_file']}", flush=True)
        planned, submitted, skipped = run_city(source, args, submitted_keys, task_gate)
        total_planned += planned
        total_submitted += submitted
        total_skipped += skipped
    print(f"All cities: planned {total_planned:,} export task(s)", flush=True)
    print(f"All cities: submitted/planned now {total_submitted:,} export task(s)", flush=True)
    if total_skipped:
        print(f"All cities: skipped {total_skipped:,} completed local export(s)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
