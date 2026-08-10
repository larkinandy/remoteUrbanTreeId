#!/usr/bin/env python3
"""Submit Google Satellite Embedding exports for missing tree-centered cells.

This mirrors the original McCoy satellite-embedding exporter, but its input is
the per-city missing-cell CSVs produced for the tree-centered Sentinel workflow.
Those cells are the tree-crown Sentinel grid cells not covered by the previous
cell-centered downloads, so they are also the cells needing Satellite Embedding
exports.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path

import pandas as pd


DATASET_ID = "GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL"
EMBEDDING_BANDS = [f"A{i:02d}" for i in range(64)]
ACTIVE_TASK_STATES = {"READY", "RUNNING"}

DEFAULT_POINTS_DIR = Path(r"E:\TreeCenteredModelInputs\tree_centered_sentinel_missing_raw15day")
DEFAULT_COMPLETED_DIR = Path(r"E:\TreeCenterSatelliteEmbedding")
DEFAULT_DRIVE_FOLDER = "GEE_TREE_CENTERED_SATELLITE_EMBEDDING"
DEFAULT_MANIFEST = Path("dataCollection/satellite_embedding_tree_centered_export_manifest.jsonl")
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
    parser.add_argument("--points-dir", type=Path, default=DEFAULT_POINTS_DIR)
    parser.add_argument("--points-file", type=Path)
    parser.add_argument("--city-token", action="append", default=None)
    parser.add_argument("--exclude-city-token", action="append", default=[])
    parser.add_argument("--completed-dir", type=Path, default=DEFAULT_COMPLETED_DIR)
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
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--project", default=os.getenv("GEE_PROJECT"))
    parser.add_argument("--limit-batches", type=int)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--authenticate", action="store_true")
    return parser.parse_args()


def normalize_token(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value or "").lower())


def sanitize_name(value: object, max_len: int = 90) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "_", str(value or "")).strip("_")
    return cleaned[:max_len] or "unknown"


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
        return [{"city": city, "points_file": args.points_file}]
    if not args.points_dir.exists():
        raise FileNotFoundError(args.points_dir)

    selected = None if args.city_token is None else {normalize_token(v) for v in args.city_token if str(v).strip()}
    excluded = {normalize_token(v) for v in args.exclude_city_token if str(v).strip()}
    paths = sorted(args.points_dir.glob("*/*_tree_centered_sentinel_cells_missing_raw15day.csv"))
    paths.extend(sorted(args.points_dir.glob("*_tree_centered_sentinel_cells_missing_raw15day.csv")))
    sources: list[dict[str, object]] = []
    seen: set[Path] = set()
    for path in paths:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        city = normalize_token(path.parent.name)
        match = re.match(r"(.+?)_tree_centered_sentinel_cells_missing_raw15day$", path.stem)
        if match:
            city = normalize_token(match.group(1))
        if selected is not None and city not in selected:
            continue
        if city in excluded:
            continue
        sources.append({"city": city, "points_file": path})
    if not sources:
        raise FileNotFoundError(f"No missing-cell CSVs found under {args.points_dir}")
    return sources


def load_submitted_keys(manifest_path: Path) -> set[str]:
    submitted: set[str] = set()
    if not manifest_path.exists():
        return submitted
    with manifest_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            key = record.get("key")
            if key:
                submitted.add(str(key))
    return submitted


def append_manifest(manifest_path: Path, record: dict[str, object]) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def completed_output_exists(completed_dir: Path, export_prefix: str) -> bool:
    if not completed_dir.exists():
        return False
    return any(completed_dir.glob(f"{export_prefix}*.csv"))


def task_state(task: object) -> str:
    if isinstance(task, dict):
        return str(task.get("state") or task.get("metadata", {}).get("state") or "").upper()
    return str(task.status().get("state") or "").upper()


class TaskCapacityGate:
    def __init__(self, max_active_tasks: int, poll_seconds: int, task_list_limit: int):
        self.max_active_tasks = int(max_active_tasks)
        self.poll_seconds = int(poll_seconds)
        self.task_list_limit = int(task_list_limit)
        self.cached_slots = 0

    def acquire(self) -> None:
        if self.cached_slots > 0:
            self.cached_slots -= 1
            print(f"  using cached Earth Engine task slot; remaining cached slots={self.cached_slots}", flush=True)
            return
        while True:
            available = self.refresh_available_slots()
            if available > 0:
                self.cached_slots = available - 1
                print(f"  reserving 1 task slot; cached slots after this submit={self.cached_slots}", flush=True)
                return
            print(
                f"{self.max_active_tasks} Earth Engine tasks are READY/RUNNING; "
                f"waiting {self.poll_seconds}s for capacity...",
                flush=True,
            )
            time.sleep(self.poll_seconds)

    def refresh_available_slots(self) -> int:
        started = time.time()
        print("  checking Earth Engine task capacity...", flush=True)
        try:
            tasks = ee.data.getTaskList()
        except Exception:
            tasks = ee.batch.Task.list()
        print(f"  fetched {len(tasks):,} Earth Engine task(s) in {time.time() - started:.1f}s", flush=True)
        inspected = tasks if self.task_list_limit <= 0 else tasks[: self.task_list_limit]
        active_count = 0
        for task in inspected:
            if task_state(task) in ACTIVE_TASK_STATES:
                active_count += 1
                if active_count >= self.max_active_tasks:
                    break
        available = max(0, self.max_active_tasks - active_count)
        suffix = "" if self.task_list_limit <= 0 else f" among {len(inspected):,} inspected recent task(s)"
        print(
            f"  task capacity available: {available} slot(s); "
            f"{active_count}/{self.max_active_tasks} active{suffix}",
            flush=True,
        )
        return available


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
    points_file = Path(source["points_file"])
    columns = pd.read_csv(points_file, nrows=0).columns.tolist()
    missing = [column for column in ("longitude", "latitude") if column not in columns]
    if missing:
        raise ValueError(f"{points_file} missing required columns: {missing}")
    use_columns = [column for column in POINT_PROPERTY_COLUMNS if column in columns]
    if "longitude" not in use_columns:
        use_columns.append("longitude")
    if "latitude" not in use_columns:
        use_columns.append("latitude")

    submitted_now = 0
    skipped_completed = 0
    planned = 0
    for batch_index, chunk in enumerate(pd.read_csv(points_file, usecols=use_columns, chunksize=args.batch_size, low_memory=False)):
        if args.limit_batches is not None and batch_index >= args.limit_batches:
            break
        print(f"{city}: batch {batch_index:05d}; rows={len(chunk):,}", flush=True)
        points = None
        for year in args.years:
            key = f"{points_file.name}|{year}|{batch_index}"
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
                    "source_csv": str(points_file),
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
    task_gate = None if args.dry_run else TaskCapacityGate(args.max_active_tasks, args.task_poll_seconds, args.task_list_limit)
    print(f"Discovered {len(sources):,} missing tree-centered embedding point source(s)", flush=True)
    print(f"Years: {', '.join(str(year) for year in args.years)}", flush=True)
    print(f"Drive folder: {args.drive_folder}", flush=True)
    print(f"Completed output check: {args.completed_dir}", flush=True)
    print(f"Manifest: {args.manifest}", flush=True)

    total_planned = 0
    total_submitted = 0
    total_skipped = 0
    for index, source in enumerate(sources, start=1):
        print(f"[{index:,}/{len(sources):,}] {source['city']}: {source['points_file']}", flush=True)
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
