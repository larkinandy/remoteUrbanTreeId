"""Export Sentinel-2 L2A measurements from Google Earth Engine.

This workflow samples reduced 10 m Sentinel cell-center points instead of every
tree. By default it discovers every
``sentinel_10m_cells/<city>/sentinel10m_unique_cells.shp`` file and
iterates over all cities. The exported rows are keyed by ``row_index``, which
defaults to the reduced-cell ``reduced_id``. Join ``row_index`` back to
each city's ``tree_to_sentinel10m_cell.csv.reduced_id`` when tree-level rows
are needed.

Example:
    python dataCollectionPreprocessing/Sentinel2/export_sentinel2_time_series_gee.py ^
        --project ee-larkinan ^
        --start-date 2021-01-01 ^
        --end-date 2024-01-01

City-specific example:
    python dataCollectionPreprocessing/Sentinel2/export_sentinel2_time_series_gee.py ^
        --city Albuquerque ^
        --project ee-larkinan ^
        --start-date 2021-01-01 ^
        --end-date 2024-01-01
"""

from __future__ import annotations

import argparse
from datetime import date, datetime, timedelta, timezone
import json
import os
from pathlib import Path
import re
import time
from typing import Iterable

import ee
from dotenv import load_dotenv


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_ENV_PATH = SCRIPT_DIR / ".env"
DEFAULT_START_DATE = "2021-01-01"
DEFAULT_END_DATE = "2023-12-31"
DEFAULT_INTERVAL_DAYS = 30
DEFAULT_DRIVE_FOLDER = "GEE_TREE_EXPORT"
DEFAULT_COMPLETED_DIR = Path(r"E:\TreeId\Sentinel2")
DEFAULT_POINTS_DIR = SCRIPT_DIR / "sentinel_10m_cells"
DEFAULT_CLOUD_FILTER = 70
DEFAULT_CLOUD_PROBABILITY_THRESHOLD = 40
ALBUQUERQUE_TEST_CITY = "Albuquerque"
ALBUQUERQUE_TEST_POINTS = (
    SCRIPT_DIR / "albuquerque_sentinel_10m_cells" / "sentinel10m_unique_cells.shp"
)

# Exact raw Sentinel-2 bands used by the preliminary deep learning model.
MODEL_BANDS = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]
QUALITY_BANDS = ["SCL"]
EXPORT_BANDS = MODEL_BANDS + QUALITY_BANDS

# Reject no data, saturated pixels, cloud shadow, clouds, cirrus, and snow/ice.
BAD_SCL_CLASSES = [0, 1, 3, 8, 9, 10, 11]

POINT_EXPORT_SELECTORS = ["row_index", "latitude", "longitude"]
METADATA_SELECTORS = [
    "date",
    "datetime",
    "source_image_id",
    "mgrs_tile",
    "cloudy_pixel_percentage",
    "valid_pixel",
]
EXPORT_SELECTORS = POINT_EXPORT_SELECTORS + METADATA_SELECTORS + MODEL_BANDS + QUALITY_BANDS
ACTIVE_TASK_STATES = {"READY", "RUNNING"}
STALE_TASK_STATE = "UNKNOWN_STALE_OPERATION"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", type=Path, default=DEFAULT_ENV_PATH)
    parser.add_argument(
        "--project",
        help="Earth Engine Cloud project. Defaults to GEE_PROJECT in the env file.",
    )
    parser.add_argument(
        "--points-asset",
        help=(
            "Earth Engine FeatureCollection asset containing reduced 10 m cell "
            "points. By default, the reduced_id property becomes row_index."
        ),
    )
    parser.add_argument(
        "--points-file",
        "--points-geojson",
        dest="points_file",
        type=Path,
        help=(
            "Local reduced-cell point file. Supports shapefile, GeoJSON, and "
            "CSV with latitude/longitude columns."
        ),
    )
    parser.add_argument(
        "--points-dir",
        type=Path,
        default=DEFAULT_POINTS_DIR,
        help=(
            "Folder containing per-city reduced point folders. Default: "
            "dataCollectionPreprocessing/Sentinel2/sentinel_10m_cells. Ignored when "
            "--points-file or --points-asset is supplied."
        ),
    )
    parser.add_argument(
        "--city",
        action="append",
        help=(
            "Optional city label/filter. Repeat to run multiple specific "
            "cities. Defaults to every discovered reduced point file."
        ),
    )
    parser.add_argument(
        "--test-albuquerque",
        "--test-alberquerque",
        dest="test_albuquerque",
        action="store_true",
        help=(
            "Use the local Albuquerque reduced-cell shapefile and city label "
            "for a test run. Explicit --city, --points-asset, or --points-file "
            "values override the preset."
        ),
    )
    parser.add_argument(
        "--row-index-property",
        default="reduced_id",
        help=(
            "Input point property to export as row_index. Use source_row or a "
            "CSV row field if exporting unreduced original points."
        ),
    )
    parser.add_argument("--start-date", default=DEFAULT_START_DATE)
    parser.add_argument("--end-date", default=DEFAULT_END_DATE)
    parser.add_argument("--interval-days", type=int, default=DEFAULT_INTERVAL_DAYS)
    parser.add_argument("--cloud-filter", type=float, default=DEFAULT_CLOUD_FILTER)
    parser.add_argument(
        "--cloud-probability-threshold",
        type=float,
        default=DEFAULT_CLOUD_PROBABILITY_THRESHOLD,
    )
    parser.add_argument("--tile-scale", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=5000)
    parser.add_argument(
        "--max-active-tasks",
        type=int,
        default=20,
        help=(
            "Maximum Earth Engine READY/RUNNING export tasks allowed before "
            "waiting to submit another task (default: 20)."
        ),
    )
    parser.add_argument(
        "--task-list-page-size",
        type=int,
        default=200,
        help=(
            "Maximum number of Earth Engine operations to request when checking "
            "active task capacity. Smaller values avoid very slow full task "
            "history scans (default: 200)."
        ),
    )
    parser.add_argument(
        "--skip-active-task-check",
        action="store_true",
        help=(
            "Submit exports without checking Earth Engine READY/RUNNING task "
            "capacity. Use only when the Earth Engine task-list endpoint is "
            "too slow or unavailable."
        ),
    )
    parser.add_argument(
        "--limit-batches",
        type=int,
        help="Testing only: submit at most this many point batches.",
    )
    parser.add_argument(
        "--export-destination",
        choices=["drive", "cloud-storage"],
        default="drive",
    )
    parser.add_argument("--drive-folder")
    parser.add_argument("--gcs-bucket", help="Cloud Storage bucket for exports.")
    parser.add_argument("--gcs-prefix", default="sentinel2")
    parser.add_argument("--description-prefix", default="s2_reduced_cells")
    parser.add_argument(
        "--completed-dir",
        type=Path,
        default=DEFAULT_COMPLETED_DIR,
        help=(
            "Local folder checked for completed CSV exports before submitting "
            "a matching Earth Engine task."
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--skip-asset-check",
        action="store_true",
        help="Skip the upfront FeatureCollection size check for --points-asset.",
    )
    parser.add_argument(
        "--authenticate",
        action="store_true",
        help="Run ee.Authenticate() before initialization.",
    )
    return parser.parse_args()


def apply_test_presets(args: argparse.Namespace) -> None:
    if not args.test_albuquerque:
        return
    if not args.city:
        args.city = [ALBUQUERQUE_TEST_CITY]
    if not args.points_asset and not args.points_file:
        args.points_file = ALBUQUERQUE_TEST_POINTS


def validate_args(args: argparse.Namespace) -> None:
    if args.points_asset and args.points_file:
        raise ValueError("Pass only one of --points-asset or --points-file")
    if args.points_asset and not args.city:
        raise ValueError("--points-asset requires --city because the city name cannot be inferred")
    if args.points_file and not args.points_file.exists():
        raise FileNotFoundError(f"Local point file does not exist: {args.points_file}")
    if not args.points_asset and not args.points_file and not args.points_dir.exists():
        raise FileNotFoundError(f"Reduced point directory does not exist: {args.points_dir}")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.max_active_tasks <= 0:
        raise ValueError("--max-active-tasks must be positive")
    if args.task_list_page_size <= 0:
        raise ValueError("--task-list-page-size must be positive")
    if args.limit_batches is not None and args.limit_batches <= 0:
        raise ValueError("--limit-batches must be positive")


def slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", value).strip("_").lower() or "unknown"


def parse_date(value: str) -> date:
    return date.fromisoformat(value)


def city_filter_slugs(args: argparse.Namespace) -> set[str]:
    return {slug(city) for city in args.city or []}


def manifest_city_lookup(points_dir: Path) -> dict[Path, str]:
    manifest_path = points_dir / "mccoy_sentinel10m_utm_reduction_manifest.json"
    if not manifest_path.exists():
        return {}
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    result = {}
    for entry in manifest:
        output_dir = entry.get("output_dir")
        city = entry.get("city")
        unique_cells = entry.get("unique_cells", 0)
        if output_dir and city and unique_cells:
            try:
                result[Path(output_dir).resolve()] = city
            except OSError:
                continue
    return result


def infer_city_from_points_file(path: Path, lookup: dict[Path, str] | None = None) -> str:
    parent = path.parent.resolve()
    if lookup and parent in lookup:
        return lookup[parent]
    return path.parent.name


def discover_point_sources(args: argparse.Namespace) -> list[dict[str, object]]:
    requested = city_filter_slugs(args)

    if args.points_asset:
        city = args.city[0]
        return [{"city": city, "points_asset": args.points_asset, "points_file": None}]

    if args.points_file:
        city = args.city[0] if args.city else infer_city_from_points_file(args.points_file)
        return [{"city": city, "points_asset": None, "points_file": args.points_file}]

    lookup = manifest_city_lookup(args.points_dir)
    paths = sorted(args.points_dir.glob("*/sentinel10m_unique_cells.shp"))
    if not paths:
        paths = sorted(args.points_dir.glob("*/sentinel10m_unique_cells.csv"))

    sources = []
    for path in paths:
        city = infer_city_from_points_file(path, lookup)
        if requested and slug(city) not in requested and slug(path.parent.name) not in requested:
            continue
        sources.append({"city": city, "points_asset": None, "points_file": path})

    if not sources:
        if requested:
            raise FileNotFoundError(
                f"No reduced point files found in {args.points_dir} for city filter(s): "
                f"{', '.join(args.city or [])}"
            )
        raise FileNotFoundError(f"No reduced point files found in {args.points_dir}")
    return sources


def load_completed_csv_stems(completed_dir: Path | None) -> list[str]:
    if not completed_dir or not completed_dir.exists():
        return []
    print(f"Indexing completed Sentinel CSVs in {completed_dir}...", flush=True)
    stems = sorted(
        path.stem for path in completed_dir.rglob("*.csv")
        if path.is_file() and path.stat().st_size > 0
    )
    print(f"Indexed {len(stems):,} completed Sentinel CSV file(s)", flush=True)
    return stems


def completed_output_exists(
    completed_dir: Path | None,
    file_prefix: str,
    completed_stems: list[str] | None = None,
) -> bool:
    if completed_stems is not None:
        return any(
            stem == file_prefix
            or stem.startswith(f"{file_prefix}-")
            or stem.startswith(f"{file_prefix}_")
            for stem in completed_stems
        )
    if not completed_dir or not completed_dir.exists():
        return False
    patterns = [
        f"{file_prefix}.csv",
        f"{file_prefix}-*.csv",
        f"{file_prefix}_*.csv",
    ]
    for pattern in patterns:
        for path in completed_dir.rglob(pattern):
            if path.is_file() and path.stat().st_size > 0:
                return True
    return False


def operation_state(operation: dict) -> str | None:
    metadata = operation.get("metadata") or {}
    state = metadata.get("state")
    if state:
        return str(state)
    if operation.get("done") is False:
        return "RUNNING"
    return None


def active_operation_count(project: str | None, page_size: int) -> int | None:
    """Count active EE operations without fetching the full task history.

    ``ee.batch.Task.list()`` can be very slow for accounts with many historical
    tasks. Newer Earth Engine clients expose ``ee.data.listOperations()``, which
    supports paged requests. We only need to know whether capacity exists, so a
    bounded page is enough for this gate.
    """
    list_operations = getattr(ee.data, "listOperations", None)
    if list_operations is None:
        return None

    parent = None
    if project:
        parent = project if project.startswith("projects/") else f"projects/{project}"

    call_attempts = []
    if parent:
        call_attempts.extend(
            [
                lambda: list_operations(parent, pageSize=page_size, filter="done=false"),
                lambda: list_operations(parent=parent, pageSize=page_size, filter="done=false"),
                lambda: list_operations(parent, page_size=page_size, filter="done=false"),
                lambda: list_operations(parent=parent, page_size=page_size, filter="done=false"),
            ]
        )
    if project:
        call_attempts.extend(
            [
                lambda: list_operations(project=project, pageSize=page_size, filter="done=false"),
                lambda: list_operations(project=project, page_size=page_size, filter="done=false"),
            ]
        )
    call_attempts.extend(
        [
            lambda: list_operations(pageSize=page_size, filter="done=false"),
            lambda: list_operations(page_size=page_size, filter="done=false"),
        ]
    )

    last_type_error = None
    for call in call_attempts:
        try:
            response = call()
            break
        except TypeError as exc:
            last_type_error = exc
            continue
    else:
        if last_type_error:
            return None
        return None

    if isinstance(response, dict):
        operations = response.get("operations", [])
    else:
        operations = response or []

    return sum(
        1 for operation in operations
        if operation.get("done") is False
        or operation_state(operation) in ACTIVE_TASK_STATES
    )


def task_list_state(task: ee.batch.Task) -> str | None:
    """Read the task-list state cached on an EE task without a status request."""
    for name in ("state", "_state"):
        state = getattr(task, name, None)
        if state:
            return str(state)

    for name in ("config", "_config"):
        config = getattr(task, name, None)
        if isinstance(config, dict):
            state = config.get("state") or config.get("status")
            if state:
                return str(state)

    task_vars = getattr(task, "__dict__", {})
    for value in task_vars.values():
        if isinstance(value, dict):
            state = value.get("state") or value.get("status")
            if state:
                return str(state)
    return None


def active_task_list_count(tasks: Iterable[ee.batch.Task]) -> tuple[int, int]:
    active_count = 0
    unknown_count = 0
    for task in tasks:
        state = task_list_state(task)
        if state in ACTIVE_TASK_STATES:
            active_count += 1
        elif state is None:
            unknown_count += 1
    return active_count, unknown_count


class TaskCapacityGate:
    def __init__(
        self,
        max_active_tasks: int,
        poll_seconds: int = 60,
        project: str | None = None,
        page_size: int = 200,
        skip_check: bool = False,
    ):
        self.max_active_tasks = max_active_tasks
        self.poll_seconds = poll_seconds
        self.project = project
        self.page_size = page_size
        self.skip_check = skip_check
        self.available_slots = 0

    def active_task_count(self) -> int:
        start_time = time.monotonic()
        active_count = active_operation_count(self.project, self.page_size)
        if active_count is not None:
            elapsed = time.monotonic() - start_time
            print(
                f"Checked Earth Engine active operations in {elapsed:.1f}s "
                f"(page size {self.page_size}).",
                flush=True,
            )
            return active_count

        print(
            "Paged Earth Engine operation listing is unavailable; falling back "
            "to full Task.list() history scan...",
            flush=True,
        )
        tasks = ee.batch.Task.list()
        elapsed = time.monotonic() - start_time
        print(f"Fetched Earth Engine task history in {elapsed:.1f}s.", flush=True)
        active_count, unknown_count = active_task_list_count(tasks)
        if unknown_count:
            print(
                f"Skipped {unknown_count:,} task(s) without cached state instead "
                "of calling task.status() for each historical task.",
                flush=True,
            )
        return active_count

    def wait_for_slot(self) -> None:
        if self.skip_check:
            print(
                "Skipping Earth Engine active task count check; submitting next task...",
                flush=True,
            )
            return

        if self.available_slots > 0:
            self.available_slots -= 1
            print(
                f"Using cached Earth Engine submission slot; "
                f"{self.available_slots} cached slot(s) remain before rechecking...",
                flush=True,
            )
            return

        while True:
            print(
                "Checking Earth Engine active task count "
                f"(bounded page size {self.page_size})...",
                flush=True,
            )
            active_count = self.active_task_count()
            if active_count < self.max_active_tasks:
                self.available_slots = self.max_active_tasks - active_count - 1
                print(
                    f"{active_count}/{self.max_active_tasks} Earth Engine tasks are READY/RUNNING; "
                    f"submitting next task with {self.available_slots} cached slot(s) remaining...",
                    flush=True,
                )
                return

            print(
                f"{active_count}/{self.max_active_tasks} Earth Engine tasks are READY/RUNNING; "
                f"waiting {self.poll_seconds}s for capacity...",
                flush=True,
            )
            time.sleep(self.poll_seconds)


def wait_for_task_capacity(max_active_tasks: int, poll_seconds: int = 60) -> None:
    while True:
        print("Checking Earth Engine active task count...", flush=True)
        tasks = ee.batch.Task.list()
        active_count, unknown_count = active_task_list_count(tasks)
        if unknown_count:
            print(
                f"Skipped {unknown_count:,} task(s) without cached state instead "
                "of calling task.status() for each historical task.",
                flush=True,
            )
        if active_count < max_active_tasks:
            print(
                f"{active_count}/{max_active_tasks} Earth Engine tasks are READY/RUNNING; "
                "submitting next task...",
                flush=True,
            )
            return

        print(
            f"{active_count}/{max_active_tasks} Earth Engine tasks are READY/RUNNING; "
            f"waiting {poll_seconds}s for capacity...",
            flush=True,
        )
        time.sleep(poll_seconds)


def is_stale_operation_error(exc: Exception) -> bool:
    message = str(exc)
    return "Operation" in message and "not found" in message


def safe_task_status(task: ee.batch.Task) -> dict:
    """Return task status, treating evicted Earth Engine operations as inactive.

    Earth Engine occasionally returns tasks from ``Task.list()`` whose backing
    operation can no longer be fetched. Those stale operation references should
    not crash the submission loop or count against the active task gate.
    """
    try:
        return task.status()
    except ee.ee_exception.EEException as exc:
        if not is_stale_operation_error(exc):
            raise
        operation = getattr(task, "operation_name", None) or getattr(task, "id", None) or "<unknown>"
        print(
            f"WARNING: Earth Engine task operation is no longer available; "
            f"treating as inactive: {operation}",
            flush=True,
        )
        return {"state": STALE_TASK_STATE, "error_message": str(exc)}


def date_windows(start: date, end: date, interval_days: int) -> Iterable[tuple[date, date]]:
    if end < start:
        raise ValueError("--end-date must not precede --start-date")
    if interval_days <= 0:
        raise ValueError("--interval-days must be positive")

    current = start
    exclusive_end = end + timedelta(days=1)
    while current < exclusive_end:
        window_end = min(current + timedelta(days=interval_days), exclusive_end)
        yield current, window_end
        current = window_end


def initialize_earth_engine(args: argparse.Namespace) -> None:
    load_dotenv(args.env_file)
    project = args.project or os.getenv("GEE_PROJECT")
    if not project:
        raise ValueError("Pass --project or define GEE_PROJECT in the env file")
    if not args.drive_folder:
        args.drive_folder = os.getenv("GEE_SENTINEL_FOLDER", DEFAULT_DRIVE_FOLDER)
    if args.authenticate:
        ee.Authenticate()
    ee.Initialize(project=project)
    args.project = project


def find_column(columns, candidates: list[str]) -> str | None:
    lookup = {str(column).lower(): str(column) for column in columns}
    for candidate in candidates:
        match = lookup.get(candidate.lower())
        if match:
            return match
    return None


def json_safe(value):
    if value is None:
        return None
    try:
        if hasattr(value, "item"):
            value = value.item()
    except ValueError:
        pass
    return value


def batched(values: list, size: int) -> Iterable[tuple[int, list]]:
    for start in range(0, len(values), size):
        yield start // size, values[start : start + size]


def point_feature(lon: float, lat: float, row_index) -> ee.Feature:
    return ee.Feature(
        ee.Geometry.Point([lon, lat]),
        {"row_index": json_safe(row_index), "longitude": lon, "latitude": lat},
    )


def local_point_records(path: Path, args: argparse.Namespace) -> list[dict[str, object]]:
    suffix = path.suffix.lower()
    records = []

    if suffix == ".csv":
        try:
            import pandas as pd
        except ModuleNotFoundError as exc:
            raise SystemExit("Install pandas to read local CSV point files") from exc

        table = pd.read_csv(path, low_memory=False)
        lon_col = find_column(
            table.columns,
            ["longitude", "lon", "longitude_coordinate", "longitude_coordinates"],
        )
        lat_col = find_column(
            table.columns,
            ["latitude", "lat", "latitude_coordinate", "latitude_coordinates"],
        )
        row_col = find_column(table.columns, [args.row_index_property, "reduced_id", "row_index"])
        if not lon_col or not lat_col or not row_col:
            raise ValueError(
                f"{path} must contain longitude, latitude, and {args.row_index_property!r} columns"
            )
        for values in table.to_dict(orient="records"):
            lon = float(values[lon_col])
            lat = float(values[lat_col])
            records.append({"row_index": json_safe(values[row_col]), "longitude": lon, "latitude": lat})
    else:
        try:
            import geopandas as gpd
        except ModuleNotFoundError as exc:
            raise SystemExit(
                "Install geopandas and pyogrio to read local shapefile/GeoJSON point files"
            ) from exc

        frame = gpd.read_file(path)
        if frame.empty:
            raise ValueError(f"No point features found in {path}")
        if frame.crs is None:
            raise ValueError(f"Local point file has no CRS: {path}")
        frame = frame.to_crs("EPSG:4326")
        row_col = find_column(frame.columns, [args.row_index_property, "reduced_id", "row_index"])
        if not row_col:
            raise ValueError(f"{path} must contain {args.row_index_property!r} or reduced_id")
        non_points = frame.geometry.geom_type != "Point"
        if non_points.any():
            raise ValueError(f"{path} contains {int(non_points.sum()):,} non-point features")
        for _, row in frame.iterrows():
            lon = float(row.geometry.x)
            lat = float(row.geometry.y)
            records.append({"row_index": json_safe(row[row_col]), "longitude": lon, "latitude": lat})

    if not records:
        raise ValueError(f"No valid local points were loaded from {path}")
    print(f"Loaded {len(records):,} local points: {path}")
    return records


def feature_collection_from_local_batch(records: list[dict[str, object]]) -> ee.FeatureCollection:
    return ee.FeatureCollection(
        [
            point_feature(
                float(record["longitude"]),
                float(record["latitude"]),
                record["row_index"],
            )
            for record in records
        ]
    )


def load_point_batches(args: argparse.Namespace) -> list[tuple[int, ee.FeatureCollection, int]]:
    if args.points_file:
        records = local_point_records(args.points_file, args)
        batches = []
        for batch_index, chunk in batched(records, args.batch_size):
            if args.limit_batches is not None and len(batches) >= args.limit_batches:
                break
            batches.append((batch_index, feature_collection_from_local_batch(chunk), len(chunk)))
        print(f"Prepared {len(batches):,} local point batches of <= {args.batch_size:,}")
        return batches

    points = ee.FeatureCollection(args.points_asset)
    point_count = None

    if args.points_asset and not args.skip_asset_check:
        try:
            point_count = points.size().getInfo()
        except ee.EEException as exc:
            raise RuntimeError(
                "Could not load the Earth Engine point asset "
                f"{args.points_asset!r}. Upload the reduced-cell table first, "
                "or pass a local reduced-cell file with --points-file."
            ) from exc
        print(f"Loaded point asset with {point_count:,} features: {args.points_asset}")

    def keep_export_properties(feature: ee.Feature) -> ee.Feature:
        geometry_wgs84 = feature.geometry().transform("EPSG:4326", 1)
        coordinates = geometry_wgs84.coordinates()
        return ee.Feature(
            geometry_wgs84,
            {
                "row_index": feature.get(args.row_index_property),
                "longitude": coordinates.get(0),
                "latitude": coordinates.get(1),
            },
        )

    mapped = points.map(keep_export_properties)
    if point_count is None:
        point_count = mapped.size().getInfo()
    batches = []
    for batch_index, start in enumerate(range(0, point_count, args.batch_size)):
        if args.limit_batches is not None and len(batches) >= args.limit_batches:
            break
        batch_points = ee.FeatureCollection(mapped.toList(args.batch_size, start))
        batches.append((batch_index, batch_points, min(args.batch_size, point_count - start)))
    print(f"Prepared {len(batches):,} asset point batches of <= {args.batch_size:,}")
    return batches


def joined_sentinel_collection(
    points: ee.FeatureCollection,
    start: date,
    end: date,
    cloud_filter: float,
) -> ee.ImageCollection:
    start_str = start.isoformat()
    end_str = end.isoformat()
    s2_sr = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterDate(start_str, end_str)
        .filterBounds(points)
        .filter(ee.Filter.lte("CLOUDY_PIXEL_PERCENTAGE", cloud_filter))
    )
    s2_clouds = (
        ee.ImageCollection("COPERNICUS/S2_CLOUD_PROBABILITY")
        .filterDate(start_str, end_str)
        .filterBounds(points)
    )
    return ee.ImageCollection(
        ee.Join.saveFirst("s2cloudless").apply(
            primary=s2_sr,
            secondary=s2_clouds,
            condition=ee.Filter.equals(
                leftField="system:index",
                rightField="system:index",
            ),
        )
    )


def mask_sentinel_image(
    image: ee.Image,
    cloud_probability_threshold: float,
) -> ee.Image:
    cloud_probability = ee.Image(image.get("s2cloudless")).select("probability")
    cloud_good = cloud_probability.lt(cloud_probability_threshold)

    scl = image.select("SCL")
    scl_good = ee.Image(1)
    for scl_class in BAD_SCL_CLASSES:
        scl_good = scl_good.And(scl.neq(scl_class))

    valid_mask = cloud_good.And(scl_good)
    selected = image.select(EXPORT_BANDS)
    continuous = selected.select(MODEL_BANDS).resample("bilinear")
    quality = selected.select(QUALITY_BANDS)
    output = continuous.addBands(quality).updateMask(valid_mask)

    return output.set(
        {
            "date": image.date().format("YYYY-MM-dd"),
            "datetime": image.date().format("YYYY-MM-dd'T'HH:mm:ss'Z'"),
            "source_image_id": image.get("system:index"),
            "mgrs_tile": image.get("MGRS_TILE"),
            "cloudy_pixel_percentage": image.get("CLOUDY_PIXEL_PERCENTAGE"),
        }
    )


def sample_one_image(
    image: ee.Image,
    points: ee.FeatureCollection,
    tile_scale: int,
) -> ee.FeatureCollection:
    samples = image.sampleRegions(
        collection=points,
        properties=POINT_EXPORT_SELECTORS,
        scale=10,
        geometries=False,
        tileScale=tile_scale,
    )

    def add_metadata(feature: ee.Feature) -> ee.Feature:
        return feature.set(
            {
                "date": image.get("date"),
                "datetime": image.get("datetime"),
                "source_image_id": image.get("source_image_id"),
                "mgrs_tile": image.get("mgrs_tile"),
                "cloudy_pixel_percentage": image.get("cloudy_pixel_percentage"),
                "valid_pixel": True,
            }
        )

    return samples.map(add_metadata)


def build_export_table(
    points: ee.FeatureCollection,
    start: date,
    end: date,
    args: argparse.Namespace,
) -> ee.FeatureCollection:
    joined = joined_sentinel_collection(points, start, end, args.cloud_filter)
    masked = joined.map(
        lambda image: mask_sentinel_image(image, args.cloud_probability_threshold)
    )
    return masked.map(lambda image: sample_one_image(image, points, args.tile_scale)).flatten()


def start_export(
    table: ee.FeatureCollection,
    description: str,
    file_prefix: str,
    args: argparse.Namespace,
) -> ee.batch.Task:
    if args.export_destination == "drive":
        return ee.batch.Export.table.toDrive(
            collection=table,
            description=description,
            folder=args.drive_folder,
            fileNamePrefix=file_prefix,
            fileFormat="CSV",
            selectors=EXPORT_SELECTORS,
        )

    if not args.gcs_bucket:
        raise ValueError("--gcs-bucket is required for --export-destination cloud-storage")
    return ee.batch.Export.table.toCloudStorage(
        collection=table,
        description=description,
        bucket=args.gcs_bucket,
        fileNamePrefix=f"{args.gcs_prefix.rstrip('/')}/{file_prefix}",
        fileFormat="CSV",
        selectors=EXPORT_SELECTORS,
    )


def run_city_exports(
    args: argparse.Namespace,
    start: date,
    end: date,
    submitted_at: str,
    completed_stems: list[str],
    task_gate: TaskCapacityGate,
) -> tuple[list[ee.batch.Task], int, int]:
    point_batches = load_point_batches(args)
    city_slug = slug(args.city)

    tasks = []
    skipped_completed = 0
    for window_start, window_end in date_windows(start, end, args.interval_days):
        inclusive_end = window_end - timedelta(days=1)
        for batch_index, points, batch_count in point_batches:
            file_prefix = (
                f"{args.description_prefix}_{city_slug}_"
                f"{window_start:%Y%m%d}_{inclusive_end:%Y%m%d}_"
                f"batch_{batch_index:05d}"
            )
            if completed_output_exists(args.completed_dir, file_prefix, completed_stems):
                skipped_completed += 1
                print(f"Skipping local completed CSV: {file_prefix}", flush=True)
                continue

            print(
                f"Prepared {args.city}: {window_start.isoformat()} through "
                f"{inclusive_end.isoformat()}, batch {batch_index:05d} "
                f"({batch_count:,} points) -> {file_prefix}",
                flush=True,
            )
            if args.dry_run:
                continue
            task_gate.wait_for_slot()
            print(f"Building Earth Engine export table: {file_prefix}", flush=True)
            table = build_export_table(points, window_start, window_end, args)
            description = f"{file_prefix}_{submitted_at}"
            task = start_export(table, description, file_prefix, args)
            print(f"Starting Earth Engine export task: {description}", flush=True)
            task.start()
            tasks.append(task)
            print(f"Started task: {description}", flush=True)

    window_count = len(list(date_windows(start, end, args.interval_days)))
    planned_count = window_count * len(point_batches)
    print(
        f"{args.city}: prepared {window_count:,} windows x {len(point_batches):,} "
        f"point batches = {planned_count:,} export task(s)"
    )
    if skipped_completed:
        print(f"{args.city}: skipped {skipped_completed:,} completed task(s) in {args.completed_dir}")
    if tasks:
        print(f"{args.city}: started {len(tasks):,} Earth Engine export task(s)")
        for task in tasks:
            print(safe_task_status(task))
    return tasks, skipped_completed, planned_count


def main() -> int:
    args = parse_args()
    apply_test_presets(args)
    validate_args(args)
    initialize_earth_engine(args)
    start = parse_date(args.start_date)
    end = parse_date(args.end_date)
    submitted_at = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    sources = discover_point_sources(args)
    completed_stems = load_completed_csv_stems(args.completed_dir)
    task_gate = TaskCapacityGate(
        args.max_active_tasks,
        project=args.project,
        page_size=args.task_list_page_size,
        skip_check=args.skip_active_task_check,
    )

    print(f"Discovered {len(sources):,} city point source(s)")
    print(f"Date range: {start} through {end}; interval days: {args.interval_days}")
    print(f"Completed output check: {args.completed_dir}")
    print(f"Drive folder: {args.drive_folder}")

    all_tasks = []
    total_skipped = 0
    total_planned = 0
    for index, source in enumerate(sources, start=1):
        city_args = argparse.Namespace(**vars(args))
        city_args.city = source["city"]
        city_args.points_asset = source["points_asset"]
        city_args.points_file = source["points_file"]
        print(f"[{index:,}/{len(sources):,}] {city_args.city}: {city_args.points_file or city_args.points_asset}")
        tasks, skipped_completed, planned_count = run_city_exports(
            city_args,
            start,
            end,
            submitted_at,
            completed_stems,
            task_gate,
        )
        all_tasks.extend(tasks)
        total_skipped += skipped_completed
        total_planned += planned_count

    print(f"All cities: planned {total_planned:,} export task(s)")
    if total_skipped:
        print(f"All cities: skipped {total_skipped:,} completed task(s)")
    if all_tasks:
        print(f"All cities: started {len(all_tasks):,} Earth Engine export task(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
