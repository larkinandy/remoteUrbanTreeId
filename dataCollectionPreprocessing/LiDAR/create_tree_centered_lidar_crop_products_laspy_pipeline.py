#!/usr/bin/env python3
"""Create tree-centered DTM, DSM, CHM, and point-count products from LAZ/LAS.

This is the single entry point for the clean tree_id-centered LiDAR workflow.
It reads current crown metadata directly and bins LiDAR points into the
footprint of each tree-centered NAIP crop.

Default outputs:
  E:/TreeCenteredModelInputs/tree_centered_lidar_products_all_cell_screen_disabled/Candidates/<city>/<project>/Binned/
  E:/TreeCenteredModelInputs/tree_centered_lidar_products_all_cell_screen_disabled/DTM/<city>/
  E:/TreeCenteredModelInputs/tree_centered_lidar_products_all_cell_screen_disabled/DSM/<city>/
  E:/TreeCenteredModelInputs/tree_centered_lidar_products_all_cell_screen_disabled/CHM/<city>/
  E:/TreeCenteredModelInputs/tree_centered_lidar_products_all_cell_screen_disabled/Point_Counts/<city>/
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import re
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import tree_centered_lidar_utils as lidar_utils


DEFAULT_CROP_ROOT = Path(r"H:\TreeCenteredModelInputs\tree_centered_naip_crops_clean")
DEFAULT_LIDAR_ROOT = Path(r"E:\LiDAR")
DEFAULT_OUTPUT_ROOT = Path(r"H:\TreeCenteredModelInputs\tree_centered_lidar_products_clean")
DEFAULT_RUN_SUMMARY = HERE / "tree_centered_lidar_crop_products_laspy_summary.csv"


@dataclass(frozen=True)
class TreeCropRow:
    position: int
    tree_id: int
    tree_centered_index: int
    crop_index: int
    row_index: int
    cell_id: str
    cell_epsg: int
    center_x: float
    center_y: float
    crop_metres: float
    crop_size: int
    source_file: str
    source_row: int
    taxon_label: str
    crop_failed: bool


@dataclass(frozen=True)
class TreeCityJob:
    token: str
    name: str
    code: str
    project: str
    preference_rank: int
    metadata_path: Path
    manifest_rows: list[dict[str, str]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--stage", choices=("all", "bin", "derive"), default="all")
    parser.add_argument("--manifest", type=Path, default=lidar_utils.DEFAULT_MANIFEST)
    parser.add_argument("--city-summary", type=Path, default=lidar_utils.DEFAULT_CITY_SUMMARY)
    parser.add_argument("--crop-root", type=Path, default=DEFAULT_CROP_ROOT)
    parser.add_argument("--lidar-root", type=Path, default=DEFAULT_LIDAR_ROOT)
    parser.add_argument(
        "--use-manifest-local-path",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Prefer the manifest local_path column when that file exists; otherwise use --lidar-root/relative_path.",
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--candidate-root", type=Path)
    parser.add_argument("--dtm-root", type=Path)
    parser.add_argument("--dsm-root", type=Path)
    parser.add_argument("--chm-root", type=Path)
    parser.add_argument("--count-root", type=Path)
    parser.add_argument("--run-summary", type=Path, default=DEFAULT_RUN_SUMMARY)
    parser.add_argument("--city-token", action="append", default=[], help="Restrict to city token/name/code. Repeatable.")
    parser.add_argument("--exclude-city-token", action="append", default=[])
    parser.add_argument("--metadata-pattern", default="*_tree_centered_nearest_64px_metadata.csv")
    parser.add_argument("--max-cities", type=int, default=0)
    parser.add_argument("--max-records", type=int, default=0)
    parser.add_argument("--max-tiles", type=int, default=0)
    parser.add_argument(
        "--crop-metres",
        type=float,
        default=0.0,
        help="Override footprint width in metres. 0 uses crop_metres from metadata, usually 38.",
    )
    parser.add_argument("--pixel-size", type=float, default=1.0)
    parser.add_argument(
        "--pixels",
        type=int,
        default=0,
        help="Override output pixels per side. 0 uses crop_metres / pixel-size; for 38 m crops at 1 m this gives 38x38.",
    )
    parser.add_argument("--target-epsg", type=int, default=0, help="Output CRS. 0 uses the most common parsed metadata EPSG per city.")
    parser.add_argument("--laspy-source-epsg", type=int, default=0)
    parser.add_argument("--z-scale", type=float, default=1.0)
    parser.add_argument("--z-units", default="source")
    parser.add_argument("--z-scale-table", type=Path)
    parser.add_argument("--auto-z-scale", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--z-scale-audit-tiles", type=int, default=3)
    parser.add_argument("--dtm-class-codes", default="2")
    parser.add_argument("--dsm-class-codes", default="1,2,3,4,5,6,9,17")
    parser.add_argument("--checkpoint-every-tiles", type=int, default=1)
    parser.add_argument(
        "--tile-point-chunk-size",
        type=int,
        default=0,
        help=(
            "If >0, read each LAZ/LAS tile in point chunks of this size instead of as one full "
            "tile. Useful for rescuing pathological tiles that stall or exhaust memory."
        ),
    )
    parser.add_argument(
        "--tile-chunk-progress-interval",
        type=int,
        default=5,
        help="Print progress every N point chunks when --tile-point-chunk-size is enabled.",
    )
    parser.add_argument(
        "--retry-active-tile",
        action="store_true",
        help=(
            "Allow retrying a tile left in bin_state.active_tile by a stopped run. Use only when "
            "you have evidence that the active tile did not flush partial updates, or after --overwrite."
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--allow-empty-products", action="store_true")
    parser.add_argument("--auto-accept-coverage", type=float, default=0.75)
    parser.add_argument("--preference-penalty", type=float, default=0.10)
    parser.add_argument(
        "--derive-workers",
        type=int,
        default=1,
        help="Number of parallel city workers for the derive stage.",
    )
    parser.add_argument(
        "--bin-workers",
        type=int,
        default=1,
        help=(
            "Number of parallel candidate city/project workers for the bin stage. Keep low for "
            "large cities because each worker reads LAZ tiles and writes large memmaps."
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def normalize_token(value: object) -> str:
    return lidar_utils.normalize_token(value)


def ensure_roots(args: argparse.Namespace) -> None:
    args.candidate_root = args.candidate_root or args.output_root / "Candidates"
    args.dtm_root = args.dtm_root or args.output_root / "DTM"
    args.dsm_root = args.dsm_root or args.output_root / "DSM"
    args.chm_root = args.chm_root or args.output_root / "CHM"
    args.count_root = args.count_root or args.output_root / "Point_Counts"


def parse_bool(value: object) -> bool:
    text = str(value or "").strip().lower()
    return text in {"1", "true", "t", "yes", "y"}


def parse_int(value: object, default: int = -1) -> int:
    try:
        if str(value).strip() == "":
            return default
        return int(float(value))
    except Exception:
        return default


def parse_float(value: object, default: float = math.nan) -> float:
    try:
        if str(value).strip() == "":
            return default
        return float(value)
    except Exception:
        return default


def epsg_from_cell_id(cell_id: object) -> int:
    match = re.search(r"epsg(\d+)", str(cell_id or "").lower())
    return int(match.group(1)) if match else 0


def discover_metadata(crop_root: Path, pattern: str) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for city_dir in sorted(path for path in crop_root.iterdir() if path.is_dir()):
        matches = sorted(city_dir.glob(pattern))
        if not matches:
            continue
        if len(matches) > 1:
            print(f"WARNING {city_dir.name}: multiple metadata files; using {matches[0].name}", flush=True)
        out[normalize_token(city_dir.name)] = matches[0]
    return out


def read_tree_rows(path: Path, args: argparse.Namespace) -> list[TreeCropRow]:
    rows: list[TreeCropRow] = []
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        fields = set(reader.fieldnames or [])
        required = {"crop_index", "crop_failed"}
        missing = sorted(required.difference(fields))
        if not ({"peak_x_utm", "crown_x_utm"} & fields):
            missing.append("peak_x_utm or crown_x_utm")
        if not ({"peak_y_utm", "crown_y_utm"} & fields):
            missing.append("peak_y_utm or crown_y_utm")
        if not ({"crown_epsg", "cell_id"} & fields):
            missing.append("crown_epsg or cell_id")
        if missing:
            raise ValueError(f"{path} is missing required columns: {missing}")
        for position, row in enumerate(reader):
            crop_failed = parse_bool(row.get("crop_failed"))
            crop_metres = float(args.crop_metres) if args.crop_metres > 0 else parse_float(row.get("crop_metres"), 38.0)
            crop_size = parse_int(row.get("crop_size"), 64)
            cell_epsg = parse_int(row.get("crown_epsg"), epsg_from_cell_id(row.get("cell_id")))
            cell_id = str(row.get("cell_id") or (f"epsg{cell_epsg}" if cell_epsg > 0 else ""))
            rows.append(
                TreeCropRow(
                    position=position,
                    tree_id=parse_int(row.get("tree_id"), -1),
                    tree_centered_index=parse_int(row.get("tree_centered_index"), parse_int(row.get("crop_index"), position)),
                    crop_index=parse_int(row.get("crop_index"), position),
                    row_index=parse_int(row.get("reduced_id"), parse_int(row.get("row_index"), position)),
                    cell_id=cell_id,
                    cell_epsg=cell_epsg,
                    center_x=parse_float(row.get("peak_x_utm"), parse_float(row.get("crown_x_utm"))),
                    center_y=parse_float(row.get("peak_y_utm"), parse_float(row.get("crown_y_utm"))),
                    crop_metres=crop_metres,
                    crop_size=crop_size,
                    source_file=str(row.get("source_file") or ""),
                    source_row=parse_int(row.get("source_row"), -1),
                    taxon_label=str(row.get("taxon_label") or ""),
                    crop_failed=crop_failed,
                )
            )
    if args.max_records > 0:
        rows = rows[: int(args.max_records)]
    rows = [row for row in rows if math.isfinite(row.center_x) and math.isfinite(row.center_y)]
    return rows


def output_pixels(rows: list[TreeCropRow], args: argparse.Namespace) -> int:
    if args.pixels > 0:
        return int(args.pixels)
    crop_metres = rows[0].crop_metres if rows else (args.crop_metres or 38.0)
    pixels = int(round(float(crop_metres) / float(args.pixel_size)))
    if pixels <= 0:
        raise SystemExit("Output pixels must be positive. Check --crop-metres, --pixel-size, or --pixels.")
    return pixels


def target_epsg_for_rows(rows: list[TreeCropRow], args: argparse.Namespace) -> int:
    if args.target_epsg:
        return int(args.target_epsg)
    counts = Counter(row.cell_epsg for row in rows if row.cell_epsg > 0)
    if not counts:
        raise RuntimeError("Could not infer target EPSG from metadata cell_id; pass --target-epsg.")
    return int(counts.most_common(1)[0][0])


def transformed_tree_centers(rows: list[TreeCropRow], target_epsg: int):
    import numpy as np
    import pyproj

    centers_x = np.empty(len(rows), dtype=np.float64)
    centers_y = np.empty(len(rows), dtype=np.float64)
    for epsg in sorted({row.cell_epsg for row in rows}):
        idx = np.asarray([i for i, row in enumerate(rows) if row.cell_epsg == epsg], dtype=np.int64)
        xs = np.asarray([rows[i].center_x for i in idx], dtype=np.float64)
        ys = np.asarray([rows[i].center_y for i in idx], dtype=np.float64)
        if epsg == target_epsg or epsg <= 0:
            centers_x[idx] = xs
            centers_y[idx] = ys
        else:
            transformer = pyproj.Transformer.from_crs(pyproj.CRS.from_epsg(epsg), pyproj.CRS.from_epsg(target_epsg), always_xy=True)
            tx, ty = transformer.transform(xs, ys)
            centers_x[idx] = tx
            centers_y[idx] = ty
    return centers_x, centers_y


def build_jobs(args: argparse.Namespace) -> list[TreeCityJob]:
    filters = {normalize_token(value) for value in args.city_token}
    excluded = {normalize_token(value) for value in args.exclude_city_token}
    metadata_by_city = discover_metadata(args.crop_root, args.metadata_pattern)
    summary_rows = lidar_utils.eligible_summary_rows(lidar_utils.read_csv(args.city_summary), filters)
    manifest_by_city = lidar_utils.manifest_rows_by_city(lidar_utils.read_csv(args.manifest))
    jobs: list[TreeCityJob] = []
    for row in summary_rows:
        city_token = str(row.get("city_token") or "")
        city_norm = normalize_token(city_token)
        city_code = str(row.get("city_code") or "").upper()
        if not city_token or not city_code or city_norm in excluded:
            continue
        metadata_path = metadata_by_city.get(city_norm)
        if metadata_path is None:
            continue
        city_rows = manifest_by_city.get(city_code, [])
        if not city_rows:
            continue
        by_project: dict[str, list[dict[str, str]]] = {}
        for manifest_row in city_rows:
            project = str(manifest_row.get("project") or "unknown").strip() or "unknown"
            by_project.setdefault(project, []).append(manifest_row)
        ordered_projects = lidar_utils.project_preference_order(row, set(by_project))
        for preference_rank, project in enumerate(ordered_projects, start=1):
            jobs.append(
                TreeCityJob(
                    token=city_norm,
                    name=str(row.get("city_name") or city_token),
                    code=city_code,
                    project=project,
                    preference_rank=preference_rank,
                    metadata_path=metadata_path,
                    manifest_rows=by_project[project],
                )
            )
    if args.max_cities:
        city_order: list[str] = []
        for job in jobs:
            if job.token not in city_order:
                city_order.append(job.token)
        keep = set(city_order[: int(args.max_cities)])
        jobs = [job for job in jobs if job.token in keep]
    return jobs


def tile_paths(job: TreeCityJob, args: argparse.Namespace) -> list[Path]:
    paths = []
    for row in job.manifest_rows:
        relative = row.get("relative_path") or str(Path(job.code) / row.get("filename", ""))
        manifest_path = Path(row.get("local_path") or "")
        path = manifest_path if args.use_manifest_local_path and manifest_path.exists() else args.lidar_root / relative
        if lidar_utils.complete_lidar_file(path, row):
            paths.append(path)
    unique = {path.name.lower(): path for path in paths}
    result = sorted(unique.values(), key=lambda path: path.name.lower())
    return result[: args.max_tiles] if args.max_tiles else result


def safe_project_token(project: str) -> str:
    return lidar_utils.safe_project_token(project)


def product_stem(job: TreeCityJob) -> str:
    name = job.metadata_path.name.replace("_metadata.csv", "")
    return name if name else f"{job.token}_tree_centered_lidar"


def city_output_paths(job: TreeCityJob, args: argparse.Namespace) -> dict[str, Path]:
    city = job.token
    stem = product_stem(job)
    project_token = safe_project_token(job.project)
    candidate_root = args.candidate_root / city / project_token
    bin_root = candidate_root / "Binned"
    return {
        "candidate_root": candidate_root,
        "bin_dir": bin_root,
        "dtm_dir": args.dtm_root / city,
        "dsm_dir": args.dsm_root / city,
        "chm_dir": args.chm_root / city,
        "count_dir": args.count_root / city,
        "ground_min": bin_root / f"{stem}_ground_min.npy",
        "surface_max": bin_root / f"{stem}_surface_max.npy",
        "ground_count": bin_root / f"{stem}_ground_count.npy",
        "surface_count": bin_root / f"{stem}_surface_count.npy",
        "all_count": bin_root / f"{stem}_all_count.npy",
        "bin_state": bin_root / f"{stem}_bin_state.json",
        "bin_marker": bin_root / f"{stem}_bin_complete.json",
        "dtm": args.dtm_root / city / f"{stem}_dtm.npy",
        "dsm": args.dsm_root / city / f"{stem}_dsm.npy",
        "chm": args.chm_root / city / f"{stem}_chm.npy",
        "ground_count_out": args.count_root / city / f"{stem}_ground_count.npy",
        "surface_count_out": args.count_root / city / f"{stem}_surface_count.npy",
        "all_count_out": args.count_root / city / f"{stem}_all_count.npy",
        "lidar_index": args.chm_root / city / f"{stem}_lidar_index.csv",
    }


def open_or_create_memmaps(paths: dict[str, Path], shape: tuple[int, int, int], overwrite: bool):
    return lidar_utils.open_or_create_memmaps(paths, shape, overwrite)


def close_memmap(array) -> None:
    lidar_utils.close_memmap(array)


def load_bin_state(path: Path, expected_shape: tuple[int, int, int], allow_active_tile: bool = False) -> dict:
    if not path.exists():
        return {"processed_tiles": [], "active_tile": "", "shape": list(expected_shape)}
    state = json.loads(path.read_text(encoding="utf-8"))
    if tuple(state.get("shape", ())) != expected_shape:
        raise RuntimeError(f"Bin shape changed for {path}; rerun with --overwrite")
    active_tile = state.get("active_tile") or ""
    if active_tile and not allow_active_tile:
        raise RuntimeError(
            f"Previous run stopped while processing {active_tile}; rerun with --overwrite to avoid "
            "double-counting, or use --retry-active-tile only after checking that partial updates "
            "were not flushed."
        )
    return state


def write_bin_state(path: Path, state: dict) -> None:
    lidar_utils.write_bin_state(path, state)


def transform_xy_arrays(x, y, source_crs, target_epsg: int):
    import numpy as np
    import pyproj

    if source_crs is None:
        return np.asarray(x), np.asarray(y), "untransformed"
    target_crs = pyproj.CRS.from_epsg(target_epsg)
    source_crs = pyproj.CRS.from_user_input(source_crs)
    if source_crs == target_crs:
        return np.asarray(x), np.asarray(y), source_crs.to_string()
    transformer = pyproj.Transformer.from_crs(source_crs, target_crs, always_xy=True)
    tx, ty = transformer.transform(x, y)
    return np.asarray(tx), np.asarray(ty), f"{source_crs.to_string()}->{target_crs.to_string()}"


def tile_source_crs(reader, fallback_source_epsg: int):
    import pyproj

    try:
        source_crs = reader.header.parse_crs()
    except Exception:
        source_crs = None
    if source_crs is None and fallback_source_epsg > 0:
        source_crs = pyproj.CRS.from_epsg(fallback_source_epsg)
    return source_crs


def update_bins_for_points(
    *,
    x,
    y,
    z,
    classification,
    centers_x,
    centers_y,
    half: float,
    cell_size: float,
    pixels: int,
    dtm_codes,
    dsm_codes,
    ground_min,
    surface_max,
    ground_count,
    surface_count,
    all_count,
) -> tuple[int, int]:
    import numpy as np

    if len(x) == 0:
        return 0, 0
    xmin = float(np.nanmin(x))
    xmax = float(np.nanmax(x))
    ymin = float(np.nanmin(y))
    ymax = float(np.nanmax(y))
    candidates = np.flatnonzero(
        (centers_x >= xmin - half)
        & (centers_x <= xmax + half)
        & (centers_y >= ymin - half)
        & (centers_y <= ymax + half)
    )
    point_updates = 0
    dtm_code_list = list(dtm_codes)
    dsm_code_list = list(dsm_codes)
    for crop_pos in candidates:
        chip_xmin = centers_x[crop_pos] - half
        chip_xmax = centers_x[crop_pos] + half
        chip_ymin = centers_y[crop_pos] - half
        chip_ymax = centers_y[crop_pos] + half
        in_chip = (x >= chip_xmin) & (x < chip_xmax) & (y >= chip_ymin) & (y < chip_ymax)
        if not np.any(in_chip):
            continue
        cols = np.floor((x[in_chip] - chip_xmin) / cell_size).astype(np.int32)
        out_rows = np.floor((chip_ymax - y[in_chip]) / cell_size).astype(np.int32)
        valid = (out_rows >= 0) & (out_rows < pixels) & (cols >= 0) & (cols < pixels)
        if not np.any(valid):
            continue
        cols = cols[valid]
        out_rows = out_rows[valid]
        chip_z = z[in_chip][valid]
        chip_class = classification[in_chip][valid]
        np.add.at(all_count[crop_pos], (out_rows, cols), 1)
        ground_mask = np.isin(chip_class, dtm_code_list)
        if np.any(ground_mask):
            np.minimum.at(ground_min[crop_pos], (out_rows[ground_mask], cols[ground_mask]), chip_z[ground_mask])
            np.add.at(ground_count[crop_pos], (out_rows[ground_mask], cols[ground_mask]), 1)
        surface_mask = np.isin(chip_class, dsm_code_list)
        if np.any(surface_mask):
            np.maximum.at(surface_max[crop_pos], (out_rows[surface_mask], cols[surface_mask]), chip_z[surface_mask])
            np.add.at(surface_count[crop_pos], (out_rows[surface_mask], cols[surface_mask]), 1)
        point_updates += int(np.count_nonzero(valid))
    return len(candidates), point_updates


def infer_z_scale_for_job(job: TreeCityJob, args: argparse.Namespace) -> tuple[float, str, str, str]:
    return lidar_utils.infer_z_scale_for_job(job, args, tile_paths)


def bin_city(job: TreeCityJob, args: argparse.Namespace) -> dict[str, object]:
    import laspy
    import numpy as np

    rows = read_tree_rows(job.metadata_path, args)
    pixels = output_pixels(rows, args)
    shape = (len(rows), pixels, pixels)
    paths = city_output_paths(job, args)
    tiles = tile_paths(job, args)
    if not rows:
        return {"status": "skipped", "reason": "no_tree_rows", "chip_rows": 0, "tile_count": len(tiles), **paths}
    if not tiles:
        return {"status": "skipped", "reason": "no_complete_lidar_tiles", "chip_rows": len(rows), "tile_count": 0, **paths}
    if paths["bin_marker"].exists() and not args.overwrite:
        return {"status": "skipped", "reason": "bin_already_complete", "chip_rows": len(rows), "tile_count": len(tiles), **paths}

    target_epsg = target_epsg_for_rows(rows, args)
    centers_x, centers_y = transformed_tree_centers(rows, target_epsg)
    crop_metres = float(args.crop_metres) if args.crop_metres > 0 else float(rows[0].crop_metres)
    half = crop_metres / 2.0
    cell_size = crop_metres / pixels
    dtm_codes = lidar_utils.class_code_set(args.dtm_class_codes)
    dsm_codes = lidar_utils.class_code_set(args.dsm_class_codes)

    ground_min, surface_max, ground_count, surface_count, all_count = open_or_create_memmaps(paths, shape, args.overwrite)
    state = load_bin_state(paths["bin_state"], shape, allow_active_tile=args.retry_active_tile)
    processed = set(state.get("processed_tiles", []))
    active_tile = state.get("active_tile") or ""
    if active_tile:
        print(
            f"{job.token}: retrying active tile from previous run: {active_tile}. "
            "This assumes partial updates were not flushed.",
            flush=True,
        )
    started = time.time()

    print(
        f"{job.token}: binning {len(tiles):,} tile(s), {len(rows):,} tree-centered crop(s), "
        f"shape={shape}, crop_metres={crop_metres:g}, target_epsg={target_epsg}",
        flush=True,
    )
    for tile_number, tile_path in enumerate(tiles, start=1):
        if tile_path.name in processed:
            continue
        if active_tile and tile_path.name != active_tile:
            continue
        state["active_tile"] = tile_path.name
        write_bin_state(paths["bin_state"], state)
        tile_started = time.time()

        tile_point_updates = 0
        max_candidates = 0
        crs_text = ""
        if args.tile_point_chunk_size and args.tile_point_chunk_size > 0:
            with laspy.open(str(tile_path)) as reader:
                source_crs = tile_source_crs(reader, args.laspy_source_epsg)
                total_points = int(reader.header.point_count)
                for chunk_number, chunk in enumerate(reader.chunk_iterator(args.tile_point_chunk_size), start=1):
                    x, y, crs_text = transform_xy_arrays(
                        np.asarray(chunk.x),
                        np.asarray(chunk.y),
                        source_crs,
                        target_epsg,
                    )
                    z = np.asarray(chunk.z, dtype=np.float32)
                    classification = np.asarray(chunk.classification)
                    candidates_n, updates_n = update_bins_for_points(
                        x=x,
                        y=y,
                        z=z,
                        classification=classification,
                        centers_x=centers_x,
                        centers_y=centers_y,
                        half=half,
                        cell_size=cell_size,
                        pixels=pixels,
                        dtm_codes=dtm_codes,
                        dsm_codes=dsm_codes,
                        ground_min=ground_min,
                        surface_max=surface_max,
                        ground_count=ground_count,
                        surface_count=surface_count,
                        all_count=all_count,
                    )
                    max_candidates = max(max_candidates, candidates_n)
                    tile_point_updates += updates_n
                    if chunk_number % max(1, args.tile_chunk_progress_interval) == 0:
                        processed_points = min(chunk_number * args.tile_point_chunk_size, total_points)
                        print(
                            f"    {tile_path.name}: chunk={chunk_number:,}; "
                            f"points={processed_points:,}/{total_points:,}; "
                            f"point_updates={tile_point_updates:,}",
                            flush=True,
                        )
                    del chunk, x, y, z, classification
        else:
            data = laspy.read(str(tile_path))
            x, y, crs_text = lidar_utils.transform_points(data, target_epsg, args.laspy_source_epsg)
            z = np.asarray(data.z, dtype=np.float32)
            classification = np.asarray(data.classification)
            max_candidates, tile_point_updates = update_bins_for_points(
                x=x,
                y=y,
                z=z,
                classification=classification,
                centers_x=centers_x,
                centers_y=centers_y,
                half=half,
                cell_size=cell_size,
                pixels=pixels,
                dtm_codes=dtm_codes,
                dsm_codes=dsm_codes,
                ground_min=ground_min,
                surface_max=surface_max,
                ground_count=ground_count,
                surface_count=surface_count,
                all_count=all_count,
            )
            del data, x, y, z, classification

        for array in (ground_min, surface_max, ground_count, surface_count, all_count):
            array.flush()
        processed.add(tile_path.name)
        state["processed_tiles"] = sorted(processed)
        state["active_tile"] = ""
        write_bin_state(paths["bin_state"], state)
        elapsed = time.time() - tile_started
        if tile_number % max(1, args.checkpoint_every_tiles) == 0:
            print(
                f"  {tile_number:,}/{len(tiles):,} {tile_path.name}: "
                f"candidates={max_candidates:,}, point_updates={tile_point_updates:,}, crs={crs_text}, {elapsed:.1f}s",
                flush=True,
            )
        gc.collect()
        active_tile = ""

    for array in (ground_min, surface_max, ground_count, surface_count, all_count):
        close_memmap(array)
    marker = {
        "city": job.token,
        "city_code": job.code,
        "project": job.project,
        "preference_rank": job.preference_rank,
        "metadata_path": str(job.metadata_path),
        "shape": list(shape),
        "crop_metres": crop_metres,
        "pixel_size": cell_size,
        "target_epsg": target_epsg,
        "tile_count": len(tiles),
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "elapsed_seconds": round(time.time() - started, 3),
    }
    paths["bin_marker"].write_text(json.dumps(marker, indent=2), encoding="utf-8")
    return {"status": "complete", "reason": "bin_complete", "chip_rows": len(rows), "tile_count": len(tiles), **paths}


def write_lidar_index(path: Path, rows: list[TreeCropRow], dtm, dsm, chm, selections: list[dict[str, object]] | None = None) -> None:
    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "tree_id",
        "tree_centered_index",
        "crop_index",
        "row_index",
        "cell_id",
        "cell_epsg",
        "peak_x_utm",
        "peak_y_utm",
        "crop_metres",
        "crop_size",
        "source_file",
        "source_row",
        "taxon_label",
        "crop_failed",
        "dtm_valid_fraction",
        "dsm_valid_fraction",
        "chm_valid_fraction",
        "chm_mean_m",
        "chm_max_m",
        "chm_p95_m",
        "selected_project",
        "selected_preference_rank",
        "selected_score",
        "selected_reason",
        "candidate_count",
        "z_scale",
        "z_units",
        "z_scale_confidence",
        "z_scale_reason",
    ]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for position, row in enumerate(rows):
            chm_chip = chm[position]
            if np.isfinite(chm_chip).any():
                chm_mean = float(np.nanmean(chm_chip))
                chm_max = float(np.nanmax(chm_chip))
                chm_p95 = float(np.nanpercentile(chm_chip, 95))
            else:
                chm_mean = chm_max = chm_p95 = math.nan
            writer.writerow(
                {
                    "tree_centered_index": row.tree_centered_index,
                    "tree_id": row.tree_id,
                    "crop_index": row.crop_index,
                    "row_index": row.row_index,
                    "cell_id": row.cell_id,
                    "cell_epsg": row.cell_epsg,
                    "peak_x_utm": row.center_x,
                    "peak_y_utm": row.center_y,
                    "crop_metres": row.crop_metres,
                    "crop_size": row.crop_size,
                    "source_file": row.source_file,
                    "source_row": row.source_row,
                    "taxon_label": row.taxon_label,
                    "crop_failed": row.crop_failed,
                    "dtm_valid_fraction": float(np.isfinite(dtm[position]).mean()),
                    "dsm_valid_fraction": float(np.isfinite(dsm[position]).mean()),
                    "chm_valid_fraction": float(np.isfinite(chm_chip).mean()),
                    "chm_mean_m": chm_mean,
                    "chm_max_m": chm_max,
                    "chm_p95_m": chm_p95,
                    "selected_project": selections[position].get("project", "") if selections else "",
                    "selected_preference_rank": selections[position].get("preference_rank", "") if selections else "",
                    "selected_score": selections[position].get("score", "") if selections else "",
                    "selected_reason": selections[position].get("reason", "") if selections else "",
                    "candidate_count": selections[position].get("candidate_count", "") if selections else "",
                    "z_scale": selections[position].get("z_scale", "") if selections else "",
                    "z_units": selections[position].get("z_units", "") if selections else "",
                    "z_scale_confidence": selections[position].get("z_scale_confidence", "") if selections else "",
                    "z_scale_reason": selections[position].get("z_scale_reason", "") if selections else "",
                }
            )


def derive_candidate_arrays(job: TreeCityJob, args: argparse.Namespace, shape: tuple[int, int, int]):
    import numpy as np

    paths = city_output_paths(job, args)
    if not paths["bin_marker"].exists():
        return None
    for key in ("ground_min", "surface_max", "ground_count", "surface_count", "all_count"):
        if not paths[key].exists():
            return None

    ground_min = np.load(paths["ground_min"], mmap_mode="r")
    surface_max = np.load(paths["surface_max"], mmap_mode="r")
    if tuple(ground_min.shape) != shape or tuple(surface_max.shape) != shape:
        raise RuntimeError(f"Bin shape does not match metadata for {job.token} {job.project}; rerun bin with --overwrite")
    z_scale, z_units, z_confidence, z_reason = infer_z_scale_for_job(job, args)
    dtm = np.where(np.isfinite(ground_min), ground_min * z_scale, np.nan).astype(np.float32)
    dsm = np.where(np.isfinite(surface_max), surface_max * z_scale, np.nan).astype(np.float32)
    chm = np.where(np.isfinite(dtm) & np.isfinite(dsm), np.maximum(dsm - dtm, 0.0), np.nan).astype(np.float32)
    return {
        "job": job,
        "paths": paths,
        "dtm": dtm,
        "dsm": dsm,
        "chm": chm,
        "ground_count": np.load(paths["ground_count"], mmap_mode="r"),
        "surface_count": np.load(paths["surface_count"], mmap_mode="r"),
        "all_count": np.load(paths["all_count"], mmap_mode="r"),
        "coverage": np.isfinite(chm).mean(axis=(1, 2)),
        "z_scale": z_scale,
        "z_units": z_units,
        "z_scale_confidence": z_confidence,
        "z_scale_reason": z_reason,
    }


def derive_city_from_candidates(city_jobs: list[TreeCityJob], args: argparse.Namespace) -> dict[str, object]:
    import numpy as np

    primary = sorted(city_jobs, key=lambda job: job.preference_rank)[0]
    rows = read_tree_rows(primary.metadata_path, args)
    pixels = output_pixels(rows, args)
    shape = (len(rows), pixels, pixels)
    paths = city_output_paths(primary, args)
    if paths["dtm"].exists() and paths["dsm"].exists() and paths["chm"].exists() and paths["lidar_index"].exists() and not args.overwrite:
        return {"status": "skipped", "reason": "products_already_complete", "chip_rows": len(rows), "tile_count": "", **paths}

    candidates = []
    for job in sorted(city_jobs, key=lambda item: item.preference_rank):
        candidate = derive_candidate_arrays(job, args, shape)
        if candidate is not None:
            candidates.append(candidate)
    if not candidates:
        return {"status": "skipped", "reason": "no_complete_candidate_bins", "chip_rows": len(rows), "tile_count": "", **paths}

    for key in ("dtm_dir", "dsm_dir", "chm_dir", "count_dir"):
        paths[key].mkdir(parents=True, exist_ok=True)

    final_dtm = np.full(shape, np.nan, dtype=np.float32)
    final_dsm = np.full(shape, np.nan, dtype=np.float32)
    final_chm = np.full(shape, np.nan, dtype=np.float32)
    final_ground_count = np.zeros(shape, dtype=np.uint32)
    final_surface_count = np.zeros(shape, dtype=np.uint32)
    final_all_count = np.zeros(shape, dtype=np.uint32)
    selections: list[dict[str, object]] = []
    selected_counts: Counter[str] = Counter()
    selected_scale_counts: Counter[tuple[float, str, str, str]] = Counter()

    primary_candidate = next((candidate for candidate in candidates if candidate["job"].preference_rank == 1), None)
    for crop_pos in range(shape[0]):
        selected = None
        reason = ""
        if primary_candidate is not None and float(primary_candidate["coverage"][crop_pos]) >= args.auto_accept_coverage:
            selected = primary_candidate
            reason = f"auto_accept_rank1_chm_valid_ge_{args.auto_accept_coverage:g}"
            score = float(primary_candidate["coverage"][crop_pos])
        else:
            scored = []
            for candidate in candidates:
                job = candidate["job"]
                coverage = float(candidate["coverage"][crop_pos])
                score_value = coverage - args.preference_penalty * (int(job.preference_rank) - 1)
                scored.append((score_value, coverage, -int(job.preference_rank), candidate))
            score, _, _, selected = max(scored, key=lambda item: item[:3])
            reason = "score_coverage_minus_preference_penalty"
        job = selected["job"]
        final_dtm[crop_pos] = selected["dtm"][crop_pos]
        final_dsm[crop_pos] = selected["dsm"][crop_pos]
        final_chm[crop_pos] = selected["chm"][crop_pos]
        final_ground_count[crop_pos] = selected["ground_count"][crop_pos]
        final_surface_count[crop_pos] = selected["surface_count"][crop_pos]
        final_all_count[crop_pos] = selected["all_count"][crop_pos]
        selected_counts[job.project] += 1
        selected_scale_key = (
            float(selected["z_scale"]),
            str(selected["z_units"]),
            str(selected["z_scale_confidence"]),
            str(selected["z_scale_reason"]),
        )
        selected_scale_counts[selected_scale_key] += 1
        selections.append(
            {
                "project": job.project,
                "preference_rank": job.preference_rank,
                "score": round(float(score), 6),
                "reason": reason,
                "candidate_count": len(candidates),
                "z_scale": selected["z_scale"],
                "z_units": selected["z_units"],
                "z_scale_confidence": selected["z_scale_confidence"],
                "z_scale_reason": selected["z_scale_reason"],
            }
        )

    if not args.allow_empty_products and not np.isfinite(final_chm).any():
        return {"status": "failed", "reason": "all_empty_chm", "chip_rows": len(rows), "tile_count": "", **paths}
    np.save(paths["dtm"], final_dtm)
    np.save(paths["dsm"], final_dsm)
    np.save(paths["chm"], final_chm)
    np.save(paths["ground_count_out"], final_ground_count)
    np.save(paths["surface_count_out"], final_surface_count)
    np.save(paths["all_count_out"], final_all_count)
    write_lidar_index(paths["lidar_index"], rows, final_dtm, final_dsm, final_chm, selections)
    metadata_path = paths["chm"].with_name(paths["chm"].name.replace("_chm.npy", "_lidar_product_metadata.json"))
    selected_scale_summary = [
        {
            "z_scale": scale,
            "z_units": units,
            "z_scale_confidence": confidence,
            "z_scale_reason": reason,
            "crop_count": count,
        }
        for (scale, units, confidence, reason), count in sorted(selected_scale_counts.items())
    ]
    if len(selected_scale_counts) == 1:
        (metadata_z_scale, metadata_z_units, metadata_z_confidence, metadata_z_reason), _ = next(iter(selected_scale_counts.items()))
    else:
        metadata_z_scale = "mixed"
        metadata_z_units = "mixed"
        metadata_z_confidence = "mixed"
        metadata_z_reason = "mixed_selected_products"
    metadata_path.write_text(
        json.dumps(
            {
                "city": primary.token,
                "city_code": primary.code,
                "stem": product_stem(primary),
                "metadata_path": str(primary.metadata_path),
                "crop_metres": float(args.crop_metres) if args.crop_metres > 0 else float(rows[0].crop_metres if rows else 0),
                "pixels": int(pixels),
                "pixel_size": float((float(args.crop_metres) if args.crop_metres > 0 else float(rows[0].crop_metres)) / pixels) if rows else None,
                "z_scale": metadata_z_scale,
                "z_units": metadata_z_units,
                "z_scale_confidence": metadata_z_confidence,
                "z_scale_reason": metadata_z_reason,
                "auto_accept_coverage": args.auto_accept_coverage,
                "preference_penalty": args.preference_penalty,
                "candidate_projects": [
                    {
                        "project": candidate["job"].project,
                        "preference_rank": candidate["job"].preference_rank,
                        "z_scale": candidate["z_scale"],
                        "z_units": candidate["z_units"],
                        "z_scale_confidence": candidate["z_scale_confidence"],
                        "z_scale_reason": candidate["z_scale_reason"],
                    }
                    for candidate in candidates
                ],
                "selected_counts": dict(sorted(selected_counts.items())),
                "selected_z_scale_counts": selected_scale_summary,
                "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    selection_summary = ";".join(f"{project}:{count}" for project, count in sorted(selected_counts.items()))
    return {
        "status": "complete",
        "reason": f"derive_complete;selected={selection_summary}",
        "chip_rows": len(rows),
        "tile_count": "",
        **paths,
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    lidar_utils.write_csv(path, rows)


def result_record(job: TreeCityJob, stage: str, result: dict[str, object], started_at: str) -> dict[str, object]:
    return {
        "city_token": job.token,
        "city_name": job.name,
        "city_code": job.code,
        "candidate_project": job.project,
        "preference_rank": job.preference_rank,
        "stage": stage,
        "status": result.get("status", ""),
        "reason": result.get("reason", ""),
        "tree_crop_rows": result.get("chip_rows", ""),
        "manifest_rows": len(job.manifest_rows),
        "tile_count": result.get("tile_count", ""),
        "started_at": started_at,
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "bin_marker": str(result.get("bin_marker", "")),
        "dtm_path": str(result.get("dtm", "")),
        "dsm_path": str(result.get("dsm", "")),
        "chm_path": str(result.get("chm", "")),
        "lidar_index_path": str(result.get("lidar_index", "")),
    }


def derive_city_group_worker(city_jobs: list[TreeCityJob], args: argparse.Namespace) -> tuple[TreeCityJob, str, dict[str, object]]:
    primary = sorted(city_jobs, key=lambda job: job.preference_rank)[0]
    started = time.strftime("%Y-%m-%dT%H:%M:%S")
    try:
        result = derive_city_from_candidates(city_jobs, args)
    except Exception as exc:
        result = {"status": "failed", "reason": str(exc), **city_output_paths(primary, args)}
    return primary, started, result


def bin_city_worker(job: TreeCityJob, args: argparse.Namespace) -> tuple[TreeCityJob, str, dict[str, object]]:
    started = time.strftime("%Y-%m-%dT%H:%M:%S")
    try:
        result = bin_city(job, args)
    except Exception as exc:
        result = {"status": "failed", "reason": str(exc), **city_output_paths(job, args)}
    return job, started, result


def main() -> int:
    args = parse_args()
    ensure_roots(args)
    args.z_scale_rows = lidar_utils.load_z_scale_table(args.z_scale_table)
    jobs = build_jobs(args)
    city_groups: dict[tuple[str, str], list[TreeCityJob]] = {}
    for job in jobs:
        city_groups.setdefault((job.code, job.token), []).append(job)
    print(f"Eligible candidate job(s): {len(jobs):,}; city/cities={len(city_groups):,}", flush=True)
    if args.dry_run:
        for job in jobs:
            rows = read_tree_rows(job.metadata_path, args)
            tiles = tile_paths(job, args)
            target_epsg = target_epsg_for_rows(rows, args) if rows else ""
            pixels = output_pixels(rows, args) if rows else ""
            crop_metres = (args.crop_metres if args.crop_metres > 0 else rows[0].crop_metres) if rows else ""
            print(
                f"  {job.token} ({job.code}) project={job.project} rank={job.preference_rank}: "
                f"tree_crops={len(rows):,}, tiles={len(tiles):,}, target_epsg={target_epsg}, "
                f"crop_metres={crop_metres}, pixels={pixels}, "
                f"bin_dir={city_output_paths(job, args)['bin_dir']}",
                flush=True,
            )
        return 0
    if args.stage in {"all", "bin"}:
        try:
            import laspy  # noqa: F401
            import pyproj  # noqa: F401
        except ImportError as error:
            raise SystemExit(
                "The bin stage requires laspy and pyproj. Use the same environment as the original "
                "LiDAR product pipeline, or install: python -m pip install laspy lazrs pyproj"
            ) from error

    run_records: list[dict[str, object]] = []
    failures = 0
    if args.stage in {"all", "bin"}:
        bin_workers = max(1, int(args.bin_workers))
        if bin_workers == 1:
            for position, job in enumerate(jobs, start=1):
                print(
                    f"\nBIN [{position:,}/{len(jobs):,}] {job.token} ({job.code}) "
                    f"project={job.project} rank={job.preference_rank}",
                    flush=True,
                )
                job, started, result = bin_city_worker(job, args)
                print(f"  bin: {result.get('status')} {result.get('reason')}", flush=True)
                if result.get("status") == "failed":
                    failures += 1
                run_records.append(result_record(job, "bin", result, started))
                write_csv(args.run_summary, run_records)
        else:
            print(f"\nBIN: processing {len(jobs):,} candidate job(s) with bin_workers={bin_workers}", flush=True)
            with ProcessPoolExecutor(max_workers=bin_workers) as executor:
                future_to_job = {
                    executor.submit(bin_city_worker, job, args): (position, job)
                    for position, job in enumerate(jobs, start=1)
                }
                completed = 0
                for future in as_completed(future_to_job):
                    position, submitted_job = future_to_job[future]
                    completed += 1
                    try:
                        job, started, result = future.result()
                    except Exception as exc:
                        job = submitted_job
                        started = time.strftime("%Y-%m-%dT%H:%M:%S")
                        result = {"status": "failed", "reason": str(exc), **city_output_paths(job, args)}
                    print(
                        f"  BIN done [{completed:,}/{len(jobs):,}] {job.token} ({job.code}) "
                        f"project={job.project} rank={job.preference_rank}: "
                        f"{result.get('status')} {result.get('reason')}",
                        flush=True,
                    )
                    if result.get("status") == "failed":
                        failures += 1
                    run_records.append(result_record(job, "bin", result, started))
                    write_csv(args.run_summary, run_records)
    if args.stage in {"all", "derive"}:
        derive_items = list(city_groups.items())
        derive_workers = max(1, int(args.derive_workers))
        if derive_workers == 1:
            for position, ((city_code, city_token), city_jobs) in enumerate(derive_items, start=1):
                print(f"\nDERIVE [{position:,}/{len(derive_items):,}] {city_token} ({city_code}) candidates={len(city_jobs):,}", flush=True)
                primary, started, result = derive_city_group_worker(city_jobs, args)
                print(f"  derive: {result.get('status')} {result.get('reason')}", flush=True)
                if result.get("status") == "failed":
                    failures += 1
                run_records.append(result_record(primary, "derive", result, started))
                write_csv(args.run_summary, run_records)
        else:
            print(f"\nDERIVE: processing {len(derive_items):,} city/cities with derive_workers={derive_workers}", flush=True)
            with ProcessPoolExecutor(max_workers=derive_workers) as executor:
                future_to_city = {
                    executor.submit(derive_city_group_worker, city_jobs, args): (position, city_code, city_token, len(city_jobs))
                    for position, ((city_code, city_token), city_jobs) in enumerate(derive_items, start=1)
                }
                completed = 0
                for future in as_completed(future_to_city):
                    position, city_code, city_token, candidate_count = future_to_city[future]
                    completed += 1
                    try:
                        primary, started, result = future.result()
                    except Exception as exc:
                        primary = sorted(city_groups[(city_code, city_token)], key=lambda job: job.preference_rank)[0]
                        started = time.strftime("%Y-%m-%dT%H:%M:%S")
                        result = {"status": "failed", "reason": str(exc), **city_output_paths(primary, args)}
                    print(
                        f"  DERIVE done [{completed:,}/{len(derive_items):,}] "
                        f"{city_token} ({city_code}) candidates={candidate_count:,}: "
                        f"{result.get('status')} {result.get('reason')}",
                        flush=True,
                    )
                    if result.get("status") == "failed":
                        failures += 1
                    run_records.append(result_record(primary, "derive", result, started))
                    write_csv(args.run_summary, run_records)

    print(f"\nWrote {args.run_summary}", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
