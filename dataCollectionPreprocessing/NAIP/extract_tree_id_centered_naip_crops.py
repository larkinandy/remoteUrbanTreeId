#!/usr/bin/env python3
"""Extract clean tree-centered NAIP crops keyed by tree_id.

This rebuild script does not use the old tree-centered metadata IDs. The input
is the clean tree-to-nearest-crown join table, and the output metadata is keyed
by tree_id.

Run with the ArcGIS Pro Python environment because source SID reads require
ArcPy.
"""

from __future__ import annotations

import argparse
import concurrent.futures
from collections import defaultdict
import copy
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import tree_centered_naip_utils as naip_utils


DEFAULT_JOIN_CSV = Path(
    r"H:\TreeCenteredModelInputs\tree_to_detected_crowns_clean\losangeles\losangeles_tree_to_nearest_detected_crown_5m.csv"
)
DEFAULT_NAIP_DIR = Path(r"E:\NAIP_PAIRED")
DEFAULT_OUTPUT_DIR = Path(r"H:\TreeCenteredModelInputs\tree_centered_naip_crops_clean")
DEFAULT_JOIN_ROOT = Path(r"H:\TreeCenteredModelInputs\tree_to_detected_crowns_clean")
DEFAULT_COUNTY_MANIFEST = HERE / "naip_county_manifest.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--join-csv", type=Path, default=DEFAULT_JOIN_CSV)
    parser.add_argument("--join-root", type=Path, default=DEFAULT_JOIN_ROOT)
    parser.add_argument("--join-pattern", default="{city}_tree_to_nearest_detected_crown_5m.csv")
    parser.add_argument("--paired-index-csv", type=Path, default=None, help="Optional legacy paired raster index.")
    parser.add_argument("--naip-dir", type=Path, default=DEFAULT_NAIP_DIR, help="Root containing downloaded county ZIPs/SIDs by city code.")
    parser.add_argument("--county-manifest", type=Path, default=DEFAULT_COUNTY_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--city-token", action="append", default=None, help="City token(s) to process. Repeatable.")
    parser.add_argument("--exclude-city-token", action="append", default=[], help="City token(s) to skip. Repeatable.")
    parser.add_argument(
        "--job-csv",
        type=Path,
        default=None,
        help="Optional CSV with city_token,join_csv and optional source_epsg,output_prefix,paired_index_csv columns.",
    )
    parser.add_argument("--output-prefix", default=None)
    parser.add_argument("--crop-size", type=int, default=64)
    parser.add_argument("--crop-metres", type=float, default=38.0)
    parser.add_argument("--source-epsg", default="", help="Override crown CRS; otherwise inferred from crown_epsg.")
    parser.add_argument("--rgb-bands", default="1,2,3")
    parser.add_argument("--nir-band", type=int, default=1)
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument("--checkpoint-every", type=int, default=10000)
    parser.add_argument("--max-block-gb", type=float, default=80.0)
    parser.add_argument("--sid-block-tile-metres", type=float, default=1000.0)
    parser.add_argument(
        "--sid-path-rewrite",
        action="append",
        default=[],
        help="Optional OLD=NEW rewrite for SID paths stored in the paired index. Repeatable.",
    )
    parser.add_argument(
        "--sid-selection-mode",
        choices=("best-overlap", "indexed"),
        default="best-overlap",
        help="best-overlap scans all indexed SID pairs and keeps the least saturated crop.",
    )
    parser.add_argument(
        "--parallel-workers",
        type=int,
        default=1,
        help="Number of city workers. Each worker processes one city at a time.",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def parse_band_list(value: str) -> tuple[int, ...]:
    bands = tuple(int(part.strip()) for part in str(value).split(",") if part.strip())
    if len(bands) < 3 or min(bands) < 1:
        raise SystemExit("--rgb-bands must contain at least three positive 1-based bands, e.g. 1,2,3")
    return bands[:3]


def require_columns(frame: pd.DataFrame, columns: list[str], path: Path) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise RuntimeError(f"{path} is missing required column(s): {missing}")


def load_metadata(join_csv: Path, source_epsg: str, max_records: int | None) -> pd.DataFrame:
    join = pd.read_csv(join_csv, low_memory=False)
    require_columns(
        join,
        [
            "tree_id",
            "tree_lat",
            "tree_lon",
            "match_distance_m",
            "crown_id",
            "crown_lat",
            "crown_lon",
            "crown_epsg",
        ],
        join_csv,
    )
    join = join.sort_values("tree_id", kind="stable").reset_index(drop=True)
    if join["tree_id"].duplicated().any():
        duplicated = int(join["tree_id"].duplicated().sum())
        raise RuntimeError(f"{join_csv} has {duplicated:,} duplicated tree_id value(s); expected one row per tree_id")

    join["crown_epsg"] = pd.to_numeric(join["crown_epsg"], errors="coerce").astype("Int64")
    missing_geo = int(join[["crown_lat", "crown_lon", "crown_epsg"]].isna().any(axis=1).sum())
    if missing_geo:
        raise RuntimeError(f"{join_csv} has {missing_geo:,} row(s) missing crown_lat/crown_lon/crown_epsg")
    if not source_epsg:
        source_epsg = f"EPSG:{int(join['crown_epsg'].mode().iloc[0])}"
    inferred_epsg = str(source_epsg).upper().replace("EPSG:", "")
    mismatched_epsg = int(join["crown_epsg"].astype(int).ne(int(inferred_epsg)).sum()) if inferred_epsg.isdigit() else 0
    if mismatched_epsg:
        raise RuntimeError(
            f"{join_csv} has {mismatched_epsg:,} crown row(s) whose crown_epsg does not match --source-epsg {source_epsg}"
        )
    from pyproj import Transformer
    transformer = Transformer.from_crs("EPSG:4326", source_epsg, always_xy=True)
    crown_x, crown_y = transformer.transform(
        pd.to_numeric(join["crown_lon"], errors="raise").to_numpy(dtype=float),
        pd.to_numeric(join["crown_lat"], errors="raise").to_numpy(dtype=float),
    )
    join["crown_x_utm"] = np.asarray(crown_x, dtype=np.float64)
    join["crown_y_utm"] = np.asarray(crown_y, dtype=np.float64)

    metadata = join.drop(columns=[column for column in ["crown_x", "crown_y"] if column in join.columns]).copy()
    # In best-overlap mode the raster cropper only uses row_index as a stable
    # label for errors/progress. Keep it independent from old cell-centered IDs.
    metadata["row_index"] = np.arange(len(metadata), dtype=np.int64)
    metadata["reduced_id"] = metadata["row_index"].astype(np.int64)
    metadata["crop_index"] = np.arange(len(metadata), dtype=np.int64)
    if max_records is not None:
        metadata = metadata.head(int(max_records)).copy()
        metadata["crop_index"] = np.arange(len(metadata), dtype=np.int64)
    return metadata


def infer_epsg_from_paired_index(path: Path, default: str) -> str:
    try:
        values = pd.read_csv(path, usecols=["cell_epsg"], low_memory=False)["cell_epsg"]
    except Exception:
        return default
    values = pd.to_numeric(values, errors="coerce").dropna().astype(int)
    if values.empty:
        return default
    return f"EPSG:{int(values.mode().iloc[0])}"


def infer_epsg_from_join(path: Path, default: str) -> str:
    if default:
        return default
    values = pd.to_numeric(pd.read_csv(path, usecols=["crown_epsg"])["crown_epsg"], errors="coerce").dropna().astype(int)
    if values.empty:
        raise RuntimeError(f"{path} has no usable crown_epsg values")
    return f"EPSG:{int(values.mode().iloc[0])}"


def city_code_from_manifest(path: Path, city: str) -> str:
    payload = json.loads(path.read_text(encoding="utf-8"))
    wanted = "".join(ch for ch in city.lower() if ch.isalnum())
    for row in payload.get("cities", []):
        tokens = {
            "".join(ch for ch in str(row.get("city", "")).lower() if ch.isalnum()),
            "".join(ch for ch in str(row.get("code", "")).lower() if ch.isalnum()),
        }
        if wanted in tokens:
            return str(row.get("code", "")).upper()
    raise KeyError(f"City {city!r} not found in {path}")


def discover_paired_sources(naip_dir: Path, county_manifest: Path, city: str) -> list[tuple[Path, Path]]:
    code = city_code_from_manifest(county_manifest, city)
    city_dir = naip_dir / code
    if not city_dir.exists():
        raise FileNotFoundError(city_dir)
    archive_re = re.compile(r"^ortho_1-1_(?P<product>hn|nc|hc)_s_(?P<state>[a-z]{2})(?P<county>\d{3})_.*?(?P<year>\d{4})_\d+\.zip$", re.I)
    grouped: dict[tuple[str, str, str], dict[str, Path]] = {}
    for archive in sorted(city_dir.glob("*.zip")):
        match = archive_re.match(archive.name)
        if not match:
            continue
        kind = "companion" if match.group("product").lower() == "hc" else "natural"
        key = (match.group("state").lower(), match.group("county"), match.group("year"))
        grouped.setdefault(key, {})[kind] = archive
    pairs = []
    for files in grouped.values():
        if {"natural", "companion"}.issubset(files):
            natural = naip_utils.extract_sid_from_zip(files["natural"])
            companion = naip_utils.extract_sid_from_zip(files["companion"])
            pairs.append((natural, companion))
    if not pairs:
        raise RuntimeError(f"No same-county/year natural and companion NAIP archive pairs found under {city_dir}")
    return pairs


def format_pattern(pattern: str, city: str) -> str:
    return str(pattern).format(city=city, city_token=city)


def selected_city_tokens(args: argparse.Namespace) -> list[str]:
    if args.job_csv is not None:
        frame = pd.read_csv(args.job_csv, usecols=["city_token"], low_memory=False)
        cities = [str(value).strip().lower() for value in frame["city_token"].tolist() if str(value).strip()]
    elif args.city_token:
        cities = [str(value).strip().lower() for value in args.city_token if str(value).strip()]
    else:
        summary_path = Path(args.join_root) / "tree_to_detected_crowns_clean_summary.csv"
        if summary_path.exists():
            frame = pd.read_csv(summary_path, usecols=["city_token"], low_memory=False)
            cities = [str(value).strip().lower() for value in frame["city_token"].tolist() if str(value).strip()]
        else:
            cities = sorted(path.name.lower() for path in Path(args.join_root).iterdir() if path.is_dir())
    excluded = {str(value).strip().lower() for value in args.exclude_city_token}
    return sorted({city for city in cities if city and city not in excluded})


def build_city_jobs(args: argparse.Namespace) -> list[argparse.Namespace]:
    jobs: list[argparse.Namespace] = []
    if args.job_csv is not None:
        frame = pd.read_csv(args.job_csv, low_memory=False)
        required = {"city_token", "join_csv"}
        missing = required.difference(frame.columns)
        if missing:
            raise RuntimeError(f"{args.job_csv} is missing required column(s): {sorted(missing)}")
        excluded = {str(value).strip().lower() for value in args.exclude_city_token}
        for row in frame.itertuples(index=False):
            city = str(getattr(row, "city_token")).strip().lower()
            if not city or city in excluded:
                continue
            job = copy.copy(args)
            job.city_token = city
            job.join_csv = Path(str(getattr(row, "join_csv")))
            paired_value = getattr(row, "paired_index_csv", None) if "paired_index_csv" in frame.columns else None
            job.paired_index_csv = None if paired_value is None or pd.isna(paired_value) else Path(str(paired_value))
            if "source_epsg" in frame.columns and not pd.isna(getattr(row, "source_epsg")):
                job.source_epsg = str(getattr(row, "source_epsg"))
            else:
                job.source_epsg = infer_epsg_from_join(job.join_csv, args.source_epsg)
            if "output_prefix" in frame.columns and not pd.isna(getattr(row, "output_prefix")):
                job.output_prefix = str(getattr(row, "output_prefix"))
            jobs.append(job)
        return jobs

    cities = selected_city_tokens(args)
    for city in cities:
        job = copy.copy(args)
        job.city_token = city
        job.join_csv = Path(args.join_root) / city / format_pattern(args.join_pattern, city)
        if not job.join_csv.exists():
            flat_join = Path(args.join_root) / format_pattern(args.join_pattern, city)
            if flat_join.exists():
                job.join_csv = flat_join
        job.paired_index_csv = args.paired_index_csv
        job.source_epsg = infer_epsg_from_join(job.join_csv, args.source_epsg)
        job.output_prefix = None
        jobs.append(job)
    return jobs


def run_city(args: argparse.Namespace) -> dict[str, object]:
    if args.crop_size <= 0:
        raise SystemExit("--crop-size must be positive")
    if args.crop_metres <= 0:
        raise SystemExit("--crop-metres must be positive")
    if args.sid_selection_mode != "best-overlap":
        raise SystemExit(
            "Clean tree_id/crown metadata does not contain old paired-index row IDs; "
            "use --sid-selection-mode best-overlap."
        )
    if not args.join_csv.exists():
        raise FileNotFoundError(args.join_csv)
    if args.paired_index_csv is not None and not Path(args.paired_index_csv).exists():
        raise FileNotFoundError(args.paired_index_csv)

    city_dir = Path(args.output_dir) / args.city_token
    prefix = args.output_prefix or f"{args.city_token}_tree_id_centered_nearest_{args.crop_size}px"
    crops_path = city_dir / f"{prefix}_rgbnir_crops.npy"
    partial_crops_path = city_dir / f"{prefix}_rgbnir_crops.partial.npy"
    metadata_path = city_dir / f"{prefix}_metadata.csv"
    config_path = city_dir / f"{prefix}_config.json"
    progress_path = city_dir / f"{prefix}_progress.json"
    quality_progress_path = city_dir / f"{prefix}_best_overlap_quality_progress.csv"
    existing = [path for path in (crops_path, metadata_path, config_path) if path.exists()]
    if existing and not args.force:
        raise SystemExit(f"Output exists: {existing[0]}; pass --force to overwrite.")
    if args.force:
        for path in (partial_crops_path, progress_path, quality_progress_path):
            path.unlink(missing_ok=True)

    metadata = load_metadata(args.join_csv, args.source_epsg, args.max_records)
    city_dir.mkdir(parents=True, exist_ok=True)
    shape = (len(metadata), int(args.crop_size), int(args.crop_size), 4)
    if partial_crops_path.exists() and progress_path.exists():
        state = json.loads(progress_path.read_text(encoding="utf-8"))
        if tuple(state.get("shape", ())) != shape:
            raise RuntimeError("Existing progress shape does not match requested crop shape; use --force.")
        crops = np.lib.format.open_memmap(partial_crops_path, mode="r+", dtype=np.uint8, shape=shape)
        completed_positions = {int(value) for value in state.get("completed_positions", [])}
    else:
        crops = np.lib.format.open_memmap(partial_crops_path, mode="w+", dtype=np.uint8, shape=shape)
        crops[:] = 0
        crops.flush()
        completed_positions: set[int] = set()
        progress_path.write_text(json.dumps({"completed_positions": [], "shape": list(shape)}, indent=2), encoding="utf-8")

    natural_sids = [""] * len(metadata)
    companion_sids = [""] * len(metadata)
    crop_failed = np.zeros(len(metadata), dtype=bool)
    crop_blackout_fraction = np.full(len(metadata), np.nan, dtype=np.float32)
    crop_whiteout_fraction = np.full(len(metadata), np.nan, dtype=np.float32)
    crop_saturation_fraction = np.full(len(metadata), np.nan, dtype=np.float32)
    crop_valid_fraction = np.full(len(metadata), np.nan, dtype=np.float32)
    best_saturation = np.full(len(metadata), np.inf, dtype=np.float32)
    failed = 0
    start_time = time.perf_counter()
    initial_completed = len(completed_positions)
    last_checkpoint_completed = len(completed_positions)

    rows = list(metadata.itertuples(index=False))

    def write_progress() -> None:
        progress_path.write_text(
            json.dumps(
                {
                    "completed_positions": sorted(completed_positions),
                    "shape": list(shape),
                    "failed": int(failed),
                    "sid_selection_mode": args.sid_selection_mode,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    def write_quality_progress() -> None:
        pd.DataFrame(
            {
                "tree_id": metadata["tree_id"].to_numpy(),
                "crown_id": metadata["crown_id"].to_numpy(),
                "crop_natural_sid": natural_sids,
                "crop_companion_sid": companion_sids,
                "crop_blackout_fraction": crop_blackout_fraction,
                "crop_whiteout_fraction": crop_whiteout_fraction,
                "crop_saturation_fraction": crop_saturation_fraction,
                "crop_valid_fraction": crop_valid_fraction,
                "crop_failed": crop_failed,
            }
        ).to_csv(quality_progress_path, index=False)

    def report_progress() -> None:
        elapsed = max(time.perf_counter() - start_time, 1.0e-6)
        processed_this_run = max(len(completed_positions) - initial_completed, 0)
        rate = processed_this_run / elapsed
        remaining = len(metadata) - len(completed_positions)
        eta_minutes = remaining / rate / 60.0 if rate > 0 else float("inf")
        print(
            f"  cropped {len(completed_positions):,}/{len(metadata):,}; failed={failed:,}; "
            f"rate={rate:.1f}/s; eta={eta_minutes:.1f} min",
            flush=True,
        )

    def checkpoint(force: bool = False) -> None:
        nonlocal last_checkpoint_completed
        if force or len(completed_positions) - last_checkpoint_completed >= int(args.checkpoint_every):
            crops.flush()
            write_progress()
            write_quality_progress()
            report_progress()
            last_checkpoint_completed = len(completed_positions)

    def spatial_tile_key(row) -> tuple[int, int]:
        tile = float(args.sid_block_tile_metres)
        if tile <= 0:
            return (0, 0)
        return (int(np.floor(float(row.crown_x_utm) / tile)), int(np.floor(float(row.crown_y_utm) / tile)))

    def consider_candidate(position: int, crop: np.ndarray, natural_sid: str, companion_sid: str) -> bool:
        quality = naip_utils.chip_saturation_metrics(crop)
        score = float(quality["saturation_fraction"])
        valid = float(quality["valid_fraction"])
        current_score = float(best_saturation[position])
        current_valid = float(crop_valid_fraction[position]) if np.isfinite(crop_valid_fraction[position]) else -1.0
        if score < current_score or (score == current_score and valid > current_valid):
            crops[position] = crop
            natural_sids[position] = natural_sid
            companion_sids[position] = companion_sid
            crop_blackout_fraction[position] = quality["blackout_fraction"]
            crop_whiteout_fraction[position] = quality["whiteout_fraction"]
            crop_saturation_fraction[position] = quality["saturation_fraction"]
            crop_valid_fraction[position] = quality["valid_fraction"]
            best_saturation[position] = score
            crop_failed[position] = False
            return True
        return False

    def crop_one(position: int, row) -> None:
        nonlocal failed
        try:
            if args.sid_selection_mode == "best-overlap":
                crop, natural_sid, companion_sid, quality = cropper.crop_rgbnir_best_overlap(
                    int(row.row_index),
                    float(row.crown_x_utm),
                    float(row.crown_y_utm),
                    float(args.crop_metres),
                    int(args.crop_size),
                )
            else:
                crop, natural_sid, companion_sid = cropper.crop_rgbnir(
                    int(row.row_index),
                    float(row.crown_x_utm),
                    float(row.crown_y_utm),
                    float(args.crop_metres),
                    int(args.crop_size),
                )
                quality = naip_utils.chip_saturation_metrics(crop)
            crops[position] = crop
            natural_sids[position] = natural_sid
            companion_sids[position] = companion_sid
            crop_blackout_fraction[position] = quality["blackout_fraction"]
            crop_whiteout_fraction[position] = quality["whiteout_fraction"]
            crop_saturation_fraction[position] = quality["saturation_fraction"]
            crop_valid_fraction[position] = quality["valid_fraction"]
            crop_failed[position] = False
        except Exception as error:
            failed += 1
            crop_failed[position] = True
            if failed <= 10:
                print(f"WARNING: crop failed at position={position}; tree_id={row.tree_id}: {error}", flush=True)
        completed_positions.add(position)

    print(
        f"Extracting clean tree_id-centered NAIP crops: records={len(metadata):,}; "
        f"crop={args.crop_size}x{args.crop_size}; crop_metres={args.crop_metres}; "
        f"output={crops_path}; sid_selection={args.sid_selection_mode}; resume={len(completed_positions):,}",
        flush=True,
    )

    cropper = None
    try:
        sid_pairs = None if args.paired_index_csv is not None else discover_paired_sources(Path(args.naip_dir), Path(args.county_manifest), args.city_token)
        cropper = naip_utils.PairedNaipRasterCropper(
            args.paired_index_csv,
            args.source_epsg,
            rgb_bands=parse_band_list(args.rgb_bands),
            nir_band=args.nir_band,
            path_rewrites=naip_utils.parse_path_rewrites(args.sid_path_rewrite),
            sid_pairs=sid_pairs,
        )
        if args.sid_selection_mode == "best-overlap":
            open_positions = [position for position in range(len(rows)) if position not in completed_positions]
            print(
                f"  Best-overlap SID search enabled: scanning {len(cropper.unique_sid_pairs):,} SID pair(s) "
                f"over {len(open_positions):,} crop record(s) with block/tile reads.",
                flush=True,
            )
            for pair_index, (natural_sid, companion_sid) in enumerate(cropper.unique_sid_pairs, start=1):
                pair_start = time.perf_counter()
                tile_groups: dict[tuple[int, int], list[int]] = defaultdict(list)
                for position in open_positions:
                    if position not in completed_positions:
                        tile_groups[spatial_tile_key(rows[position])].append(position)
                improved = 0
                for tile_index, positions in enumerate(tile_groups.values(), start=1):
                    positions = [position for position in positions if position not in completed_positions]
                    if not positions:
                        continue
                    batch_rows = [
                        (int(position), float(rows[position].crown_x_utm), float(rows[position].crown_y_utm))
                        for position in positions
                    ]
                    try:
                        batch_results = cropper.crop_rgbnir_batch_from_pair(
                            natural_sid,
                            companion_sid,
                            batch_rows,
                            float(args.crop_metres),
                            int(args.crop_size),
                            max_block_gb=float(args.max_block_gb),
                        )
                        for position, crop, natural_text, companion_text in batch_results:
                            improved += int(consider_candidate(int(position), crop, natural_text, companion_text))
                    except Exception as error:
                        print(
                            f"WARNING: best-overlap block crop failed for SID pair {pair_index}, "
                            f"tile {tile_index} (records={len(positions):,}); falling back to per-crop: {error}",
                            flush=True,
                        )
                        for position in positions:
                            try:
                                crop, natural_text, companion_text = cropper.crop_rgbnir_from_pair(
                                    natural_sid,
                                    companion_sid,
                                    int(position),
                                    float(rows[position].crown_x_utm),
                                    float(rows[position].crown_y_utm),
                                    float(args.crop_metres),
                                    int(args.crop_size),
                                )
                                improved += int(consider_candidate(position, crop, natural_text, companion_text))
                            except Exception:
                                continue
                elapsed = max(time.perf_counter() - pair_start, 1.0e-6)
                print(
                    f"  SID pair {pair_index:,}/{len(cropper.unique_sid_pairs):,}: "
                    f"tiles={len(tile_groups):,}; improved={improved:,}; elapsed={elapsed/60.0:.1f} min",
                    flush=True,
                )
                crops.flush()
                write_progress()
                write_quality_progress()
            missing_best = np.flatnonzero(~np.isfinite(best_saturation))
            for position in missing_best:
                failed += 1
                crop_failed[int(position)] = True
            for position in range(len(rows)):
                completed_positions.add(position)
            checkpoint(force=True)
        else:
            sid_groups: dict[tuple[str, str], list[int]] = defaultdict(list)
            for position, row in enumerate(rows):
                if position in completed_positions:
                    continue
                natural_sid, companion_sid = cropper.sid_pair_for_row(int(row.row_index))
                sid_groups[(str(natural_sid), str(companion_sid))].append(position)
            for group_index, positions in enumerate(sid_groups.values(), start=1):
                positions = [position for position in positions if position not in completed_positions]
                tile_groups: dict[tuple[int, int], list[int]] = defaultdict(list)
                for position in positions:
                    tile_groups[spatial_tile_key(rows[position])].append(position)
                for tile_index, tile_positions in enumerate(tile_groups.values(), start=1):
                    batch_rows = [
                        (int(rows[position].row_index), float(rows[position].crown_x_utm), float(rows[position].crown_y_utm))
                        for position in tile_positions
                    ]
                    try:
                        batch_results = cropper.crop_rgbnir_batch(
                            batch_rows,
                            float(args.crop_metres),
                            int(args.crop_size),
                            max_block_gb=float(args.max_block_gb),
                        )
                        for position, (crop, natural_sid, companion_sid) in zip(tile_positions, batch_results):
                            quality = naip_utils.chip_saturation_metrics(crop)
                            crops[position] = crop
                            natural_sids[position] = natural_sid
                            companion_sids[position] = companion_sid
                            crop_blackout_fraction[position] = quality["blackout_fraction"]
                            crop_whiteout_fraction[position] = quality["whiteout_fraction"]
                            crop_saturation_fraction[position] = quality["saturation_fraction"]
                            crop_valid_fraction[position] = quality["valid_fraction"]
                            crop_failed[position] = False
                            completed_positions.add(position)
                    except Exception:
                        for position in tile_positions:
                            crop_one(position, rows[position])
                    checkpoint()
            checkpoint(force=True)
    finally:
        crops.flush()
        del crops
        if cropper is not None:
            cropper.close()

    output_metadata = metadata.copy()
    output_metadata["crop_source"] = "source_raster"
    output_metadata["crop_size"] = int(args.crop_size)
    output_metadata["crop_metres"] = float(args.crop_metres)
    output_metadata["crop_failed"] = crop_failed
    output_metadata["crop_blackout_fraction"] = crop_blackout_fraction
    output_metadata["crop_whiteout_fraction"] = crop_whiteout_fraction
    output_metadata["crop_saturation_fraction"] = crop_saturation_fraction
    output_metadata["crop_valid_fraction"] = crop_valid_fraction
    output_metadata["crop_natural_sid"] = natural_sids
    output_metadata["crop_companion_sid"] = companion_sids
    output_metadata.to_csv(metadata_path, index=False)
    partial_crops_path.replace(crops_path)
    progress_path.unlink(missing_ok=True)
    quality_progress_path.unlink(missing_ok=True)
    config = {
        "join_csv": str(args.join_csv),
        "paired_index_csv": str(args.paired_index_csv or ""),
        "naip_dir": str(args.naip_dir),
        "source_discovery": "paired_index" if args.paired_index_csv is not None else "downloaded_archives",
        "output_crops": str(crops_path),
        "output_metadata": str(metadata_path),
        "city_token": args.city_token,
        "primary_record_key": "tree_id",
        "supporting_crown_key": "crown_id",
        "crop_center_columns": ["crown_lat", "crown_lon", "crown_x_utm", "crown_y_utm"],
        "crop_size": int(args.crop_size),
        "crop_metres": float(args.crop_metres),
        "source_epsg": str(args.source_epsg),
        "rgb_bands": parse_band_list(args.rgb_bands),
        "nir_band": int(args.nir_band),
        "sid_selection_mode": str(args.sid_selection_mode),
        "max_block_gb": float(args.max_block_gb),
        "sid_block_tile_metres": float(args.sid_block_tile_metres),
        "record_count": int(len(output_metadata)),
        "failed_count": int(crop_failed.sum()),
    }
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(f"Wrote crops: {crops_path}", flush=True)
    print(f"Wrote metadata: {metadata_path}", flush=True)
    print(f"Wrote config: {config_path}", flush=True)
    print(f"Rows={len(output_metadata):,}; crop_failed={int(crop_failed.sum()):,}", flush=True)
    return {
        "city_token": str(args.city_token),
        "status": "completed",
        "rows": int(len(output_metadata)),
        "crop_failed": int(crop_failed.sum()),
        "metadata_csv": str(metadata_path),
        "crops_npy": str(crops_path),
    }


def run_city_guarded(args: argparse.Namespace) -> dict[str, object]:
    try:
        print(f"\n=== {args.city_token} ===", flush=True)
        return run_city(args)
    except BaseException as error:
        print(f"ERROR {args.city_token}: {error}", flush=True)
        return {
            "city_token": str(args.city_token),
            "status": "failed",
            "error": str(error),
            "rows": 0,
            "crop_failed": 0,
            "metadata_csv": "",
            "crops_npy": "",
        }


def main() -> int:
    args = parse_args()
    jobs = build_city_jobs(args)
    if not jobs:
        raise SystemExit("No city jobs selected.")
    workers = max(1, int(args.parallel_workers))
    workers = min(workers, len(jobs))
    print(
        f"Selected {len(jobs):,} city job(s): {', '.join(job.city_token for job in jobs[:20])}"
        f"{' ...' if len(jobs) > 20 else ''}; parallel_workers={workers}",
        flush=True,
    )

    if workers == 1:
        rows = [run_city_guarded(job) for job in jobs]
    else:
        rows = []
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
            future_to_city = {executor.submit(run_city_guarded, job): job.city_token for job in jobs}
            for future in concurrent.futures.as_completed(future_to_city):
                rows.append(future.result())

    summary = pd.DataFrame(rows).sort_values("city_token", kind="stable")
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    summary_path = Path(args.output_dir) / "tree_id_centered_naip_crops_clean_summary.csv"
    summary.to_csv(summary_path, index=False)
    print("\nSummary:", flush=True)
    print(summary["status"].value_counts(dropna=False).to_string(), flush=True)
    print(f"Wrote summary: {summary_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
