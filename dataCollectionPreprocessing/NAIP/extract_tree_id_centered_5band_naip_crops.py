#!/usr/bin/env python3
"""Extract clean tree_id-centered RGBNIR crops from source 5-band NAIP rasters.

Run this with the ArcGIS Pro Python environment. It reads the clean
tree-to-crown join directly, discovers source SID rasters, crops around each
identified crown, and writes row-aligned RGBNIR arrays and metadata. An older
5-band index may still be supplied for compatibility, but it is not required.

The TensorFlow embedding extraction step can then run in a separate environment
without ArcPy.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import tree_centered_naip_utils as naip_utils


DEFAULT_JOIN_ROOT = Path(r"H:\TreeCenteredModelInputs\tree_to_detected_crowns_clean")
DEFAULT_NAIP_DIR = Path(r"E:\NAIP_PAIRED")
DEFAULT_OUTPUT_DIR = Path(r"H:\TreeCenteredModelInputs\tree_centered_naip_crops_clean")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--join-csv", type=Path, default=None, help="Clean tree-to-crown join. Defaults to --join-root/<city>.")
    parser.add_argument("--join-root", type=Path, default=DEFAULT_JOIN_ROOT)
    parser.add_argument("--join-pattern", default="{city}_tree_to_nearest_detected_crown_5m.csv")
    parser.add_argument("--metadata-csv", type=Path, default=None, help="Compatibility input containing prepared crop metadata.")
    parser.add_argument("--fiveband-index-csv", type=Path, default=None, help="Optional legacy source-raster index.")
    parser.add_argument("--naip-dir", type=Path, default=DEFAULT_NAIP_DIR)
    parser.add_argument("--sid-pattern", default="*.sid")
    parser.add_argument("--bands", default="1,2,3,4", help="1-based source bands written as R,G,B,NIR.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--city-token", default="baltimore")
    parser.add_argument("--output-prefix", default=None)
    parser.add_argument("--crop-size", type=int, default=64)
    parser.add_argument("--crop-metres", type=float, default=38.0)
    parser.add_argument("--source-epsg", default="", help="Override source EPSG; otherwise inferred from crown_epsg.")
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument("--checkpoint-every", type=int, default=10000)
    parser.add_argument(
        "--crop-mode",
        choices=("sid-block", "per-crop"),
        default="sid-block",
        help="Use one larger raster read per SID source when possible, or the original per-record crop reads.",
    )
    parser.add_argument(
        "--max-block-gb",
        type=float,
        default=48.0,
        help="Maximum estimated decompressed 5-band SID block size for --crop-mode sid-block.",
    )
    parser.add_argument(
        "--min-block-records",
        type=int,
        default=2,
        help="Minimum records in a SID source before using block mode; smaller groups use per-crop mode.",
    )
    parser.add_argument(
        "--sid-block-tile-metres",
        type=float,
        default=1000.0,
        help="Spatial tile size used within each SID source for block reads. Lower this if ArcPy reports pixel blocks are too large.",
    )
    parser.add_argument(
        "--sid-path-rewrite",
        action="append",
        default=[],
        help="Optional OLD=NEW rewrite for SID paths stored in the 5-band index. Repeatable.",
    )
    parser.add_argument(
        "--reuse-crop-root",
        type=Path,
        default=None,
        help="Optional previous crop root. Matching tree_centered_index rows are copied before cropping new rows.",
    )
    parser.add_argument(
        "--sid-selection-mode",
        choices=("indexed", "best-overlap"),
        default="best-overlap",
        help=(
            "indexed uses the SID selected in the 5-band cell index. "
            "best-overlap tries every city 5-band SID covering the crown location "
            "and keeps the crop with the lowest 0/255 saturation."
        ),
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def require_columns(frame: pd.DataFrame, columns: list[str], path: Path) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise RuntimeError(f"{path} is missing required column(s): {missing}")


def parse_bands(value: str) -> tuple[int, int, int, int]:
    bands = tuple(int(part.strip()) for part in str(value).split(",") if part.strip())
    if len(bands) != 4 or min(bands) < 1:
        raise SystemExit("--bands must contain four positive 1-based bands: R,G,B,NIR")
    return bands


def discover_sid_sources(naip_dir: Path, city_token: str, pattern: str) -> list[Path]:
    city = city_token.strip().lower()
    candidates = [naip_dir, naip_dir / city, naip_dir / city.upper()]
    if naip_dir.exists():
        candidates.extend(path for path in naip_dir.iterdir() if path.is_dir() and city in path.name.lower())
    roots = list(dict.fromkeys(path.resolve() for path in candidates if path.exists()))
    sources = {path.resolve() for root in roots for path in root.rglob(pattern)}
    # Downloads are preserved as ZIP archives. Extract only matching SID members,
    # alongside their archive, when a future run has not unpacked them yet.
    for root in roots:
        for archive_path in root.rglob("*.zip"):
            with zipfile.ZipFile(archive_path) as archive:
                for member in archive.infolist():
                    member_name = Path(member.filename).name
                    if not member_name or not Path(member_name).match(pattern):
                        continue
                    destination = archive_path.parent / member_name
                    if not destination.exists() or destination.stat().st_size != member.file_size:
                        partial = destination.with_suffix(destination.suffix + ".partial")
                        partial.unlink(missing_ok=True)
                        with archive.open(member) as source, partial.open("wb") as target:
                            while chunk := source.read(16 * 1024 * 1024):
                                target.write(chunk)
                        partial.replace(destination)
                    sources.add(destination.resolve())
    sources = sorted(sources)
    if not sources:
        raise FileNotFoundError(f"No 5-band SID rasters matching {pattern!r} for {city_token} under {naip_dir}")
    return sources


def prepare_clean_metadata(join_csv: Path, source_epsg: str, max_records: int | None) -> tuple[pd.DataFrame, str]:
    metadata = pd.read_csv(join_csv, low_memory=False)
    require_columns(metadata, ["tree_id", "crown_id", "crown_epsg", "match_distance_m"], join_csv)
    metadata = metadata.sort_values("tree_id", kind="stable").reset_index(drop=True)
    if metadata["tree_id"].duplicated().any():
        raise RuntimeError(f"{join_csv} has duplicated tree_id values")
    if max_records is not None:
        metadata = metadata.head(int(max_records)).copy()
    inferred = pd.to_numeric(metadata["crown_epsg"], errors="coerce").dropna().astype(int)
    if not source_epsg:
        if inferred.empty:
            raise RuntimeError(f"{join_csv} has no usable crown_epsg values")
        source_epsg = f"EPSG:{int(inferred.mode().iloc[0])}"
    epsg_number = int(str(source_epsg).upper().replace("EPSG:", ""))
    if "crown_x_utm" in metadata and "crown_y_utm" in metadata:
        x = pd.to_numeric(metadata["crown_x_utm"], errors="raise").to_numpy(dtype=float)
        y = pd.to_numeric(metadata["crown_y_utm"], errors="raise").to_numpy(dtype=float)
    else:
        require_columns(metadata, ["crown_lon", "crown_lat"], join_csv)
        from pyproj import Transformer
        transformer = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg_number}", always_xy=True)
        x, y = transformer.transform(
            pd.to_numeric(metadata["crown_lon"], errors="raise").to_numpy(dtype=float),
            pd.to_numeric(metadata["crown_lat"], errors="raise").to_numpy(dtype=float),
        )
    metadata["crown_x_utm"], metadata["crown_y_utm"] = np.asarray(x), np.asarray(y)
    metadata["peak_x_utm"], metadata["peak_y_utm"] = np.asarray(x), np.asarray(y)
    metadata["tree_centered_index"] = np.arange(len(metadata), dtype=np.int64)
    metadata["reduced_id"] = np.arange(len(metadata), dtype=np.int64)
    metadata["assigned_peak_id"] = pd.to_numeric(metadata["crown_id"], errors="raise").astype(np.int64)
    return metadata, f"EPSG:{epsg_number}"


def main() -> int:
    args = parse_args()
    if args.crop_size <= 0:
        raise SystemExit("--crop-size must be positive")
    if args.crop_metres <= 0:
        raise SystemExit("--crop-metres must be positive")
    compatibility_metadata = getattr(args, "metadata_csv", None)
    join_csv = getattr(args, "join_csv", None)
    if compatibility_metadata is None:
        join_csv = Path(join_csv) if join_csv is not None else Path(args.join_root) / args.city_token / str(args.join_pattern).format(city=args.city_token, city_token=args.city_token)
        if not join_csv.exists():
            raise FileNotFoundError(join_csv)
        metadata, source_epsg = prepare_clean_metadata(join_csv, str(args.source_epsg), args.max_records)
        args.source_epsg = source_epsg
    else:
        compatibility_metadata = Path(compatibility_metadata)
        if not compatibility_metadata.exists():
            raise FileNotFoundError(compatibility_metadata)
        metadata = pd.read_csv(compatibility_metadata, low_memory=False)
    index_csv = getattr(args, "fiveband_index_csv", None)
    if index_csv is not None:
        index_csv = Path(index_csv)
        if not index_csv.exists():
            raise FileNotFoundError(index_csv)
    bands = parse_bands(getattr(args, "bands", "1,2,3,4"))
    sid_sources = None if index_csv is not None else discover_sid_sources(Path(args.naip_dir), args.city_token, args.sid_pattern)

    city_dir = args.output_dir / args.city_token
    prefix = args.output_prefix or f"{args.city_token}_tree_centered_nearest_{args.crop_size}px"
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
        partial_crops_path.unlink(missing_ok=True)
        progress_path.unlink(missing_ok=True)
        quality_progress_path.unlink(missing_ok=True)

    require_columns(
        metadata,
        [
            "tree_centered_index",
            "reduced_id",
            "peak_x_utm",
            "peak_y_utm",
            "assigned_peak_id",
            "match_distance_m",
        ],
        Path(compatibility_metadata or join_csv),
    )
    metadata = metadata.sort_values("tree_centered_index", kind="stable").reset_index(drop=True)
    if args.max_records is not None:
        metadata = metadata.head(int(args.max_records)).copy()
    if metadata.empty:
        raise SystemExit("No metadata rows to crop.")

    city_dir.mkdir(parents=True, exist_ok=True)
    shape = (len(metadata), int(args.crop_size), int(args.crop_size), 4)
    if partial_crops_path.exists() and progress_path.exists():
        state = json.loads(progress_path.read_text(encoding="utf-8"))
        if tuple(state.get("shape", ())) != shape:
            raise RuntimeError("Existing progress shape does not match requested crop shape; use --force.")
        crops = np.lib.format.open_memmap(partial_crops_path, mode="r+", dtype=np.uint8, shape=shape)
        if "completed_positions" in state:
            completed_positions = {int(value) for value in state.get("completed_positions", [])}
        else:
            completed_positions = set(range(int(state.get("next_index", 0))))
    else:
        crops = np.lib.format.open_memmap(partial_crops_path, mode="w+", dtype=np.uint8, shape=shape)
        crops[:] = 0
        crops.flush()
        completed_positions = set()
        progress_path.write_text(json.dumps({"completed_positions": [], "shape": list(shape)}, indent=2), encoding="utf-8")

    cropper = None
    source_sids = [""] * len(metadata)
    red_bands = np.full(len(metadata), -1, dtype=np.int16)
    green_bands = np.full(len(metadata), -1, dtype=np.int16)
    blue_bands = np.full(len(metadata), -1, dtype=np.int16)
    nir_bands = np.full(len(metadata), -1, dtype=np.int16)
    crop_failed = np.zeros(len(metadata), dtype=bool)
    crop_blackout_fraction = np.full(len(metadata), np.nan, dtype=np.float32)
    crop_whiteout_fraction = np.full(len(metadata), np.nan, dtype=np.float32)
    crop_saturation_fraction = np.full(len(metadata), np.nan, dtype=np.float32)
    crop_valid_fraction = np.full(len(metadata), np.nan, dtype=np.float32)
    best_saturation = np.full(len(metadata), np.inf, dtype=np.float32)
    failed = 0
    start_time = time.perf_counter()
    last_checkpoint_completed = len(completed_positions)

    def stable_reuse_positions(frame: pd.DataFrame) -> dict[tuple[int, int], int]:
        required = {"reduced_id", "assigned_peak_id"}
        if not required.issubset(frame.columns):
            return {}
        reduced_ids = pd.to_numeric(frame["reduced_id"], errors="coerce")
        peak_ids = pd.to_numeric(frame["assigned_peak_id"], errors="coerce")
        lookup: dict[tuple[int, int], int] = {}
        for pos, (reduced_id, peak_id) in enumerate(zip(reduced_ids, peak_ids)):
            if pd.isna(reduced_id) or pd.isna(peak_id):
                continue
            lookup.setdefault((int(reduced_id), int(peak_id)), int(pos))
        return lookup

    def seed_from_reuse_root() -> int:
        if args.reuse_crop_root is None or completed_positions:
            return 0
        reuse_dir = Path(args.reuse_crop_root) / args.city_token
        old_crops_path = reuse_dir / f"{prefix}_rgbnir_crops.npy"
        old_metadata_path = reuse_dir / f"{prefix}_metadata.csv"
        if not old_crops_path.exists() or not old_metadata_path.exists():
            return 0
        old_metadata = pd.read_csv(old_metadata_path, low_memory=False)
        old_stable_positions = stable_reuse_positions(old_metadata)
        current_stable_positions = stable_reuse_positions(metadata)
        if not old_stable_positions and "tree_centered_index" not in old_metadata.columns:
            return 0
        old_crops = np.load(old_crops_path, mmap_mode="r")
        old_index_positions = {}
        if "tree_centered_index" in old_metadata.columns and "tree_centered_index" in metadata.columns:
            old_index_positions = {
                int(value): int(pos)
                for pos, value in enumerate(pd.to_numeric(old_metadata["tree_centered_index"], errors="coerce").fillna(-1).astype(np.int64))
                if int(value) >= 0
            }
        copied = 0
        quality_columns = {
            "crop_source_sid": source_sids,
            "crop_red_band": red_bands,
            "crop_green_band": green_bands,
            "crop_blue_band": blue_bands,
            "crop_nir_band": nir_bands,
            "crop_failed": crop_failed,
            "crop_blackout_fraction": crop_blackout_fraction,
            "crop_whiteout_fraction": crop_whiteout_fraction,
            "crop_saturation_fraction": crop_saturation_fraction,
            "crop_valid_fraction": crop_valid_fraction,
        }
        current_tree_indices = (
            pd.to_numeric(metadata["tree_centered_index"], errors="coerce").fillna(-1).astype(np.int64).to_numpy()
            if "tree_centered_index" in metadata.columns
            else np.full(len(metadata), -1, dtype=np.int64)
        )
        current_keys = {
            position: key
            for key, position in current_stable_positions.items()
        }
        for position, tree_index in enumerate(current_tree_indices):
            old_position = None
            stable_key = current_keys.get(position)
            if stable_key is not None:
                old_position = old_stable_positions.get(stable_key)
            if old_position is None and int(tree_index) >= 0:
                old_position = old_index_positions.get(int(tree_index))
            if old_position is None or old_position >= old_crops.shape[0]:
                continue
            crops[position] = old_crops[old_position]
            for column, target in quality_columns.items():
                if column not in old_metadata.columns:
                    continue
                value = old_metadata[column].iloc[old_position]
                if target is source_sids:
                    target[position] = "" if pd.isna(value) else str(value)
                else:
                    try:
                        target[position] = value
                    except Exception:
                        pass
            completed_positions.add(int(position))
            copied += 1
        if copied:
            crops.flush()
            progress_path.write_text(json.dumps({"completed_positions": sorted(completed_positions), "shape": list(shape)}, indent=2), encoding="utf-8")
            method = "reduced_id+assigned_peak_id" if old_stable_positions and current_stable_positions else "tree_centered_index"
            print(f"Seeded {copied:,} existing crops from {reuse_dir} using {method}", flush=True)
        return copied

    seed_from_reuse_root()
    last_checkpoint_completed = len(completed_positions)

    def write_progress() -> None:
        progress_path.write_text(
            json.dumps(
                {
                    "completed_positions": sorted(completed_positions),
                    "shape": list(shape),
                    "failed": failed,
                    "crop_mode": args.crop_mode,
                    "sid_selection_mode": args.sid_selection_mode,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    def write_quality_progress() -> None:
        if args.sid_selection_mode != "best-overlap":
            return
        pd.DataFrame(
            {
                "tree_centered_index": metadata["tree_centered_index"].to_numpy(),
                "crop_source_sid": source_sids,
                "crop_red_band": red_bands,
                "crop_green_band": green_bands,
                "crop_blue_band": blue_bands,
                "crop_nir_band": nir_bands,
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
            report_progress()
            last_checkpoint_completed = len(completed_positions)

    def set_source(position: int, source_sid: str, bands: tuple[int, int, int, int]) -> None:
        source_sids[position] = source_sid
        red_bands[position] = int(bands[0])
        green_bands[position] = int(bands[1])
        blue_bands[position] = int(bands[2])
        nir_bands[position] = int(bands[3])

    def crop_one(position: int, row) -> None:
        nonlocal failed
        try:
            if args.sid_selection_mode == "best-overlap":
                crop, source_sid, bands, quality = cropper.crop_rgbnir_best_overlap(
                    int(getattr(row, "reduced_id")),
                    float(getattr(row, "peak_x_utm")),
                    float(getattr(row, "peak_y_utm")),
                    float(args.crop_metres),
                    int(args.crop_size),
                )
            else:
                crop, source_sid, bands = cropper.crop_rgbnir(
                    int(getattr(row, "reduced_id")),
                    float(getattr(row, "peak_x_utm")),
                    float(getattr(row, "peak_y_utm")),
                    float(args.crop_metres),
                    int(args.crop_size),
                )
                quality = naip_utils.chip_saturation_metrics(crop)
            crops[position] = crop
            set_source(position, source_sid, bands)
            crop_blackout_fraction[position] = quality["blackout_fraction"]
            crop_whiteout_fraction[position] = quality["whiteout_fraction"]
            crop_saturation_fraction[position] = quality["saturation_fraction"]
            crop_valid_fraction[position] = quality["valid_fraction"]
        except Exception as error:
            failed += 1
            crop_failed[position] = True
            if failed <= 10:
                print(f"WARNING: crop failed at position={position}; tree_centered_index={getattr(row, 'tree_centered_index')}: {error}", flush=True)
        completed_positions.add(position)

    def spatial_tile_key(row) -> tuple[int, int]:
        tile = float(args.sid_block_tile_metres)
        if tile <= 0:
            return (0, 0)
        return (
            int(np.floor(float(getattr(row, "peak_x_utm")) / tile)),
            int(np.floor(float(getattr(row, "peak_y_utm")) / tile)),
        )

    def crop_positions_with_block(positions: list[int], label: str) -> None:
        positions = [position for position in positions if position not in completed_positions]
        if not positions:
            return
        if len(positions) < int(args.min_block_records):
            for position in positions:
                crop_one(position, rows[position])
                checkpoint()
            return
        try:
            first_row = rows[positions[0]]
            source_sid, bands = cropper.sid_source_for_row(int(getattr(first_row, "reduced_id")))
            batch_rows = [
                (
                    int(position),
                    float(getattr(rows[position], "peak_x_utm")),
                    float(getattr(rows[position], "peak_y_utm")),
                )
                for position in positions
            ]
            batch_results = cropper.crop_rgbnir_batch_from_source(
                source_sid,
                bands,
                batch_rows,
                float(args.crop_metres),
                int(args.crop_size),
                max_block_gb=float(args.max_block_gb),
            )
            for position, crop, source_text, source_bands in batch_results:
                position = int(position)
                crops[position] = crop
                set_source(position, source_text, source_bands)
                quality = naip_utils.chip_saturation_metrics(crop)
                crop_blackout_fraction[position] = quality["blackout_fraction"]
                crop_whiteout_fraction[position] = quality["whiteout_fraction"]
                crop_saturation_fraction[position] = quality["saturation_fraction"]
                crop_valid_fraction[position] = quality["valid_fraction"]
                crop_failed[position] = False
                completed_positions.add(position)
        except Exception as error:
            print(
                f"WARNING: block crop failed for {label} (records={len(positions):,}); "
                f"falling back to per-crop mode: {error}",
                flush=True,
            )
            for position in positions:
                crop_one(position, rows[position])
        checkpoint()

    def consider_candidate(position: int, crop: np.ndarray, source_sid: str, bands: tuple[int, int, int, int]) -> bool:
        quality = naip_utils.chip_saturation_metrics(crop)
        score = float(quality["saturation_fraction"])
        valid = float(quality["valid_fraction"])
        current_score = float(best_saturation[position])
        current_valid = float(crop_valid_fraction[position]) if np.isfinite(crop_valid_fraction[position]) else -1.0
        if score < current_score or (score == current_score and valid > current_valid):
            crops[position] = crop
            set_source(position, source_sid, bands)
            crop_blackout_fraction[position] = quality["blackout_fraction"]
            crop_whiteout_fraction[position] = quality["whiteout_fraction"]
            crop_saturation_fraction[position] = quality["saturation_fraction"]
            crop_valid_fraction[position] = quality["valid_fraction"]
            best_saturation[position] = score
            crop_failed[position] = False
            return True
        return False

    def crop_best_overlap_tile(positions: list[int], source_sid: Path, bands: tuple[int, int, int, int], label: str) -> int:
        positions = [position for position in positions if position not in completed_positions]
        if not positions:
            return 0
        batch_rows = [
            (
                int(position),
                float(getattr(rows[position], "peak_x_utm")),
                float(getattr(rows[position], "peak_y_utm")),
            )
            for position in positions
        ]
        improved = 0
        try:
            batch_results = cropper.crop_rgbnir_batch_from_source(
                source_sid,
                bands,
                batch_rows,
                float(args.crop_metres),
                int(args.crop_size),
                max_block_gb=float(args.max_block_gb),
            )
            for position, crop, source_text, source_bands in batch_results:
                improved += int(consider_candidate(int(position), crop, source_text, source_bands))
        except Exception as error:
            print(
                f"WARNING: best-overlap block crop failed for {label} (records={len(positions):,}); "
                f"falling back to per-crop candidate mode: {error}",
                flush=True,
            )
            for position in positions:
                try:
                    crop, source_text, source_bands = cropper.crop_rgbnir_from_source(
                        source_sid,
                        bands,
                        int(position),
                        float(getattr(rows[position], "peak_x_utm")),
                        float(getattr(rows[position], "peak_y_utm")),
                        float(args.crop_metres),
                        int(args.crop_size),
                    )
                    improved += int(consider_candidate(position, crop, source_text, source_bands))
                except Exception:
                    continue
        return improved

    def crop_best_overlap_all_sources() -> None:
        nonlocal failed
        open_positions = [position for position in range(len(rows)) if position not in completed_positions]
        print(
            f"  Best-overlap SID search enabled: scanning {len(cropper.unique_sid_sources):,} 5-band SID source(s) "
            f"over {len(open_positions):,} crop record(s) with block/tile reads.",
            flush=True,
        )
        for source_index, (source_sid, bands) in enumerate(cropper.unique_sid_sources, start=1):
            source_start = time.perf_counter()
            tile_groups: dict[tuple[int, int], list[int]] = defaultdict(list)
            for position in open_positions:
                tile_groups[spatial_tile_key(rows[position])].append(position)
            improved = 0
            for tile_index, tile_positions in enumerate(tile_groups.values(), start=1):
                tile_start = time.perf_counter()
                before_improved = improved
                improved += crop_best_overlap_tile(
                    tile_positions,
                    source_sid,
                    bands,
                    f"SID source {source_index}/{len(cropper.unique_sid_sources)}, tile {tile_index}/{len(tile_groups)}",
                )
                tile_elapsed = max(time.perf_counter() - tile_start, 1.0e-6)
                if tile_index == 1 or tile_index == len(tile_groups) or tile_index % 10 == 0:
                    print(
                        f"    SID source {source_index:,}/{len(cropper.unique_sid_sources):,}; "
                        f"tile {tile_index:,}/{len(tile_groups):,}; records={len(tile_positions):,}; "
                        f"improved_this_tile={improved - before_improved:,}; "
                        f"elapsed={tile_elapsed:.1f}s",
                        flush=True,
                    )
                    crops.flush()
                    write_quality_progress()
            elapsed = max(time.perf_counter() - source_start, 1.0e-6)
            print(
                f"  SID source {source_index:,}/{len(cropper.unique_sid_sources):,}: "
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

    initial_completed = len(completed_positions)
    print(
        f"Extracting source-raster crown crops: records={len(metadata):,}; crop={args.crop_size}x{args.crop_size}; "
        f"crop_metres={args.crop_metres}; output={crops_path}; mode={args.crop_mode}; "
        f"sid_selection={args.sid_selection_mode}; resume={initial_completed:,}",
        flush=True,
    )
    try:
        rows = list(metadata.itertuples(index=False))
        if len(completed_positions) < len(metadata):
            cropper = naip_utils.FiveBandNaipRasterCropper(
                index_csv,
                args.source_epsg,
                path_rewrites=naip_utils.parse_path_rewrites(args.sid_path_rewrite),
                sid_sources=sid_sources,
                bands=bands,
            )
            if args.sid_selection_mode == "best-overlap":
                crop_best_overlap_all_sources()
            elif args.crop_mode == "sid-block":
                sid_groups: dict[tuple[str, tuple[int, int, int, int]], list[int]] = defaultdict(list)
                for position, row in enumerate(rows):
                    if position in completed_positions:
                        continue
                    try:
                        source_sid, bands = cropper.sid_source_for_row(int(getattr(row, "reduced_id")))
                        sid_groups[(str(source_sid), tuple(int(value) for value in bands))].append(position)
                    except Exception:
                        crop_one(position, row)
                        checkpoint()
                print(f"  5-band SID crop groups={len(sid_groups):,}", flush=True)
                for group_index, positions in enumerate(sid_groups.values(), start=1):
                    positions = [position for position in positions if position not in completed_positions]
                    if not positions:
                        continue
                    tile_groups: dict[tuple[int, int], list[int]] = defaultdict(list)
                    for position in positions:
                        tile_groups[spatial_tile_key(rows[position])].append(position)
                    if len(tile_groups) > 1:
                        print(
                            f"  SID group {group_index}/{len(sid_groups)}: records={len(positions):,}; "
                            f"spatial tiles={len(tile_groups):,}; tile_metres={args.sid_block_tile_metres:g}",
                            flush=True,
                        )
                    for tile_index, tile_positions in enumerate(tile_groups.values(), start=1):
                        crop_positions_with_block(
                            tile_positions,
                            f"SID group {group_index}/{len(sid_groups)}, tile {tile_index}/{len(tile_groups)}",
                        )
            else:
                for position, row in enumerate(rows):
                    if position in completed_positions:
                        continue
                    crop_one(position, row)
                    checkpoint()
        else:
            print("  All requested crops were seeded from reuse root; skipping SID reads.", flush=True)
        checkpoint(force=True)
    finally:
        crops.flush()
        del crops
        if cropper is not None:
            cropper.close()

    output_metadata = metadata.copy()
    output_metadata["crop_index"] = np.arange(len(output_metadata), dtype=np.int64)
    output_metadata["crop_source"] = "source_raster"
    output_metadata["crop_size"] = int(args.crop_size)
    output_metadata["crop_metres"] = float(args.crop_metres)
    output_metadata["crop_failed"] = crop_failed
    output_metadata["crop_blackout_fraction"] = crop_blackout_fraction
    output_metadata["crop_whiteout_fraction"] = crop_whiteout_fraction
    output_metadata["crop_saturation_fraction"] = crop_saturation_fraction
    output_metadata["crop_valid_fraction"] = crop_valid_fraction
    if cropper is not None:
        for position, row in enumerate(rows):
            if not source_sids[position]:
                try:
                    source_sid, bands = cropper.sid_source_for_row(int(getattr(row, "reduced_id")))
                    set_source(position, str(source_sid), bands)
                except Exception:
                    pass
    output_metadata["crop_source_sid"] = source_sids
    output_metadata["crop_red_band"] = red_bands
    output_metadata["crop_green_band"] = green_bands
    output_metadata["crop_blue_band"] = blue_bands
    output_metadata["crop_nir_band"] = nir_bands
    output_metadata.to_csv(metadata_path, index=False)
    partial_crops_path.replace(crops_path)
    progress_path.unlink(missing_ok=True)
    quality_progress_path.unlink(missing_ok=True)
    config = {
        "join_csv": str(join_csv or ""),
        "metadata_csv": str(compatibility_metadata or ""),
        "fiveband_index_csv": str(index_csv or ""),
        "output_crops": str(crops_path),
        "output_metadata": str(metadata_path),
        "city_token": args.city_token,
        "crop_size": int(args.crop_size),
        "crop_metres": float(args.crop_metres),
        "source_epsg": str(args.source_epsg),
        "band_source": "5band_index_columns" if index_csv is not None else "direct_sid_discovery",
        "crop_mode": str(args.crop_mode),
        "sid_selection_mode": str(args.sid_selection_mode),
        "max_block_gb": float(args.max_block_gb),
        "sid_block_tile_metres": float(args.sid_block_tile_metres),
        "record_count": int(len(output_metadata)),
        "failed_count": int(failed),
        "crop_failed_column": "crop_failed",
    }
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(f"Wrote crops: {crops_path}", flush=True)
    print(f"Wrote metadata: {metadata_path}", flush=True)
    print(f"Wrote config: {config_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
