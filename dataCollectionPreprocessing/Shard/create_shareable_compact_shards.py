#!/usr/bin/env python3
"""Create compact, directly trainable copies of the current model shards.

The source dataset is never modified. Each NPZ is converted one member at a
time so peak memory is bounded by the largest individual array. Companion CSVs
and root configuration/summary files are copied to the output tree.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import tarfile
import time
import zipfile
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_INPUT_ROOT = Path(r"H:\TreeCenteredModelInputs\tree_centered_complete_sharded100k_prism_daily3_clean")
DEFAULT_OUTPUT_ROOT = Path(r"H:\ShareShards")

DROP_ARRAYS = {
    "tree_centered_vegetation_chm_weight",
    # These values are already appended to sentinel_sequence by the current
    # daily-PRISM shard assembler.
    "sentinel_prism_daily",
    "sentinel_prism_daily_columns",
    "sentinel_prism_daily_mask",
}
FLOAT16_ARRAYS = {
    "tree_centered_naip_chm_structure",
    "sentinel_phenology",
    "sentinel_timeseries_match_distance_m",
    "satellite_embedding",
    "satellite_embedding_quality",
    "sentinel_sequence",
    "sentinel_quality",
    "prism_normals",
}
BOOL_ARRAYS = {
    "tree_centered_chm_valid_mask",
    "satellite_embedding_mask",
    "sentinel_sequence_mask",
    "missing_raw_sentinel",
    "missing_prism_daily",
    "missing_prism_normals",
    "used_original_satellite_embedding",
    "used_additional_satellite_embedding",
}
CHM_ARRAYS = {"tree_centered_chm", "tree_centered_vegetation_chm"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--input-pattern", default="*/*_tree_centered_complete_inputs.npz")
    parser.add_argument("--naip-crop-pixels", type=int, default=30)
    parser.add_argument("--chm-scale-metres", type=float, default=0.01, help="uint16 CHM resolution; 0.01 means centimetres.")
    parser.add_argument("--chm-max-metres", type=float, default=100.0)
    parser.add_argument("--compression-level", type=int, default=1, choices=range(0, 10))
    parser.add_argument("--city-token", action="append", default=None)
    parser.add_argument("--archive-cities", action="store_true", help="Also create one .tar.zst archive per city.")
    parser.add_argument("--archive-level", type=int, default=6)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def center_crop(array: np.ndarray, pixels: int) -> np.ndarray:
    if array.ndim < 3:
        raise RuntimeError(f"Expected an image batch with at least 3 dimensions; got {array.shape}")
    height, width = int(array.shape[1]), int(array.shape[2])
    if pixels > min(height, width):
        raise RuntimeError(f"Requested {pixels}px crop exceeds input shape {array.shape}")
    y0, x0 = (height - pixels) // 2, (width - pixels) // 2
    return np.asarray(array[:, y0 : y0 + pixels, x0 : x0 + pixels, ...])


def encode_chm(array: np.ndarray, scale: float, maximum: float) -> np.ndarray:
    values = np.asarray(array, dtype=np.float32)
    values = np.nan_to_num(values, nan=0.0, posinf=maximum, neginf=0.0)
    values = np.clip(values, 0.0, maximum)
    encoded_max = np.iinfo(np.uint16).max
    if maximum / scale > encoded_max:
        raise RuntimeError(f"CHM maximum {maximum} at scale {scale} exceeds uint16 range")
    return np.rint(values / scale).astype(np.uint16)


def transform_array(name: str, value: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    if name == "tree_centered_naip":
        return center_crop(np.asarray(value, dtype=np.uint8), args.naip_crop_pixels)
    if name in CHM_ARRAYS:
        return encode_chm(value, args.chm_scale_metres, args.chm_max_metres)
    if name in BOOL_ARRAYS:
        return np.asarray(value, dtype=bool)
    if name in FLOAT16_ARRAYS:
        return np.asarray(value, dtype=np.float16)
    return np.asarray(value)


def write_npy_member(archive: zipfile.ZipFile, name: str, value: np.ndarray) -> None:
    with archive.open(f"{name}.npy", mode="w", force_zip64=True) as stream:
        np.lib.format.write_array(stream, np.asarray(value), allow_pickle=True)


def convert_npz(source: Path, destination: Path, args: argparse.Namespace) -> dict[str, Any]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".partial")
    if partial.exists():
        partial.unlink()
    compression = zipfile.ZIP_STORED if args.compression_level == 0 else zipfile.ZIP_DEFLATED
    started = time.time()
    dropped: list[str] = []
    transformed: dict[str, dict[str, Any]] = {}
    with np.load(source, allow_pickle=True) as incoming, zipfile.ZipFile(
        partial,
        mode="w",
        compression=compression,
        compresslevel=None if args.compression_level == 0 else args.compression_level,
        allowZip64=True,
    ) as outgoing:
        for name in incoming.files:
            if name in DROP_ARRAYS:
                dropped.append(name)
                continue
            original = incoming[name]
            compact = transform_array(name, original, args)
            write_npy_member(outgoing, name, compact)
            if original.dtype != compact.dtype or original.shape != compact.shape:
                transformed[name] = {
                    "source_dtype": str(original.dtype),
                    "output_dtype": str(compact.dtype),
                    "source_shape": list(original.shape),
                    "output_shape": list(compact.shape),
                }
            del original, compact
        for name in sorted(CHM_ARRAYS):
            if name in incoming.files:
                write_npy_member(outgoing, f"{name}_scale_m", np.asarray([args.chm_scale_metres], dtype=np.float32))
        write_npy_member(outgoing, "compact_shard_format_version", np.asarray([1], dtype=np.int16))
    partial.replace(destination)
    return {
        "source": str(source),
        "output": str(destination),
        "source_bytes": int(source.stat().st_size),
        "output_bytes": int(destination.stat().st_size),
        "ratio": float(destination.stat().st_size / max(source.stat().st_size, 1)),
        "dropped_arrays": dropped,
        "transformed_arrays": transformed,
        "elapsed_seconds": float(time.time() - started),
    }


def verify_shard(source: Path, compact: Path, crop_pixels: int) -> dict[str, Any]:
    with np.load(source, allow_pickle=True) as before, np.load(compact, allow_pickle=True) as after:
        if len(before["tree_centered_naip"]) != len(after["tree_centered_naip"]):
            raise RuntimeError(f"Row count changed: {compact}")
        if tuple(after["tree_centered_naip"].shape[1:3]) != (crop_pixels, crop_pixels):
            raise RuntimeError(f"Unexpected NAIP crop shape in {compact}: {after['tree_centered_naip'].shape}")
        stale = sorted(DROP_ARRAYS.intersection(after.files))
        if stale:
            raise RuntimeError(f"Dropped arrays remain in {compact}: {stale}")
        for name in CHM_ARRAYS:
            if name in before.files:
                scale_key = f"{name}_scale_m"
                if after[name].dtype != np.uint16 or scale_key not in after.files:
                    raise RuntimeError(f"Invalid compact CHM encoding for {name} in {compact}")
                source_values = np.nan_to_num(np.asarray(before[name], dtype=np.float32), nan=0.0, posinf=100.0, neginf=0.0)
                decoded = after[name].astype(np.float32) * float(after[scale_key][0])
                maximum_error = float(np.max(np.abs(np.clip(source_values, 0.0, 100.0) - decoded)))
                if maximum_error > 0.0051:
                    raise RuntimeError(f"CHM round-trip error {maximum_error} exceeds half-centimetre tolerance")
        return {"rows": int(len(after["tree_centered_naip"])), "members": int(len(after.files))}


def copy_companion_files(source_npz: Path, output_dir: Path, force: bool) -> None:
    metadata = source_npz.with_name(source_npz.name.replace("_inputs.npz", "_metadata.csv"))
    if not metadata.exists():
        raise FileNotFoundError(metadata)
    destination = output_dir / metadata.name
    if destination.exists() and not force:
        raise FileExistsError(destination)
    shutil.copy2(metadata, destination)


def archive_city(city_dir: Path, output_root: Path, level: int) -> Path:
    try:
        import zstandard
    except ModuleNotFoundError as error:
        raise SystemExit("--archive-cities requires zstandard: python -m pip install zstandard") from error
    destination = output_root / f"{city_dir.name}.tar.zst"
    partial = destination.with_suffix(destination.suffix + ".partial")
    with partial.open("wb") as raw:
        compressor = zstandard.ZstdCompressor(level=int(level), threads=-1)
        with compressor.stream_writer(raw) as compressed:
            with tarfile.open(fileobj=compressed, mode="w|") as archive:
                archive.add(city_dir, arcname=city_dir.name)
    partial.replace(destination)
    if destination.stat().st_size <= 0:
        raise RuntimeError(f"Empty archive: {destination}")
    return destination


def copy_root_documents(input_root: Path, output_root: Path, force: bool) -> None:
    patterns = ("*.json", "*.csv", "*.md", "*.txt")
    for pattern in patterns:
        for source in input_root.glob(pattern):
            destination = output_root / source.name
            if destination.exists() and not force:
                continue
            shutil.copy2(source, destination)


def main() -> int:
    args = parse_args()
    if not args.input_root.is_dir():
        raise SystemExit(f"Input root not found: {args.input_root}")
    if args.input_root.resolve() == args.output_root.resolve():
        raise SystemExit("Input and output roots must differ.")
    if args.naip_crop_pixels < 1 or args.chm_scale_metres <= 0 or args.chm_max_metres <= 0:
        raise SystemExit("Crop size, CHM scale, and CHM maximum must be positive.")
    selected = None if args.city_token is None else {value.strip().lower() for value in args.city_token if value.strip()}
    shards = [path for path in sorted(args.input_root.glob(args.input_pattern)) if selected is None or path.parent.name.lower() in selected]
    if not shards:
        raise SystemExit("No source shards selected.")
    print(f"Selected {len(shards):,} shard(s); output={args.output_root}", flush=True)
    if args.dry_run:
        for source in shards:
            print(f"Would compact {source} -> {args.output_root / source.parent.name / source.name}")
        return 0
    args.output_root.mkdir(parents=True, exist_ok=True)
    copy_root_documents(args.input_root, args.output_root, args.force)
    reports: list[dict[str, Any]] = []
    by_city: dict[str, list[Path]] = {}
    for index, source in enumerate(shards, start=1):
        city = source.parent.name
        output_dir = args.output_root / city
        destination = output_dir / source.name
        if destination.exists() and not args.force:
            print(f"[{index}/{len(shards)}] exists; verifying {destination}", flush=True)
            report = {"source": str(source), "output": str(destination), "status": "existing"}
        else:
            print(f"[{index}/{len(shards)}] compacting {source.name}", flush=True)
            report = convert_npz(source, destination, args)
            report["status"] = "completed"
            copy_companion_files(source, output_dir, args.force)
        report.update(verify_shard(source, destination, args.naip_crop_pixels))
        reports.append(report)
        by_city.setdefault(city, []).append(destination)
        (args.output_root / "compact_shard_conversion_report.json").write_text(
            json.dumps(reports, indent=2), encoding="utf-8"
        )
        if "source_bytes" in report:
            print(
                f"  {report['source_bytes']/1e9:.2f} GB -> {report['output_bytes']/1e9:.2f} GB "
                f"({report['ratio']:.1%}); rows={report['rows']:,}",
                flush=True,
            )
    if args.archive_cities:
        for city in sorted(by_city):
            city_dir = args.output_root / city
            archive = archive_city(city_dir, args.output_root, args.archive_level)
            print(f"Archived {city}: {archive}", flush=True)
    totals = {
        "source_bytes": int(sum(row.get("source_bytes", 0) for row in reports)),
        "output_bytes": int(sum(row.get("output_bytes", 0) for row in reports)),
        "shards": len(reports),
        "naip_crop_pixels": args.naip_crop_pixels,
        "chm_scale_metres": args.chm_scale_metres,
    }
    (args.output_root / "compact_shard_summary.json").write_text(json.dumps(totals, indent=2), encoding="utf-8")
    print(f"Finished {len(reports):,} shard(s).", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
