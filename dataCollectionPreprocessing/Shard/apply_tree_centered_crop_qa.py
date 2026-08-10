#!/usr/bin/env python3
"""Apply QA flags/screens to tree-centered crown crops.

Hard exclusions are intentionally limited to records that are unsuitable for
model evaluation or training, such as failed crops, low center vegetation,
insufficient CHM/LiDAR coverage in the center window, or inventory rows that
share an exact coordinate with another record. Other potential issues are
retained as flags so they can be used for uncertainty reporting and stratified
metrics.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path

import numpy as np
import pandas as pd


CLEAN_ROOT = Path(r"H:\TreeCenteredModelInputs")
DEFAULT_CROP_ROOT = CLEAN_ROOT / "tree_centered_naip_crops_clean"
DEFAULT_STRUCTURE_DIR = CLEAN_ROOT / "tree_centered_chm_structure_clean"
DEFAULT_LIDAR_CHM_ROOT = CLEAN_ROOT / "tree_centered_lidar_products_clean" / "CHM"
DEFAULT_SENTINEL_PHENOLOGY_DIR = CLEAN_ROOT / "tree_centered_sentinel_phenology_clean"
DEFAULT_GEE_DIR = CLEAN_ROOT / "tree_centered_gee_inputs_clean"
DEFAULT_OUTPUT_DIR = CLEAN_ROOT / "tree_centered_crop_qa_clean"


STALE_CELL_CENTERED_QA_COLUMNS = {
    "screen_fail_center_naip_vegetation",
    "screen_fail_naip_saturation",
    "screen_fail_chm_tree_evidence",
    "enhanced_screen_fail_center_canopy",
    "enhanced_screening_exclusion_reason",
    "crop_failed",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--crop-root", type=Path, default=DEFAULT_CROP_ROOT)
    parser.add_argument("--structure-dir", type=Path, default=DEFAULT_STRUCTURE_DIR)
    parser.add_argument("--lidar-chm-root", type=Path, default=DEFAULT_LIDAR_CHM_ROOT)
    parser.add_argument(
        "--sentinel-phenology-dir",
        type=Path,
        default=DEFAULT_SENTINEL_PHENOLOGY_DIR,
        help=(
            "Sentinel phenology sidecar root. Rows with missing_sentinel_phenology "
            "are hard-excluded from model use."
        ),
    )
    parser.add_argument(
        "--gee-dir",
        type=Path,
        default=DEFAULT_GEE_DIR,
        help=(
            "Clean GEE/Satlas embedding sidecar root. Rows with missing_satellite_embedding "
            "are hard-excluded from model use."
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--city-token", action="append", default=None)
    parser.add_argument("--exclude-city-token", action="append", default=[])
    parser.add_argument("--metadata-pattern", default="*_tree_id_centered_nearest_64px_metadata.csv")
    parser.add_argument("--crop-npy-pattern", default="*_tree_id_centered_nearest_64px_rgbnir_crops.npy")
    parser.add_argument("--structure-suffix", default="_tree_id_centered_chm_structure_metrics.npz")
    parser.add_argument("--lidar-chm-pattern", default="*_tree_id_centered_nearest_64px_chm.npy")
    parser.add_argument("--sentinel-phenology-suffix", default="_tree_id_centered_sentinel_phenology.npz")
    parser.add_argument("--gee-sidecar-pattern", default="*_tree_id_centered_gee_inputs.npz")
    parser.add_argument(
        "--prefer-lidar-chm-product",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Prefer standalone true tree-centered LiDAR CHM products over CHM arrays in the structure sidecar.",
    )
    parser.add_argument(
        "--require-lidar-chm-product",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip cities without true tree-centered LiDAR CHM products.",
    )
    parser.add_argument("--center-metres", type=float, default=10.0)
    parser.add_argument("--ndvi-threshold", type=float, default=0.20)
    parser.add_argument("--low-center-vegetation-threshold", type=float, default=0.10)
    parser.add_argument("--low-height-threshold-m", type=float, default=2.00)
    parser.add_argument("--lidar-coverage-threshold", type=float, default=0.50)
    parser.add_argument("--min-vegetated-height-fraction", type=float, default=0.02)
    parser.add_argument("--max-saturation-fraction", type=float, default=0.25)
    parser.add_argument(
        "--crop-quality-chunk-size",
        type=int,
        default=50000,
        help="Rows per chunk when recomputing crop coverage/saturation directly from tree-centered NAIP crops.",
    )
    parser.add_argument(
        "--min-crop-valid-fraction",
        type=float,
        default=0.01,
        help=(
            "Treat a crop as failed only when its valid-pixel fraction is at or below this value. "
            "The raw crop_failed metadata column from upstream crop files is not trusted because "
            "older metadata can contain stale failure flags."
        ),
    )
    parser.add_argument(
        "--parallel-workers",
        type=int,
        default=1,
        help="Number of cities to process concurrently. Keep modest because each city reads crop and CHM arrays.",
    )
    parser.add_argument(
        "--fail-fast",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Stop after first failed city in serial mode. In parallel mode, already-started cities finish.",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def reject_non_clean_roots(args: argparse.Namespace) -> None:
    checked = {
        "crop_root": args.crop_root,
        "structure_dir": args.structure_dir,
        "lidar_chm_root": args.lidar_chm_root,
        "output_dir": args.output_dir,
    }
    if args.sentinel_phenology_dir is not None:
        checked["sentinel_phenology_dir"] = args.sentinel_phenology_dir
    if args.gee_dir is not None:
        checked["gee_dir"] = args.gee_dir
    clean_root_text = str(CLEAN_ROOT).lower()
    for label, path in checked.items():
        resolved_text = str(Path(path).resolve()).lower()
        if not resolved_text.startswith(clean_root_text):
            raise SystemExit(
                f"{label} must point inside the standalone clean tree-centered root "
                f"({CLEAN_ROOT}); got {path}"
            )


def find_sentinel_phenology_path(args: argparse.Namespace, city: str) -> Path | None:
    if args.sentinel_phenology_dir is None:
        return None
    root = Path(args.sentinel_phenology_dir)
    candidates = [
        root / city / f"{city}{args.sentinel_phenology_suffix}",
        root / f"{city}{args.sentinel_phenology_suffix}",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def find_gee_path(args: argparse.Namespace, city: str) -> Path | None:
    if args.gee_dir is None:
        return None
    root = Path(args.gee_dir)
    candidates = sorted((root / city).glob(args.gee_sidecar_pattern)) if (root / city).exists() else []
    candidates.extend(sorted(root.glob(f"{city}{args.gee_sidecar_pattern.lstrip('*')}")))
    return candidates[0] if candidates else None


def discover_cities(args: argparse.Namespace) -> dict[str, tuple[Path, Path, Path | None, Path | None, Path | None, Path | None]]:
    selected = None if args.city_token is None else {token.strip().lower() for token in args.city_token if token.strip()}
    excluded = {token.strip().lower() for token in args.exclude_city_token}
    out: dict[str, tuple[Path, Path, Path | None, Path | None, Path | None, Path | None]] = {}
    for city_dir in sorted(path for path in args.crop_root.iterdir() if path.is_dir()):
        city = city_dir.name.strip().lower()
        if selected is not None and city not in selected:
            continue
        if city in excluded:
            continue
        metadata = sorted(city_dir.glob(args.metadata_pattern))
        crops = sorted(city_dir.glob(args.crop_npy_pattern))
        structure = args.structure_dir / f"{city}{args.structure_suffix}"
        lidar_chm_dir = args.lidar_chm_root / city
        lidar_chm_matches = sorted(lidar_chm_dir.glob(args.lidar_chm_pattern)) if lidar_chm_dir.exists() else []
        lidar_chm = lidar_chm_matches[0] if lidar_chm_matches else None
        sentinel_phenology = find_sentinel_phenology_path(args, city)
        gee = find_gee_path(args, city)
        has_required_chm = lidar_chm is not None or not args.require_lidar_chm_product
        if not metadata or not crops or not has_required_chm or (not structure.exists() and lidar_chm is None):
            print(
                f"SKIP {city}: metadata={bool(metadata)} crops={bool(crops)} "
                f"lidar_chm={lidar_chm is not None} structure={structure.exists()}",
                flush=True,
            )
            continue
        out[city] = (metadata[0], crops[0], structure if structure.exists() else None, lidar_chm, sentinel_phenology, gee)
    return out


def center_slice(size: int, crop_metres: float, center_metres: float) -> tuple[slice, slice]:
    pixels = max(1, int(round(float(size) * float(center_metres) / float(crop_metres))))
    pixels = min(size, pixels)
    start = (size - pixels) // 2
    end = start + pixels
    return slice(start, end), slice(start, end)


def compute_ndvi(chips: np.ndarray) -> np.ndarray:
    red = chips[..., 0].astype(np.float32)
    nir = chips[..., 3].astype(np.float32)
    denom = nir + red
    return np.divide(nir - red, denom, out=np.zeros_like(nir, dtype=np.float32), where=denom > 0)


def compute_crop_quality(chips: np.ndarray, chunk_size: int) -> dict[str, np.ndarray]:
    count = int(chips.shape[0])
    chunk_size = max(1, int(chunk_size))
    blackout_fraction = np.zeros(count, dtype=np.float32)
    whiteout_fraction = np.zeros(count, dtype=np.float32)
    saturation_fraction = np.zeros(count, dtype=np.float32)
    valid_fraction = np.zeros(count, dtype=np.float32)
    for start in range(0, count, chunk_size):
        end = min(count, start + chunk_size)
        chunk = np.asarray(chips[start:end])
        blackout = np.all(chunk == 0, axis=3)
        whiteout = np.all(chunk == 255, axis=3)
        saturated = blackout | whiteout
        blackout_fraction[start:end] = blackout.mean(axis=(1, 2))
        whiteout_fraction[start:end] = whiteout.mean(axis=(1, 2))
        saturation_fraction[start:end] = saturated.mean(axis=(1, 2))
        valid_fraction[start:end] = (~saturated).mean(axis=(1, 2))
    return {
        "crop_blackout_fraction": blackout_fraction,
        "crop_whiteout_fraction": whiteout_fraction,
        "crop_saturation_fraction": saturation_fraction,
        "crop_valid_fraction": valid_fraction,
    }


def duplicate_inventory_coordinate_metrics(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    count = len(frame)
    empty = {
        "inventory_coordinate_reuse_count": np.ones(count, dtype=np.int32),
        "inventory_coordinate_taxon_count": np.ones(count, dtype=np.int16),
        "inventory_coordinate_species_count": np.ones(count, dtype=np.int16),
        "qa_flag_duplicate_inventory_coordinate": np.zeros(count, dtype=bool),
    }
    if "longitude" not in frame.columns or "latitude" not in frame.columns:
        return empty
    coords = frame[["longitude", "latitude"]].apply(pd.to_numeric, errors="coerce")
    valid = coords["longitude"].notna() & coords["latitude"].notna()
    if not valid.any():
        return empty
    work = coords.loc[valid].copy()
    work["_row"] = np.flatnonzero(valid.to_numpy(dtype=bool))
    work["_taxon"] = frame.loc[valid, "taxon_label"].fillna("").astype(str) if "taxon_label" in frame.columns else ""
    work["_species"] = frame.loc[valid, "scientific_name"].fillna("").astype(str) if "scientific_name" in frame.columns else ""
    grouped = (
        work.groupby(["longitude", "latitude"], dropna=False)
        .agg(
            inventory_coordinate_reuse_count=("_row", "size"),
            inventory_coordinate_taxon_count=("_taxon", "nunique"),
            inventory_coordinate_species_count=("_species", "nunique"),
        )
        .reset_index()
    )
    work = work.merge(grouped, on=["longitude", "latitude"], how="left", validate="many_to_one")
    rows = work["_row"].to_numpy(dtype=np.int64)
    empty["inventory_coordinate_reuse_count"][rows] = work["inventory_coordinate_reuse_count"].to_numpy(dtype=np.int32)
    empty["inventory_coordinate_taxon_count"][rows] = work["inventory_coordinate_taxon_count"].to_numpy(dtype=np.int16)
    empty["inventory_coordinate_species_count"][rows] = work["inventory_coordinate_species_count"].to_numpy(dtype=np.int16)
    duplicate = work["inventory_coordinate_reuse_count"].to_numpy(dtype=np.int32) >= 2
    empty["qa_flag_duplicate_inventory_coordinate"][rows] = duplicate
    return empty


def load_missing_sentinel_phenology(
    city: str,
    metadata: pd.DataFrame,
    phenology_path: Path | None,
    phenology_required: bool,
) -> tuple[np.ndarray, str]:
    if phenology_path is None:
        if phenology_required:
            print(
                f"{city}: WARNING missing Sentinel phenology sidecar; marking all rows as missing Sentinel phenology",
                flush=True,
            )
            return np.ones(len(metadata), dtype=bool), ""
        return np.zeros(len(metadata), dtype=bool), ""
    with np.load(phenology_path, allow_pickle=False) as data:
        if "missing_sentinel_phenology" not in data:
            raise RuntimeError(f"{city}: {phenology_path} is missing missing_sentinel_phenology")
        missing_raw = np.asarray(data["missing_sentinel_phenology"], dtype=bool)
        if "tree_id" in data.files and "tree_id" in metadata.columns:
            source_tree_ids = np.asarray(data["tree_id"], dtype=np.int64)
            metadata_tree_ids = pd.to_numeric(metadata["tree_id"], errors="coerce").fillna(-1).astype(np.int64).to_numpy()
            if len(source_tree_ids) == len(metadata_tree_ids) and np.array_equal(source_tree_ids, metadata_tree_ids):
                return missing_raw.copy(), str(phenology_path)
            lookup = {int(tree_id): int(row) for row, tree_id in enumerate(source_tree_ids)}
            missing = np.ones(len(metadata), dtype=bool)
            for row, tree_id in enumerate(metadata_tree_ids):
                source_row = lookup.get(int(tree_id))
                if source_row is not None and 0 <= source_row < len(missing_raw):
                    missing[row] = bool(missing_raw[source_row])
            return missing, str(phenology_path)
        if len(missing_raw) != len(metadata):
            raise RuntimeError(
                f"{city}: Sentinel phenology rows differ from metadata rows "
                f"({len(missing_raw):,} vs {len(metadata):,}) and no tree_id alignment is available"
            )
        return missing_raw.copy(), str(phenology_path)


def load_missing_gee_embedding(
    city: str,
    metadata: pd.DataFrame,
    gee_path: Path | None,
    gee_required: bool,
) -> tuple[np.ndarray, str]:
    if gee_path is None:
        if gee_required:
            print(
                f"{city}: WARNING missing GEE embedding sidecar; marking all rows as missing GEE embedding",
                flush=True,
            )
            return np.ones(len(metadata), dtype=bool), ""
        return np.zeros(len(metadata), dtype=bool), ""
    with np.load(gee_path, allow_pickle=False) as data:
        if "missing_satellite_embedding" in data.files:
            missing_raw = np.asarray(data["missing_satellite_embedding"], dtype=bool)
        elif "satellite_embedding_mask" in data.files:
            mask = np.asarray(data["satellite_embedding_mask"], dtype=bool)
            missing_raw = ~mask.reshape(mask.shape[0], -1).any(axis=1)
        else:
            raise RuntimeError(
                f"{city}: {gee_path} is missing missing_satellite_embedding or satellite_embedding_mask"
            )

        metadata_tree_ids = (
            pd.to_numeric(metadata["tree_id"], errors="coerce").fillna(-1).astype(np.int64).to_numpy()
            if "tree_id" in metadata.columns
            else None
        )
        for id_key in ("tree_id",):
            if id_key not in data.files or metadata_tree_ids is None:
                continue
            source_tree_ids = np.asarray(data[id_key], dtype=np.int64)
            if len(source_tree_ids) == len(metadata_tree_ids) and np.array_equal(source_tree_ids, metadata_tree_ids):
                return missing_raw.copy(), str(gee_path)
            lookup = {int(tree_id): int(row) for row, tree_id in enumerate(source_tree_ids)}
            missing = np.ones(len(metadata), dtype=bool)
            for row, tree_id in enumerate(metadata_tree_ids):
                source_row = lookup.get(int(tree_id))
                if source_row is not None and 0 <= source_row < len(missing_raw):
                    missing[row] = bool(missing_raw[source_row])
            return missing, str(gee_path)

        if len(missing_raw) != len(metadata):
            raise RuntimeError(
                f"{city}: GEE rows differ from metadata rows "
                f"({len(missing_raw):,} vs {len(metadata):,}) and no tree_id alignment is available"
            )
        return missing_raw.copy(), str(gee_path)


def lidar_index_path_for_chm(chm_path: Path) -> Path:
    return chm_path.with_name(chm_path.name.replace("_chm.npy", "_lidar_index.csv"))


def load_tree_centered_chm(
    city: str,
    metadata: pd.DataFrame,
    lidar_chm_path: Path,
) -> tuple[np.ndarray, np.ndarray, str, np.ndarray]:
    """Load true tree-centered CHM aligned to metadata rows.

    Some cities may have a valid city-level CHM product while individual records
    are missing from the product. Those rows are filled with NaN CHM and marked
    by the returned missing mask instead of failing the whole city.
    """

    source_tree_ids: np.ndarray | None = None
    if lidar_chm_path.suffix.lower() == ".npz":
        with np.load(lidar_chm_path, allow_pickle=False) as data:
            if "tree_centered_chm" not in data:
                raise RuntimeError(f"{city}: {lidar_chm_path} is missing tree_centered_chm")
            chm_raw = np.asarray(data["tree_centered_chm"], dtype=np.float32)
            chm_valid_raw = (
                np.asarray(data["tree_centered_chm_valid_mask"], dtype=bool)
                if "tree_centered_chm_valid_mask" in data
                else np.isfinite(chm_raw)
            )
            if "tree_id" in data:
                source_tree_ids = np.asarray(data["tree_id"], dtype=np.int64)
    else:
        chm_raw = np.load(lidar_chm_path, mmap_mode="r")
        chm_valid_raw = np.isfinite(chm_raw)
    if chm_raw.ndim != 3:
        raise RuntimeError(f"{city}: expected 3D CHM array at {lidar_chm_path}, got shape={chm_raw.shape}")
    if chm_raw.shape[0] == len(metadata):
        if source_tree_ids is None or "tree_id" not in metadata.columns:
            chm = np.asarray(chm_raw, dtype=np.float32)
            missing = ~np.asarray(chm_valid_raw, dtype=bool).any(axis=(1, 2))
            return chm, np.asarray(chm_valid_raw, dtype=bool), str(lidar_chm_path), missing
        metadata_tree_ids = pd.to_numeric(metadata["tree_id"], errors="coerce").fillna(-1).astype(np.int64).to_numpy()
        if np.array_equal(source_tree_ids, metadata_tree_ids):
            chm = np.asarray(chm_raw, dtype=np.float32)
            missing = ~np.asarray(chm_valid_raw, dtype=bool).any(axis=(1, 2))
            return chm, np.asarray(chm_valid_raw, dtype=bool), str(lidar_chm_path), missing

    if "tree_id" not in metadata.columns:
        raise RuntimeError(f"{city}: CHM rows differ from metadata rows and metadata lacks tree_id")
    if source_tree_ids is not None:
        row_lookup = {int(tree_id): int(row) for row, tree_id in enumerate(source_tree_ids) if int(tree_id) >= 0}
    else:
        index_path = lidar_index_path_for_chm(lidar_chm_path)
        if not index_path.exists():
            index_path = lidar_chm_path.with_name(lidar_chm_path.name.replace("_chm_products.npz", "_lidar_index.csv"))
    if source_tree_ids is None and not index_path.exists():
        raise RuntimeError(
            f"{city}: CHM rows differ from metadata rows ({chm_raw.shape[0]:,} vs {len(metadata):,}) "
            f"and missing LiDAR index: {index_path}"
        )
    if source_tree_ids is None:
        index = pd.read_csv(index_path, low_memory=False)
        if "tree_id" not in index.columns:
            raise RuntimeError(f"{city}: {index_path} is missing tree_id")
        index = index.reset_index().rename(columns={"index": "chm_row"})
        index["tree_id"] = pd.to_numeric(index["tree_id"], errors="coerce")
        index = index.dropna(subset=["tree_id"]).copy()
        index["tree_id"] = index["tree_id"].astype(np.int64)
        index = index.drop_duplicates("tree_id", keep="first")
        row_lookup = dict(zip(index["tree_id"].tolist(), index["chm_row"].tolist()))

    out_shape = (len(metadata), int(chm_raw.shape[1]), int(chm_raw.shape[2]))
    chm = np.full(out_shape, np.nan, dtype=np.float32)
    chm_valid = np.zeros(out_shape, dtype=bool)
    missing = np.ones(len(metadata), dtype=bool)
    tree_ids = pd.to_numeric(metadata["tree_id"], errors="coerce").fillna(-1).astype(np.int64).to_numpy()
    for out_row, tree_id in enumerate(tree_ids):
        chm_row = row_lookup.get(int(tree_id))
        if chm_row is None or chm_row < 0 or chm_row >= chm_raw.shape[0]:
            continue
        chm[out_row] = np.asarray(chm_raw[int(chm_row)], dtype=np.float32)
        chm_valid[out_row] = np.asarray(chm_valid_raw[int(chm_row)], dtype=bool)
        missing[out_row] = not chm_valid[out_row].any()
    return chm, chm_valid, str(lidar_chm_path), missing


def process_city(
    city: str,
    paths: tuple[Path, Path, Path | None, Path | None, Path | None, Path | None],
    args: argparse.Namespace,
) -> dict[str, object]:
    metadata_path, crop_path, structure_path, lidar_chm_path, sentinel_phenology_path, gee_path = paths
    output_path = args.output_dir / f"{city}_tree_centered_qa_metadata.csv"
    if output_path.exists() and not args.force:
        print(f"{city}: QA exists; skipping {output_path}", flush=True)
        return {"city_token": city, "status": "skipped_exists"}

    metadata = pd.read_csv(metadata_path, low_memory=False)
    chips = np.load(crop_path, mmap_mode="r")
    use_lidar_chm = bool(args.prefer_lidar_chm_product and lidar_chm_path is not None)
    if use_lidar_chm:
        chm, chm_valid, chm_source, missing_lidar_chm_row = load_tree_centered_chm(city, metadata, lidar_chm_path)
    else:
        if structure_path is None:
            raise RuntimeError(f"{city}: no structure sidecar or tree-centered LiDAR CHM product available")
        with np.load(structure_path, allow_pickle=False) as data:
            chm = np.asarray(data["tree_centered_chm"], dtype=np.float32)
            chm_valid = np.asarray(data["tree_centered_chm_valid_mask"], dtype=bool)
        chm_source = str(structure_path)
        missing_lidar_chm_row = ~np.isfinite(chm).any(axis=(1, 2))
    if len(metadata) != chips.shape[0] or len(metadata) != chm.shape[0]:
        raise RuntimeError(
            f"{city}: row mismatch metadata={len(metadata):,} crops={chips.shape[0]:,} structure={chm.shape[0]:,}"
        )

    crop_metres = float(metadata["crop_metres"].iloc[0]) if "crop_metres" in metadata.columns else 38.0
    y_slice, x_slice = center_slice(chips.shape[1], crop_metres, args.center_metres)
    ndvi_center = compute_ndvi(np.asarray(chips[:, y_slice, x_slice, :]))
    center_veg_fraction = (ndvi_center >= float(args.ndvi_threshold)).mean(axis=(1, 2))

    chm_y, chm_x = center_slice(chm.shape[1], crop_metres, args.center_metres)
    center_chm = chm[:, chm_y, chm_x]
    center_chm_valid = chm_valid[:, chm_y, chm_x]
    center_lidar_coverage = center_chm_valid.mean(axis=(1, 2))
    center_veg_small = ndvi_center >= float(args.ndvi_threshold)
    if center_veg_small.shape[1:] != center_chm.shape[1:]:
        # Downsample vegetation evidence to the CHM center grid by nearest index.
        yy = np.linspace(0, center_veg_small.shape[1] - 1, center_chm.shape[1]).round().astype(int)
        xx = np.linspace(0, center_veg_small.shape[2] - 1, center_chm.shape[2]).round().astype(int)
        center_veg_small = center_veg_small[:, yy][:, :, xx]
    vegetated_height = center_veg_small & center_chm_valid & (center_chm >= float(args.low_height_threshold_m))
    vegetated_height_fraction = vegetated_height.mean(axis=(1, 2))

    crop_quality = compute_crop_quality(chips, args.crop_quality_chunk_size)
    saturation = crop_quality["crop_saturation_fraction"]
    crop_valid_fraction = crop_quality["crop_valid_fraction"]
    low_center_vegetation = center_veg_fraction < float(args.low_center_vegetation_threshold)
    sufficient_lidar = center_lidar_coverage >= float(args.lidar_coverage_threshold)
    insufficient_lidar_for_height = ~sufficient_lidar
    low_vegetated_height = sufficient_lidar & (vegetated_height_fraction < float(args.min_vegetated_height_fraction))
    high_saturation = saturation > float(args.max_saturation_fraction)
    crop_failed = crop_valid_fraction <= float(args.min_crop_valid_fraction)
    duplicate_metrics = duplicate_inventory_coordinate_metrics(metadata)
    duplicate_inventory_coordinate = duplicate_metrics["qa_flag_duplicate_inventory_coordinate"]
    missing_sentinel_phenology, sentinel_phenology_source = load_missing_sentinel_phenology(
        city,
        metadata,
        sentinel_phenology_path,
        args.sentinel_phenology_dir is not None,
    )
    missing_gee_embedding, gee_source = load_missing_gee_embedding(
        city,
        metadata,
        gee_path,
        args.gee_dir is not None,
    )

    # This file is the standalone tree-centered QA boundary. Do not propagate
    # old cell-centered screening fields or stale crop_failed values into
    # downstream products.
    out = metadata.drop(columns=[c for c in STALE_CELL_CENTERED_QA_COLUMNS if c in metadata.columns]).copy()
    for column, values in crop_quality.items():
        out[column] = values
    out["qa_center_veg_fraction_ndvi_ge_threshold"] = center_veg_fraction.astype(np.float32)
    out["qa_center_lidar_coverage_fraction"] = center_lidar_coverage.astype(np.float32)
    out["qa_center_vegetated_height_fraction"] = vegetated_height_fraction.astype(np.float32)
    out["qa_flag_low_center_vegetation"] = low_center_vegetation
    out["qa_flag_insufficient_lidar_coverage_for_height"] = insufficient_lidar_for_height
    out["qa_flag_low_vegetated_height_with_lidar"] = low_vegetated_height
    out["qa_flag_high_crop_saturation"] = high_saturation
    out["qa_flag_crop_failed"] = crop_failed
    out["qa_flag_missing_lidar_chm"] = missing_lidar_chm_row
    out["qa_flag_missing_sentinel_phenology"] = missing_sentinel_phenology
    out["qa_flag_missing_gee_embedding"] = missing_gee_embedding
    for column, values in duplicate_metrics.items():
        out[column] = values
    out["qa_lidar_chm_source"] = chm_source
    out["qa_sentinel_phenology_source"] = sentinel_phenology_source
    out["qa_gee_embedding_source"] = gee_source
    out["qa_lidar_chm_pixels"] = int(chm.shape[1])
    hard_exclude = (
        low_center_vegetation
        | insufficient_lidar_for_height
        | crop_failed
        | duplicate_inventory_coordinate
        | missing_sentinel_phenology
        | missing_gee_embedding
    )
    out["qa_exclude_from_model"] = hard_exclude
    out["qa_exclude_from_train_val"] = hard_exclude
    out["qa_any_warning_flag"] = (
        low_center_vegetation
        | insufficient_lidar_for_height
        | low_vegetated_height
        | high_saturation
        | crop_failed
        | missing_lidar_chm_row
        | duplicate_inventory_coordinate
        | missing_sentinel_phenology
        | missing_gee_embedding
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False)

    summary = {
        "city_token": city,
        "status": "completed",
        "rows": int(len(out)),
        "exclude_train_val": int(out["qa_exclude_from_train_val"].sum()),
        "low_center_vegetation": int(low_center_vegetation.sum()),
        "insufficient_lidar_coverage_for_height": int(insufficient_lidar_for_height.sum()),
        "missing_lidar_chm": int(missing_lidar_chm_row.sum()),
        "low_vegetated_height_with_lidar": int(low_vegetated_height.sum()),
        "high_crop_saturation": int(high_saturation.sum()),
        "crop_failed": int(crop_failed.sum()),
        "duplicate_inventory_coordinate": int(duplicate_inventory_coordinate.sum()),
        "missing_sentinel_phenology": int(missing_sentinel_phenology.sum()),
        "missing_gee_embedding": int(missing_gee_embedding.sum()),
        "mean_center_veg_fraction": float(np.mean(center_veg_fraction)),
        "mean_lidar_coverage": float(np.mean(center_lidar_coverage)),
        "lidar_chm_source": chm_source,
        "sentinel_phenology_source": sentinel_phenology_source,
        "gee_embedding_source": gee_source,
        "lidar_chm_pixels": int(chm.shape[1]),
    }
    print(
        f"{city}: rows={summary['rows']:,}; exclude_train_val={summary['exclude_train_val']:,} "
        f"({summary['exclude_train_val']/max(summary['rows'],1):.2%}); warnings={int(out['qa_any_warning_flag'].sum()):,}",
        flush=True,
    )
    return summary


def merge_city_rows(
    path: Path,
    new_rows: list[dict[str, object]],
    processed_cities: set[str],
    columns: list[str] | None = None,
) -> None:
    if path.exists():
        existing = pd.read_csv(path)
        if "city_token" in existing.columns:
            existing = existing[~existing["city_token"].astype(str).str.lower().isin(processed_cities)].copy()
    else:
        existing = pd.DataFrame(columns=columns or [])
    additions = pd.DataFrame(new_rows, columns=columns) if columns else pd.DataFrame(new_rows)
    merged = pd.concat([existing, additions], ignore_index=True, sort=False)
    if "city_token" in merged.columns and len(merged):
        merged = merged.sort_values("city_token").reset_index(drop=True)
    if columns:
        for column in columns:
            if column not in merged.columns:
                merged[column] = pd.Series(dtype="object")
        merged = merged[columns]
    merged.to_csv(path, index=False)


def main() -> int:
    args = parse_args()
    reject_non_clean_roots(args)
    cities = discover_cities(args)
    if not cities:
        raise SystemExit("No complete city crop/structure products found.")
    worker_count = max(1, min(int(args.parallel_workers), len(cities)))
    print(f"Processing {len(cities):,} city QA job(s) with parallel_workers={worker_count}.", flush=True)
    summaries = []
    failures: list[tuple[str, str]] = []
    if worker_count == 1:
        for city, paths in cities.items():
            try:
                summaries.append(process_city(city, paths, args))
            except Exception as error:
                print(f"{city}: ERROR: {error}", flush=True)
                failures.append((city, str(error)))
                if args.fail_fast:
                    break
    else:
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            future_to_city = {
                executor.submit(process_city, city, paths, args): city
                for city, paths in cities.items()
            }
            for future in as_completed(future_to_city):
                city = future_to_city[future]
                try:
                    summaries.append(future.result())
                except Exception as error:
                    print(f"{city}: ERROR: {error}", flush=True)
                    failures.append((city, str(error)))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    processed_cities = set(cities.keys())
    summary_rows = [row for row in summaries if row.get("status") != "skipped_exists"]
    summary_cities = {str(row["city_token"]).lower() for row in summary_rows if "city_token" in row}
    summary_path = args.output_dir / "tree_centered_qa_summary.csv"
    if summary_path.exists() or summary_rows:
        merge_city_rows(summary_path, summary_rows, summary_cities)
    failure_path = args.output_dir / "tree_centered_qa_failures.csv"
    failure_rows = [{"city_token": city, "error": error} for city, error in failures]
    if failure_path.exists() or failure_rows:
        merge_city_rows(failure_path, failure_rows, processed_cities, columns=["city_token", "error"])
    if failures:
        print(f"Wrote QA failures: {failure_path}", flush=True)
    config_path = args.output_dir / "tree_centered_qa_config.json"
    config_path.write_text(json.dumps(vars(args), indent=2, default=str), encoding="utf-8")
    print(f"Wrote QA summary: {summary_path}", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
