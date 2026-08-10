#!/usr/bin/env python3
"""Derive CHM/structure sidecars for the clean tree_id-centered dataset.

This consumes the clean standalone products:

* NAIP crops and metadata:
  ``H:/TreeCenteredModelInputs/tree_centered_naip_crops_clean/<city>/``
* LiDAR CHM products:
  ``H:/TreeCenteredModelInputs/tree_centered_lidar_products_clean/CHM/<city>/``

Outputs one row-aligned sidecar per city under:
``H:/TreeCenteredModelInputs/tree_centered_chm_structure_clean``.

Rows are aligned by ``tree_id`` when the LiDAR index contains it.  For older
clean LiDAR products generated before ``tree_id`` was added to the index, rows
fall back to ``crop_index`` / ``tree_centered_index`` alignment.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path

import numpy as np
import pandas as pd


import naip_chm_structure_metrics as derived_metrics


DEFAULT_CROP_ROOT = Path(r"H:\TreeCenteredModelInputs\tree_centered_naip_crops_clean")
DEFAULT_LIDAR_PRODUCT_ROOT = Path(r"H:\TreeCenteredModelInputs\tree_centered_lidar_products_clean")
DEFAULT_OUTPUT_DIR = Path(r"H:\TreeCenteredModelInputs\tree_centered_chm_structure_clean")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--crop-root", type=Path, default=DEFAULT_CROP_ROOT)
    parser.add_argument("--lidar-product-root", type=Path, default=DEFAULT_LIDAR_PRODUCT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--city-token", action="append", default=None)
    parser.add_argument("--exclude-city-token", action="append", default=[])
    parser.add_argument("--metadata-pattern", default="*_tree_id_centered_nearest_64px_metadata.csv")
    parser.add_argument("--crop-pattern", default="*_tree_id_centered_nearest_64px_rgbnir_crops.npy")
    parser.add_argument("--chm-pattern", default="*_tree_id_centered_nearest_64px_chm.npy")
    parser.add_argument("--lidar-index-pattern", default="*_tree_id_centered_nearest_64px_lidar_index.csv")
    parser.add_argument("--output-suffix", default="_tree_id_centered_chm_structure_metrics.npz")
    parser.add_argument("--center-fraction", type=float, default=0.5)
    parser.add_argument("--vegetation-ndvi-threshold", type=float, default=0.20)
    parser.add_argument("--tree-ndvi-threshold", type=float, default=0.20)
    parser.add_argument("--tree-height-threshold-m", type=float, default=2.00)
    parser.add_argument("--component-mode", choices=("fast", "connected", "none"), default="fast")
    parser.add_argument("--progress-interval", type=int, default=10000)
    parser.add_argument("--max-records", type=int, default=0)
    parser.add_argument("--parallel-workers", type=int, default=1)
    parser.add_argument("--fail-fast", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def normalize_token(value: object) -> str:
    return "".join(ch for ch in str(value or "").strip().lower() if ch.isalnum())


def one_match(paths: list[Path], label: str, root: Path) -> Path:
    if not paths:
        raise FileNotFoundError(f"No {label} found under {root}")
    if len(paths) > 1:
        print(f"WARNING: multiple {label} files under {root}; using {paths[0].name}", flush=True)
    return paths[0]


def discover_city_jobs(args: argparse.Namespace) -> dict[str, dict[str, Path]]:
    if not args.crop_root.exists():
        raise FileNotFoundError(args.crop_root)
    selected = None if args.city_token is None else {normalize_token(token) for token in args.city_token if str(token).strip()}
    excluded = {normalize_token(token) for token in args.exclude_city_token if str(token).strip()}
    out: dict[str, dict[str, Path]] = {}
    for crop_dir in sorted(path for path in args.crop_root.iterdir() if path.is_dir()):
        city = normalize_token(crop_dir.name)
        if selected is not None and city not in selected:
            continue
        if city in excluded:
            continue
        try:
            metadata = one_match(sorted(crop_dir.glob(args.metadata_pattern)), "clean crop metadata", crop_dir)
            crops = one_match(sorted(crop_dir.glob(args.crop_pattern)), "clean crop array", crop_dir)
            chm_dir = args.lidar_product_root / "CHM" / city
            chm = one_match(sorted(chm_dir.glob(args.chm_pattern)), "clean CHM array", chm_dir)
            lidar_index = one_match(sorted(chm_dir.glob(args.lidar_index_pattern)), "clean LiDAR index", chm_dir)
        except FileNotFoundError as error:
            print(f"SKIP {city}: {error}", flush=True)
            continue
        out[city] = {
            "metadata": metadata,
            "crops": crops,
            "chm": chm,
            "lidar_index": lidar_index,
            "output": args.output_dir / f"{city}{args.output_suffix}",
        }
    return out


def numeric_int(series: pd.Series) -> np.ndarray:
    return pd.to_numeric(series, errors="coerce").fillna(-1).astype(np.int64).to_numpy()


def naip_ndvi(rgbnir: np.ndarray) -> np.ndarray:
    chip = np.asarray(rgbnir, dtype=np.float32)
    if chip.ndim != 3 or chip.shape[2] < 4:
        return np.zeros(chip.shape[:2], dtype=np.float32)
    # Clean RGBN crops use band order red, green, blue, nir.
    red = chip[:, :, 0]
    nir = chip[:, :, 3]
    denom = nir + red
    return np.divide(nir - red, denom, out=np.zeros_like(nir, dtype=np.float32), where=np.abs(denom) > 1e-6)


def naip_valid_mask(rgbnir: np.ndarray) -> np.ndarray:
    chip = np.asarray(rgbnir)
    if chip.ndim != 3:
        return np.zeros(chip.shape[:2], dtype=bool)
    finite = np.isfinite(chip).all(axis=2)
    checked = chip[:, :, : min(4, chip.shape[2])]
    all_zero = (checked == 0).all(axis=2)
    all_white = (checked >= 255).all(axis=2)
    return finite & ~all_zero & ~all_white


def compute_structure_row(
    rgbnir: np.ndarray,
    chm: np.ndarray,
    chm_valid: np.ndarray,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    ndvi = naip_ndvi(rgbnir)
    valid = naip_valid_mask(rgbnir)
    chm_clean = np.nan_to_num(np.asarray(chm, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    chm_valid = np.asarray(chm_valid, dtype=bool) & np.isfinite(chm)
    ndvi_on_chm = derived_metrics.resize_bilinear_2d(ndvi, chm_clean.shape)
    valid_on_chm = derived_metrics.resize_bilinear_2d(valid.astype(np.float32), chm_clean.shape) >= 0.5
    veg_weight = (valid_on_chm & chm_valid & (ndvi_on_chm >= float(args.vegetation_ndvi_threshold))).astype(np.float32)
    veg_chm = np.where(veg_weight > 0, chm_clean, 0.0).astype(np.float32)
    sample_data = {
        "naip_rgbnir": rgbnir[np.newaxis, ...],
        "naip_ndvi": ndvi[np.newaxis, ...],
        "naip_valid_mask": valid[np.newaxis, ...],
        "chm": chm_clean[np.newaxis, ...],
        "chm_valid_mask": chm_valid[np.newaxis, ...],
        "vegetation_chm": veg_chm[np.newaxis, ...],
        "vegetation_chm_weight": veg_weight[np.newaxis, ...],
    }
    structure = derived_metrics.compute_naip_chm_structure(
        sample_data,
        0,
        center_fraction=args.center_fraction,
        vegetation_ndvi_threshold=args.vegetation_ndvi_threshold,
        tree_ndvi_threshold=args.tree_ndvi_threshold,
        tree_height_threshold_m=args.tree_height_threshold_m,
        component_mode=args.component_mode,
    )
    return chm_clean, chm_valid, veg_chm, veg_weight, structure


def build_lidar_lookup(metadata: pd.DataFrame, lidar_index: pd.DataFrame, city: str) -> tuple[str, dict[int, int], np.ndarray]:
    if "tree_id" in metadata.columns and "tree_id" in lidar_index.columns:
        metadata_ids = numeric_int(metadata["tree_id"])
        lidar_ids = numeric_int(lidar_index["tree_id"])
        key_name = "tree_id"
    elif "crop_index" in metadata.columns and "crop_index" in lidar_index.columns:
        metadata_ids = numeric_int(metadata["crop_index"])
        lidar_ids = numeric_int(lidar_index["crop_index"])
        key_name = "crop_index"
    elif "crop_index" in metadata.columns and "tree_centered_index" in lidar_index.columns:
        metadata_ids = numeric_int(metadata["crop_index"])
        lidar_ids = numeric_int(lidar_index["tree_centered_index"])
        key_name = "crop_index_to_tree_centered_index"
    else:
        raise RuntimeError(
            f"{city}: cannot align metadata to LiDAR index; need tree_id or crop_index-compatible columns."
        )
    lookup: dict[int, int] = {}
    duplicates = 0
    for pos, key in enumerate(lidar_ids.tolist()):
        key = int(key)
        if key < 0:
            continue
        if key in lookup:
            duplicates += 1
            continue
        lookup[key] = pos
    if duplicates:
        print(f"{city}: WARNING ignored {duplicates:,} duplicate LiDAR index key(s) for {key_name}", flush=True)
    return key_name, lookup, metadata_ids


def process_city(city: str, paths: dict[str, Path], args: argparse.Namespace) -> dict[str, object]:
    output_npz = paths["output"]
    if output_npz.exists() and not args.force:
        print(f"{city}: exists; skipping {output_npz}", flush=True)
        return {"city_token": city, "status": "skipped_exists", "output_npz": str(output_npz)}
    if args.dry_run:
        print(f"{city}: would write {output_npz}", flush=True)
        return {"city_token": city, "status": "dry_run", "output_npz": str(output_npz)}

    metadata = pd.read_csv(paths["metadata"], low_memory=False)
    if args.max_records > 0:
        metadata = metadata.iloc[: int(args.max_records)].copy()
    required_metadata = {"tree_id", "crop_index"}
    missing = sorted(required_metadata.difference(metadata.columns))
    if missing:
        raise RuntimeError(f"{city}: {paths['metadata']} missing required column(s): {missing}")

    lidar_index = pd.read_csv(paths["lidar_index"], low_memory=False)
    key_name, lidar_lookup, metadata_lidar_keys = build_lidar_lookup(metadata, lidar_index, city)
    crop_indices = numeric_int(metadata["crop_index"])
    row_indices = numeric_int(metadata["row_index"]) if "row_index" in metadata.columns else np.arange(len(metadata), dtype=np.int64)
    tree_ids = numeric_int(metadata["tree_id"])

    crops = np.load(paths["crops"], mmap_mode="r")
    chm = np.load(paths["chm"], mmap_mode="r")
    if chm.ndim != 3:
        raise RuntimeError(f"{city}: expected 3D CHM array, got {chm.shape}")

    count = len(metadata)
    chm_shape = tuple(chm.shape[1:3])
    structure = np.zeros((count, len(derived_metrics.NAIP_CHM_STRUCTURE_COLUMNS)), dtype=np.float32)
    chm_out = np.zeros((count, *chm_shape), dtype=np.float32)
    chm_valid_out = np.zeros((count, *chm_shape), dtype=bool)
    veg_chm_out = np.zeros((count, *chm_shape), dtype=np.float32)
    veg_chm_weight_out = np.zeros((count, *chm_shape), dtype=np.float32)
    source_crop_index = np.full(count, -1, dtype=np.int64)
    source_lidar_index = np.full(count, -1, dtype=np.int64)
    missing_crop = np.zeros(count, dtype=bool)
    missing_chm = np.zeros(count, dtype=bool)

    print(
        f"{city}: deriving clean CHM/structure; rows={count:,}; crops={crops.shape}; "
        f"chm={chm.shape}; alignment={key_name}",
        flush=True,
    )
    for out_index, (crop_index, lidar_key) in enumerate(zip(crop_indices, metadata_lidar_keys, strict=False)):
        if crop_index < 0 or crop_index >= crops.shape[0]:
            missing_crop[out_index] = True
            continue
        lidar_pos = lidar_lookup.get(int(lidar_key), -1)
        if lidar_pos < 0 or lidar_pos >= chm.shape[0]:
            missing_chm[out_index] = True
            continue
        source_crop_index[out_index] = int(crop_index)
        source_lidar_index[out_index] = int(lidar_pos)
        chm_row = np.asarray(chm[lidar_pos], dtype=np.float32)
        chm_valid = np.isfinite(chm_row)
        chm_clean, chm_valid_clean, veg_chm, veg_weight, row_structure = compute_structure_row(
            np.asarray(crops[crop_index]),
            chm_row,
            chm_valid,
            args,
        )
        chm_out[out_index] = chm_clean
        chm_valid_out[out_index] = chm_valid_clean
        veg_chm_out[out_index] = veg_chm
        veg_chm_weight_out[out_index] = veg_weight
        structure[out_index] = row_structure
        if args.progress_interval > 0 and (out_index + 1) % args.progress_interval == 0:
            print(f"{city}: processed {out_index + 1:,}/{count:,}", flush=True)

    output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_npz,
        tree_centered_chm=chm_out,
        tree_centered_chm_valid_mask=chm_valid_out,
        tree_centered_vegetation_chm=veg_chm_out,
        tree_centered_vegetation_chm_weight=veg_chm_weight_out,
        tree_centered_naip_chm_structure=structure,
        tree_centered_naip_chm_structure_columns=np.asarray(derived_metrics.NAIP_CHM_STRUCTURE_COLUMNS, dtype="U96"),
        tree_id=tree_ids,
        row_index=row_indices,
        crop_index=crop_indices,
        source_crop_index=source_crop_index,
        source_lidar_index=source_lidar_index,
        missing_crop=missing_crop,
        missing_chm=missing_chm,
        config_json=np.asarray(
            json.dumps(
                {
                    "city_token": city,
                    "metadata_csv": str(paths["metadata"]),
                    "crop_path": str(paths["crops"]),
                    "chm_path": str(paths["chm"]),
                    "lidar_index_path": str(paths["lidar_index"]),
                    "alignment_key": key_name,
                    "center_fraction": float(args.center_fraction),
                    "vegetation_ndvi_threshold": float(args.vegetation_ndvi_threshold),
                    "tree_ndvi_threshold": float(args.tree_ndvi_threshold),
                    "tree_height_threshold_m": float(args.tree_height_threshold_m),
                    "component_mode": args.component_mode,
                },
                sort_keys=True,
            )
        ),
    )
    summary = {
        "city_token": city,
        "status": "completed",
        "rows": int(count),
        "missing_crop": int(missing_crop.sum()),
        "missing_chm": int(missing_chm.sum()),
        "mean_chm_valid": float(chm_valid_out.mean()),
        "structure_columns": int(structure.shape[1]),
        "alignment_key": key_name,
        "output_npz": str(output_npz),
    }
    print(
        f"{city}: wrote {output_npz}; rows={count:,}; missing_crop={summary['missing_crop']:,}; "
        f"missing_chm={summary['missing_chm']:,}; mean_chm_valid={summary['mean_chm_valid']:.3f}",
        flush=True,
    )
    return summary


def main() -> int:
    args = parse_args()
    jobs = discover_city_jobs(args)
    if not jobs:
        raise SystemExit("No complete clean NAIP + CHM city products found.")
    print(f"Selected {len(jobs):,} clean city job(s).", flush=True)
    worker_count = max(1, min(int(args.parallel_workers), len(jobs)))
    summaries: list[dict[str, object]] = []
    failures: list[tuple[str, str]] = []
    if worker_count == 1:
        for city, paths in jobs.items():
            try:
                summaries.append(process_city(city, paths, args))
            except Exception as error:
                print(f"{city}: FAILED: {error}", flush=True)
                failures.append((city, str(error)))
                if args.fail_fast:
                    break
    else:
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            future_to_city = {executor.submit(process_city, city, paths, args): city for city, paths in jobs.items()}
            for future in as_completed(future_to_city):
                city = future_to_city[future]
                try:
                    summaries.append(future.result())
                except Exception as error:
                    print(f"{city}: FAILED: {error}", flush=True)
                    failures.append((city, str(error)))

    summary = pd.DataFrame(summaries).sort_values("city_token", kind="stable") if summaries else pd.DataFrame()
    if args.dry_run:
        print("Dry run: no summary or sidecars written.", flush=True)
    else:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        summary_path = args.output_dir / "clean_tree_centered_chm_structure_summary.csv"
        summary.to_csv(summary_path, index=False)
        print(f"Wrote summary: {summary_path}", flush=True)
    if failures:
        if not args.dry_run:
            failure_path = args.output_dir / "clean_tree_centered_chm_structure_failures.csv"
            pd.DataFrame(failures, columns=["city_token", "error"]).to_csv(failure_path, index=False)
            print(f"Wrote failures: {failure_path}", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
