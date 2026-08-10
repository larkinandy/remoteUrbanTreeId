#!/usr/bin/env python3
"""Assemble clean tree_id-centered model input shards.

This is the clean-dataset shard boundary. It reads only the standalone
``H:\\TreeCenteredModelInputs`` products, aligns every modality by ``tree_id``,
screens to rows with all required inputs, and writes complete physical shards
that can be consumed by the tree-centered training scripts.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


CLEAN_ROOT = Path(r"H:\TreeCenteredModelInputs")
DEFAULT_QA_ROOT = CLEAN_ROOT / "tree_centered_crop_qa_clean"
DEFAULT_CROP_ROOT = CLEAN_ROOT / "tree_centered_naip_crops_clean"
DEFAULT_STRUCTURE_ROOT = CLEAN_ROOT / "tree_centered_chm_structure_clean"
DEFAULT_SENTINEL_ROOT = CLEAN_ROOT / "tree_centered_sentinel_phenology_clean"
DEFAULT_GEE_ROOT = CLEAN_ROOT / "tree_centered_gee_inputs_clean"
DEFAULT_PRISM_ROOT = CLEAN_ROOT / "tree_centered_prism_normals_clean"
DEFAULT_PRISM_DAILY_ROOT = CLEAN_ROOT / "tree_centered_prism_daily_clean"
DEFAULT_OUTPUT_ROOT = CLEAN_ROOT / "tree_centered_complete_sharded100k_clean"
DEFAULT_ORIGINAL_RAW_SENTINEL_ROOT = Path(r"E:\cell\sentinel2_timeseries")
DEFAULT_SUPPLEMENTAL_RAW_SENTINEL_ROOT = Path(
    r"E:\TreeCenteredModelInputs\tree_centered_sentinel_timeseries_supplemental"
)

SENTINEL_FEATURE_COLUMNS = [
    "B2",
    "B3",
    "B4",
    "B5",
    "B6",
    "B7",
    "B8",
    "B8A",
    "B11",
    "B12",
    "NDVI",
    "GNDVI",
    "CIg",
    "CIre",
    "MTCI",
    "MCARI",
    "NDVIre1",
    "NDVIre2",
    "REPI",
    "NDII",
    "MSAVI",
    "LAI_re",
    "LAI_ndvi",
]
SENTINEL_META_COLUMNS = [
    "sentinel_observed",
    "sentinel_interpolated",
    "data_quality_mask",
    "source_image_count",
    "interpolation_gap_days",
    "interpolation_fraction",
    "days_since_previous_observation",
    "days_until_next_observation",
    "delta_days",
    "doy_sin",
    "doy_cos",
]
SENTINEL_SEQUENCE_COLUMNS = SENTINEL_FEATURE_COLUMNS + SENTINEL_META_COLUMNS
SENTINEL_QUALITY_COLUMNS = [
    "sentinel_sequence_length",
    "sentinel_observed_fraction",
    "sentinel_interpolated_fraction",
    "sentinel_mean_data_quality_mask",
    "sentinel_max_interpolation_gap_days",
    "sentinel_mean_interpolation_gap_days",
    "sentinel_mean_source_image_count",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--qa-root", type=Path, default=DEFAULT_QA_ROOT)
    parser.add_argument("--crop-root", type=Path, default=DEFAULT_CROP_ROOT)
    parser.add_argument("--structure-root", type=Path, default=DEFAULT_STRUCTURE_ROOT)
    parser.add_argument("--sentinel-phenology-root", type=Path, default=DEFAULT_SENTINEL_ROOT)
    parser.add_argument("--original-raw-sentinel-timeseries-root", type=Path, default=DEFAULT_ORIGINAL_RAW_SENTINEL_ROOT)
    parser.add_argument(
        "--supplemental-raw-sentinel-timeseries-root",
        type=Path,
        default=DEFAULT_SUPPLEMENTAL_RAW_SENTINEL_ROOT,
    )
    parser.add_argument("--gee-root", type=Path, default=DEFAULT_GEE_ROOT)
    parser.add_argument("--prism-root", type=Path, default=DEFAULT_PRISM_ROOT)
    parser.add_argument("--prism-daily-root", type=Path, default=DEFAULT_PRISM_DAILY_ROOT)
    parser.add_argument(
        "--use-prism-daily-temperature",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Join daily PRISM ppt, tmean, and vpdmax to each raw Sentinel "
            "timestep by tree_id and exact date, then append all three values "
            "to sentinel_sequence. The option name is retained for compatibility."
        ),
    )
    parser.add_argument(
        "--prism-daily-sentinel-end-date",
        type=int,
        default=20231231,
        help=(
            "Last Sentinel date retained when daily PRISM is enabled, formatted "
            "as YYYYMMDD. Later timesteps are removed before PRISM completeness "
            "is evaluated."
        ),
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--city-token",
        action="append",
        default=None,
        help=(
            "Process only this normalized city token. Repeat the option to rebuild "
            "multiple cities, for example --city-token denver --city-token auroraco."
        ),
    )
    parser.add_argument("--exclude-city-token", action="append", default=[])
    parser.add_argument("--shard-size", type=int, default=100_000)
    parser.add_argument("--max-sentinel-sequence-length", type=int, default=80)
    parser.add_argument("--coordinate-decimals", type=int, default=7)
    parser.add_argument("--require-raw-sentinel", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--exclude-qa-from-model", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--require-prism-normals", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--compress", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--parallel-workers",
        type=int,
        default=1,
        help="Number of cities to process concurrently. Each city writes to its own output subfolder.",
    )
    parser.add_argument(
        "--fail-fast",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Stop after the first failed city in serial mode. Parallel mode lets already-started cities finish.",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def normalize_token(value: object) -> str:
    return "".join(ch for ch in str(value or "").strip().lower() if ch.isalnum())


def require_clean_roots(args: argparse.Namespace) -> None:
    clean_root = str(CLEAN_ROOT).lower()
    for name in (
        "qa_root",
        "crop_root",
        "structure_root",
        "sentinel_phenology_root",
        "gee_root",
        "prism_root",
        "output_root",
    ):
        path = Path(getattr(args, name)).resolve()
        if not str(path).lower().startswith(clean_root):
            raise SystemExit(f"{name} must point inside {CLEAN_ROOT}; got {path}")
    if args.use_prism_daily_temperature:
        path = Path(args.prism_daily_root).resolve()
        if not str(path).lower().startswith(clean_root):
            raise SystemExit(f"prism_daily_root must point inside {CLEAN_ROOT}; got {path}")


def coord_key(lat: np.ndarray | pd.Series, lon: np.ndarray | pd.Series, decimals: int) -> np.ndarray:
    lat_values = pd.to_numeric(pd.Series(lat), errors="coerce").round(int(decimals)).astype(str)
    lon_values = pd.to_numeric(pd.Series(lon), errors="coerce").round(int(decimals)).astype(str)
    return (lat_values + "|" + lon_values).to_numpy(dtype=str)


def bool_series(series: pd.Series) -> np.ndarray:
    if series.dtype == bool:
        return series.fillna(False).to_numpy(dtype=bool)
    return series.fillna(False).astype(str).str.lower().isin({"1", "true", "yes", "y"}).to_numpy(dtype=bool)


def finite_rows(array: np.ndarray) -> np.ndarray:
    arr = np.asarray(array)
    if arr.ndim <= 1:
        return np.isfinite(arr)
    return np.isfinite(arr.reshape(arr.shape[0], -1)).all(axis=1)


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as data:
        return {key: np.asarray(data[key]) for key in data.files}


def raw_sentinel_files(root: Path, city: str) -> list[Path]:
    city_dir = Path(root) / city
    if not city_dir.exists():
        return []
    paths = sorted(city_dir.glob("*sentinel2_15day_time_series_batch_*.csv"))
    return [path for path in paths if "_summary_" not in path.name]


def pad_sentinel_sequence(
    group: pd.DataFrame,
    max_len: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.int16, np.ndarray]:
    group = group.sort_values("date", kind="stable")
    if len(group) > max_len:
        # Keep the most recent max_len evenly enough for a fixed tensor while
        # preserving temporal order. The phenology sidecar carries the full
        # aggregate; this sequence is the bounded raw/context branch.
        group = group.tail(max_len)
    seq = np.zeros((max_len, len(SENTINEL_SEQUENCE_COLUMNS)), dtype=np.float32)
    mask = np.zeros(max_len, dtype=bool)
    dates_yyyymmdd = np.zeros(max_len, dtype=np.int32)
    values = group[SENTINEL_SEQUENCE_COLUMNS].to_numpy(dtype=np.float32, copy=True)
    length = int(min(len(values), max_len))
    if length:
        seq[:length] = np.nan_to_num(values[:length], nan=0.0, posinf=0.0, neginf=0.0)
        mask[:length] = True
        dates_yyyymmdd[:length] = (
            pd.to_datetime(group["date"].iloc[:length], errors="coerce")
            .dt.strftime("%Y%m%d")
            .fillna("0")
            .astype(np.int32)
            .to_numpy()
        )
    quality = np.zeros(len(SENTINEL_QUALITY_COLUMNS), dtype=np.float32)
    quality[0] = length
    if length:
        for out_index, column in enumerate(
            [
                "sentinel_observed",
                "sentinel_interpolated",
                "data_quality_mask",
                "interpolation_gap_days",
                "interpolation_gap_days",
                "source_image_count",
            ],
            start=1,
        ):
            series = pd.to_numeric(group[column], errors="coerce")
            if column == "interpolation_gap_days" and out_index == 4:
                quality[out_index] = float(series.max(skipna=True)) if series.notna().any() else 0.0
            else:
                quality[out_index] = float(series.mean(skipna=True)) if series.notna().any() else 0.0
    return seq, mask, dates_yyyymmdd, np.int16(length), quality


def load_raw_sentinel_groups(root: Path, city: str, source_label: str, decimals: int) -> dict[str, pd.DataFrame]:
    files = raw_sentinel_files(root, city)
    if not files:
        return {}
    usecols = ["latitude", "longitude", "date", *SENTINEL_SEQUENCE_COLUMNS]
    frames: list[pd.DataFrame] = []
    for file_index, path in enumerate(files, start=1):
        frame = pd.read_csv(path, usecols=lambda column: column in usecols, low_memory=False)
        missing = sorted(set(usecols).difference(frame.columns))
        if missing:
            raise RuntimeError(f"{path} is missing raw Sentinel column(s): {missing}")
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
        frame = frame.dropna(subset=["latitude", "longitude", "date"]).copy()
        if frame.empty:
            continue
        frame["sentinel_coord_key"] = coord_key(frame["latitude"], frame["longitude"], decimals)
        frames.append(frame)
        if file_index == 1 or file_index % 10 == 0 or file_index == len(files):
            print(
                f"  {city}: loaded {source_label} raw Sentinel file {file_index:,}/{len(files):,}; "
                f"rows={sum(len(item) for item in frames):,}",
                flush=True,
            )
    if not frames:
        return {}
    raw = pd.concat(frames, ignore_index=True)
    out: dict[str, pd.DataFrame] = {}
    for key, group in raw.groupby("sentinel_coord_key", sort=False):
        out[str(key)] = group.sort_values("date", kind="stable")
    print(f"  {city}: indexed {source_label} raw Sentinel coordinate groups={len(out):,}", flush=True)
    return out


def load_raw_sentinel_sequences(
    city: str,
    metadata: pd.DataFrame,
    sentinel: dict[str, np.ndarray],
    sentinel_pos: np.ndarray,
    args: argparse.Namespace,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    count = len(metadata)
    max_len = max(1, int(args.max_sentinel_sequence_length))
    sequence = np.zeros((count, max_len, len(SENTINEL_SEQUENCE_COLUMNS)), dtype=np.float32)
    mask = np.zeros((count, max_len), dtype=bool)
    dates_yyyymmdd = np.zeros((count, max_len), dtype=np.int32)
    lengths = np.zeros(count, dtype=np.int16)
    quality = np.zeros((count, len(SENTINEL_QUALITY_COLUMNS)), dtype=np.float32)
    source_code = np.zeros(count, dtype=np.int8)
    valid = np.zeros(count, dtype=bool)

    original = load_raw_sentinel_groups(
        Path(args.original_raw_sentinel_timeseries_root),
        city,
        "original",
        int(args.coordinate_decimals),
    )
    supplemental = load_raw_sentinel_groups(
        Path(args.supplemental_raw_sentinel_timeseries_root),
        city,
        "supplemental",
        int(args.coordinate_decimals),
    )
    if not original and not supplemental:
        if args.require_raw_sentinel:
            raise FileNotFoundError(
                f"{city}: no raw Sentinel files under {args.original_raw_sentinel_timeseries_root} "
                f"or {args.supplemental_raw_sentinel_timeseries_root}"
            )
        return {
            "sentinel_sequence": sequence,
            "sentinel_sequence_mask": mask,
            "sentinel_sequence_dates_yyyymmdd": dates_yyyymmdd,
            "sentinel_sequence_length": lengths,
            "sentinel_quality": quality,
            "sentinel_quality_columns": np.asarray(SENTINEL_QUALITY_COLUMNS),
            "sentinel_feature_columns": np.asarray(SENTINEL_SEQUENCE_COLUMNS),
            "raw_sentinel_source_code": source_code,
            "missing_raw_sentinel": ~valid,
        }, valid

    valid_sentinel_pos = sentinel_pos.clip(min=0)
    keys = coord_key(
        np.asarray(sentinel["sentinel_lat"])[valid_sentinel_pos],
        np.asarray(sentinel["sentinel_lon"])[valid_sentinel_pos],
        int(args.coordinate_decimals),
    )
    labels = np.asarray(sentinel.get("sentinel_source_used", np.full(len(valid_sentinel_pos), "", dtype=object)))[
        valid_sentinel_pos
    ].astype(str)
    for row, key in enumerate(keys.tolist()):
        group = None
        preferred = labels[row].strip().lower()
        if preferred == "supplemental":
            group = supplemental.get(key)
            selected_code = 2
            if group is None:
                group = original.get(key)
                selected_code = 1 if group is not None else 0
        else:
            group = original.get(key)
            selected_code = 1
            if group is None:
                group = supplemental.get(key)
                selected_code = 2 if group is not None else 0
        if group is None or group.empty:
            continue
        seq, seq_mask, seq_dates, seq_len, seq_quality = pad_sentinel_sequence(group, max_len)
        sequence[row] = seq
        mask[row] = seq_mask
        dates_yyyymmdd[row] = seq_dates
        lengths[row] = seq_len
        quality[row] = seq_quality
        source_code[row] = np.int8(selected_code)
        valid[row] = seq_len > 0
    print(f"  {city}: matched raw Sentinel sequences={int(valid.sum()):,}/{count:,}", flush=True)
    return {
        "sentinel_sequence": sequence,
        "sentinel_sequence_mask": mask,
        "sentinel_sequence_dates_yyyymmdd": dates_yyyymmdd,
        "sentinel_sequence_length": lengths,
        "sentinel_quality": quality,
        "sentinel_quality_columns": np.asarray(SENTINEL_QUALITY_COLUMNS),
        "sentinel_feature_columns": np.asarray(SENTINEL_SEQUENCE_COLUMNS),
        "raw_sentinel_source_code": source_code,
        "missing_raw_sentinel": ~valid,
    }, valid


def align_positions(metadata: pd.DataFrame, sidecar: dict[str, np.ndarray], label: str) -> np.ndarray:
    if "tree_id" not in metadata.columns:
        raise RuntimeError(f"{label}: metadata is missing tree_id")
    if "tree_id" not in sidecar:
        raise RuntimeError(f"{label}: sidecar is missing tree_id")
    target = pd.to_numeric(metadata["tree_id"], errors="coerce").fillna(-1).astype(np.int64).to_numpy()
    source = np.asarray(sidecar["tree_id"], dtype=np.int64)
    if len(target) == len(source) and np.array_equal(target, source):
        return np.arange(len(target), dtype=np.int64)
    lookup = {int(tree_id): int(row) for row, tree_id in enumerate(source.tolist())}
    positions = np.full(len(target), -1, dtype=np.int64)
    for row, tree_id in enumerate(target.tolist()):
        positions[row] = lookup.get(int(tree_id), -1)
    missing = int((positions < 0).sum())
    if missing:
        print(f"  {label}: {missing:,}/{len(target):,} metadata tree_id(s) missing from sidecar", flush=True)
    return positions


def clip_raw_sentinel_end_date(raw_sentinel: dict[str, np.ndarray], end_date: int) -> int:
    """Remove masked Sentinel timesteps later than ``end_date`` in place."""
    dates = np.asarray(raw_sentinel["sentinel_sequence_dates_yyyymmdd"], dtype=np.int32)
    mask = np.asarray(raw_sentinel["sentinel_sequence_mask"], dtype=bool).copy()
    sequence = np.asarray(raw_sentinel["sentinel_sequence"], dtype=np.float32).copy()
    clipped = mask & (dates > int(end_date))
    clipped_count = int(clipped.sum())
    if not clipped_count:
        return 0

    mask[clipped] = False
    dates = dates.copy()
    dates[clipped] = 0
    sequence[clipped] = 0.0
    lengths = mask.sum(axis=1).astype(np.int16)

    quality = np.asarray(raw_sentinel["sentinel_quality"], dtype=np.float32).copy()
    quality[:, 0] = lengths
    count = np.maximum(lengths.astype(np.float32), 1.0)
    mean_columns = (
        "sentinel_observed",
        "sentinel_interpolated",
        "data_quality_mask",
        "interpolation_gap_days",
        "source_image_count",
    )
    quality_indices = (1, 2, 3, 5, 6)
    for column, quality_index in zip(mean_columns, quality_indices):
        feature_index = SENTINEL_SEQUENCE_COLUMNS.index(column)
        values = sequence[:, :, feature_index]
        quality[:, quality_index] = np.where(mask, values, 0.0).sum(axis=1) / count
    gap_index = SENTINEL_SEQUENCE_COLUMNS.index("interpolation_gap_days")
    quality[:, 4] = np.where(mask, sequence[:, :, gap_index], -np.inf).max(axis=1)
    quality[lengths == 0, 4] = 0.0

    raw_sentinel["sentinel_sequence"] = sequence
    raw_sentinel["sentinel_sequence_mask"] = mask
    raw_sentinel["sentinel_sequence_dates_yyyymmdd"] = dates
    raw_sentinel["sentinel_sequence_length"] = lengths
    raw_sentinel["sentinel_quality"] = quality
    return clipped_count


def append_prism_daily_temperature(
    city: str,
    metadata: pd.DataFrame,
    raw_sentinel: dict[str, np.ndarray],
    args: argparse.Namespace,
) -> np.ndarray:
    """Append date-matched daily PRISM ppt, tmean, and vpdmax."""
    count = len(metadata)
    global_path = Path(args.prism_daily_root) / "prism_daily_values.npz"
    city_path = Path(args.prism_daily_root) / city / f"{city}_prism_daily_index.npz"
    if not global_path.exists():
        raise FileNotFoundError(f"Missing PRISM daily values: {global_path}")
    if not city_path.exists():
        raise FileNotFoundError(f"Missing PRISM daily city index: {city_path}")

    daily = load_npz(global_path)
    city_index = load_npz(city_path)
    variables = [str(value) for value in np.asarray(daily["variable_names"]).tolist()]
    requested_variables = ("ppt", "tmean", "vpdmax")
    missing_variables = [variable for variable in requested_variables if variable not in variables]
    if missing_variables:
        raise RuntimeError(
            f"{global_path} is missing required daily variable(s) {missing_variables}; "
            f"available={variables}"
        )
    variable_indices = np.asarray([variables.index(variable) for variable in requested_variables], dtype=np.int64)
    daily_dates = np.asarray(daily["dates_yyyymmdd"], dtype=np.int32)
    if len(daily_dates) == 0 or np.any(np.diff(daily_dates) <= 0):
        raise RuntimeError(f"{global_path} dates must be nonempty, unique, and ascending")

    city_positions = align_positions(metadata, city_index, f"{city} PRISM daily")
    valid_city_positions = city_positions.clip(min=0)
    source_rows = np.asarray(city_index["prism_daily_source_row"], dtype=np.int64)[valid_city_positions]
    source_rows[city_positions < 0] = -1

    clipped_timesteps = clip_raw_sentinel_end_date(
        raw_sentinel,
        int(args.prism_daily_sentinel_end_date),
    )
    sequence_dates = np.asarray(raw_sentinel["sentinel_sequence_dates_yyyymmdd"], dtype=np.int32)
    sentinel_mask = np.asarray(raw_sentinel["sentinel_sequence_mask"], dtype=bool)
    date_positions = np.searchsorted(daily_dates, sequence_dates)
    date_in_range = date_positions < len(daily_dates)
    safe_date_positions = np.clip(date_positions, 0, max(len(daily_dates) - 1, 0))
    date_matches = date_in_range & (daily_dates[safe_date_positions] == sequence_dates)
    source_valid = (source_rows >= 0) & (source_rows < np.asarray(daily["values"]).shape[0])
    safe_source_rows = np.clip(source_rows, 0, max(np.asarray(daily["values"]).shape[0] - 1, 0))

    climate = np.asarray(daily["values"], dtype=np.float32)[
        safe_source_rows[:, None],
        safe_date_positions,
    ][..., variable_indices]
    climate_valid = np.asarray(daily["valid_mask"], dtype=bool)[
        safe_source_rows[:, None],
        safe_date_positions,
    ][..., variable_indices]
    joined = (
        sentinel_mask
        & date_matches
        & source_valid[:, None]
        & climate_valid.all(axis=2)
    )
    climate = np.where(joined[..., None], climate, 0.0).astype(np.float32)
    row_valid = (~sentinel_mask | joined).all(axis=1) & sentinel_mask.any(axis=1)

    raw_sentinel["sentinel_sequence"] = np.concatenate(
        [
            np.asarray(raw_sentinel["sentinel_sequence"], dtype=np.float32),
            climate,
        ],
        axis=2,
    )
    raw_sentinel["sentinel_feature_columns"] = np.concatenate(
        [
            np.asarray(raw_sentinel["sentinel_feature_columns"]).astype(str),
            np.asarray([f"prism_daily_{variable}" for variable in requested_variables]),
        ]
    )
    raw_sentinel["sentinel_prism_daily"] = climate
    raw_sentinel["sentinel_prism_daily_columns"] = np.asarray(requested_variables)
    raw_sentinel["sentinel_prism_daily_mask"] = joined
    raw_sentinel["missing_prism_daily"] = ~row_valid
    print(
        f"  {city}: date-matched PRISM ppt+tmean+vpdmax rows={int(row_valid.sum()):,}/{count:,}; "
        f"timesteps={int(joined.sum()):,}/{int(sentinel_mask.sum()):,}; "
        f"Sentinel timesteps after {int(args.prism_daily_sentinel_end_date)} clipped={clipped_timesteps:,}",
        flush=True,
    )
    return row_valid


def subset_by_positions(array: np.ndarray, positions: np.ndarray) -> np.ndarray:
    arr = np.asarray(array)
    return arr[positions] if arr.shape and arr.shape[0] == len(positions) else arr


def sidecar_path(root: Path, city: str, suffix: str) -> Path:
    path = root / city / f"{city}{suffix}"
    if path.exists():
        return path
    return root / f"{city}{suffix}"


def city_paths(args: argparse.Namespace, city: str) -> dict[str, Path]:
    return {
        "qa": args.qa_root / f"{city}_tree_centered_qa_metadata.csv",
        "crop": args.crop_root / city / f"{city}_tree_id_centered_nearest_64px_rgbnir_crops.npy",
        "structure": sidecar_path(args.structure_root, city, "_tree_id_centered_chm_structure_metrics.npz"),
        "sentinel": sidecar_path(args.sentinel_phenology_root, city, "_tree_id_centered_sentinel_phenology.npz"),
        "gee": sidecar_path(args.gee_root, city, "_tree_id_centered_gee_inputs.npz"),
        "prism": sidecar_path(args.prism_root, city, "_prism_normals.npz"),
    }


def discover_cities(args: argparse.Namespace) -> list[str]:
    selected = None if args.city_token is None else {normalize_token(value) for value in args.city_token}
    excluded = {normalize_token(value) for value in args.exclude_city_token}
    available: set[str] = set()
    cities: list[str] = []
    for path in sorted(Path(args.qa_root).glob("*_tree_centered_qa_metadata.csv")):
        city = normalize_token(path.name.replace("_tree_centered_qa_metadata.csv", ""))
        available.add(city)
        if selected is not None and city not in selected:
            continue
        if city in excluded:
            continue
        cities.append(city)
    if selected is not None:
        missing = sorted(selected.difference(available))
        if missing:
            raise SystemExit(
                "Requested --city-token value(s) were not found under "
                f"{args.qa_root}: {', '.join(missing)}"
            )
    return cities


def screen_rows(
    metadata: pd.DataFrame,
    crops: np.ndarray,
    structure: dict[str, np.ndarray],
    structure_pos: np.ndarray,
    sentinel: dict[str, np.ndarray],
    sentinel_pos: np.ndarray,
    gee: dict[str, np.ndarray],
    gee_pos: np.ndarray,
    raw_sentinel: dict[str, np.ndarray],
    raw_sentinel_valid: np.ndarray,
    prism: dict[str, np.ndarray] | None,
    prism_pos: np.ndarray | None,
    args: argparse.Namespace,
) -> tuple[np.ndarray, dict[str, int]]:
    count = len(metadata)
    reasons: dict[str, np.ndarray] = {}
    crop_index = pd.to_numeric(metadata["crop_index"], errors="coerce").fillna(-1).astype(np.int64).to_numpy()
    reasons["missing_crop"] = (crop_index < 0) | (crop_index >= int(crops.shape[0]))
    reasons["qa_exclude_from_model"] = (
        bool_series(metadata["qa_exclude_from_model"])
        if args.exclude_qa_from_model and "qa_exclude_from_model" in metadata.columns
        else np.zeros(count, dtype=bool)
    )
    reasons["missing_structure_row"] = structure_pos < 0
    reasons["missing_sentinel_row"] = sentinel_pos < 0
    reasons["missing_gee_row"] = gee_pos < 0
    valid_structure_pos = structure_pos.clip(min=0)
    valid_sentinel_pos = sentinel_pos.clip(min=0)
    valid_gee_pos = gee_pos.clip(min=0)

    reasons["missing_chm"] = (
        np.asarray(structure.get("missing_chm", np.zeros(len(valid_structure_pos), dtype=bool))[valid_structure_pos], dtype=bool)
        | reasons["missing_structure_row"]
    )
    if "tree_centered_chm_valid_mask" in structure:
        chm_valid = np.asarray(structure["tree_centered_chm_valid_mask"][valid_structure_pos], dtype=bool)
        reasons["missing_chm"] |= ~chm_valid.reshape(chm_valid.shape[0], -1).any(axis=1)
    if "qa_flag_missing_lidar_chm" in metadata.columns:
        reasons["missing_chm"] |= bool_series(metadata["qa_flag_missing_lidar_chm"])
    reasons["nonfinite_structure"] = (
        ~finite_rows(np.asarray(structure["tree_centered_naip_chm_structure"])[valid_structure_pos])
        | reasons["missing_structure_row"]
    )
    reasons["missing_sentinel_phenology"] = (
        np.asarray(sentinel["missing_sentinel_phenology"][valid_sentinel_pos], dtype=bool)
        | reasons["missing_sentinel_row"]
    )
    reasons["nonfinite_sentinel_phenology"] = (
        ~finite_rows(np.asarray(sentinel["sentinel_phenology"])[valid_sentinel_pos])
        | reasons["missing_sentinel_row"]
    )
    reasons["missing_satellite_embedding"] = (
        np.asarray(gee["missing_satellite_embedding"][valid_gee_pos], dtype=bool)
        | reasons["missing_gee_row"]
    )
    reasons["nonfinite_satellite_embedding"] = (
        ~finite_rows(np.asarray(gee["satellite_embedding"])[valid_gee_pos])
        | reasons["missing_gee_row"]
    )
    if args.require_raw_sentinel:
        reasons["missing_raw_sentinel"] = ~raw_sentinel_valid
        reasons["nonfinite_raw_sentinel"] = ~finite_rows(np.asarray(raw_sentinel["sentinel_sequence"], dtype=np.float32))
    if args.use_prism_daily_temperature:
        reasons["missing_prism_daily"] = np.asarray(
            raw_sentinel.get("missing_prism_daily", np.ones(count, dtype=bool)),
            dtype=bool,
        )
    if args.require_prism_normals:
        if prism is None or prism_pos is None:
            reasons["missing_prism_normals"] = np.ones(count, dtype=bool)
        else:
            valid_prism_pos = prism_pos.clip(min=0)
            reasons["missing_prism_normals"] = (
                np.asarray(prism["missing_prism_normals"][valid_prism_pos], dtype=bool)
                | (prism_pos < 0)
            )
            reasons["nonfinite_prism_normals"] = (
                ~finite_rows(np.asarray(prism["prism_normals"])[valid_prism_pos])
                | (prism_pos < 0)
            )

    keep = np.ones(count, dtype=bool)
    for mask in reasons.values():
        keep &= ~mask
    return keep, {key: int(value.sum()) for key, value in reasons.items()}


def write_npz(path: Path, arrays: dict[str, Any], compress: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if compress:
        np.savez_compressed(path, **arrays)
    else:
        np.savez(path, **arrays)


def process_city(city: str, args: argparse.Namespace) -> list[dict[str, Any]]:
    paths = city_paths(args, city)
    missing_paths = [f"{name}={path}" for name, path in paths.items() if name != "prism" and not path.exists()]
    if args.require_prism_normals and not paths["prism"].exists():
        missing_paths.append(f"prism={paths['prism']}")
    if missing_paths:
        raise FileNotFoundError(f"{city}: missing required input(s): " + "; ".join(missing_paths))

    metadata = pd.read_csv(paths["qa"], low_memory=False)
    crops = np.load(paths["crop"], mmap_mode="r")
    structure = load_npz(paths["structure"])
    sentinel = load_npz(paths["sentinel"])
    gee = load_npz(paths["gee"])
    prism = load_npz(paths["prism"]) if paths["prism"].exists() else None

    structure_pos = align_positions(metadata, structure, f"{city} structure")
    sentinel_pos = align_positions(metadata, sentinel, f"{city} sentinel")
    gee_pos = align_positions(metadata, gee, f"{city} gee")
    prism_pos = align_positions(metadata, prism, f"{city} prism") if prism is not None else None
    raw_sentinel, raw_sentinel_valid = load_raw_sentinel_sequences(city, metadata, sentinel, sentinel_pos, args)
    if args.use_prism_daily_temperature:
        append_prism_daily_temperature(city, metadata, raw_sentinel, args)
    keep, reason_counts = screen_rows(
        metadata,
        crops,
        structure,
        structure_pos,
        sentinel,
        sentinel_pos,
        gee,
        gee_pos,
        raw_sentinel,
        raw_sentinel_valid,
        prism,
        prism_pos,
        args,
    )
    kept_indices = np.flatnonzero(keep)

    print(
        f"{city}: screened complete rows kept={len(kept_indices):,}/{len(metadata):,}; "
        + ", ".join(f"{key}={value:,}" for key, value in reason_counts.items() if value),
        flush=True,
    )

    out_dir = args.output_root / city
    if not args.dry_run and args.force and out_dir.exists():
        for path in out_dir.glob(f"{city}_part*_tree_centered_complete_inputs.npz"):
            path.unlink()
        for path in out_dir.glob(f"{city}_part*_tree_centered_complete_metadata.csv"):
            path.unlink()

    summaries: list[dict[str, Any]] = []
    shard_size = max(1, int(args.shard_size))
    for part, start in enumerate(range(0, len(kept_indices), shard_size)):
        row_idx = kept_indices[start : start + shard_size]
        shard_meta = metadata.iloc[row_idx].copy().reset_index(drop=True)
        shard_meta["city_token"] = city
        source_key = f"{city}_part{part:03d}_tree_centered_complete"
        shard_meta["source_key"] = source_key
        shard_meta["source_sample_index"] = np.arange(len(shard_meta), dtype=np.int64)
        tree_id = pd.to_numeric(shard_meta["tree_id"], errors="coerce").astype(np.int64).to_numpy()
        shard_meta["tree_centered_index"] = tree_id

        crop_idx = pd.to_numeric(shard_meta["crop_index"], errors="coerce").astype(np.int64).to_numpy()
        struct_idx = structure_pos[row_idx]
        sentinel_idx = sentinel_pos[row_idx]
        gee_idx = gee_pos[row_idx]
        prism_idx = prism_pos[row_idx] if prism_pos is not None else None

        arrays: dict[str, Any] = {
            "tree_centered_naip": np.asarray(crops[crop_idx], dtype=np.uint8),
            "tree_centered_chm": np.asarray(structure["tree_centered_chm"][struct_idx], dtype=np.float32),
            "tree_centered_chm_valid_mask": np.asarray(structure["tree_centered_chm_valid_mask"][struct_idx], dtype=bool),
            "tree_centered_vegetation_chm": np.asarray(structure["tree_centered_vegetation_chm"][struct_idx], dtype=np.float32),
            "tree_centered_vegetation_chm_weight": np.asarray(
                structure.get("tree_centered_vegetation_chm_weight", np.ones_like(structure["tree_centered_vegetation_chm"]))[struct_idx],
                dtype=np.float32,
            ),
            "tree_centered_naip_chm_structure": np.asarray(
                structure["tree_centered_naip_chm_structure"][struct_idx], dtype=np.float32
            ),
            "tree_centered_naip_chm_structure_columns": np.asarray(structure["tree_centered_naip_chm_structure_columns"]),
            "sentinel_phenology": np.asarray(sentinel["sentinel_phenology"][sentinel_idx], dtype=np.float32),
            "sentinel_phenology_columns": np.asarray(sentinel["sentinel_phenology_columns"]),
            "sentinel_phenology_source_code": np.asarray(sentinel["sentinel_phenology_source_code"][sentinel_idx], dtype=np.int8),
            "sentinel_timeseries_match_distance_m": np.asarray(
                sentinel["sentinel_timeseries_match_distance_m"][sentinel_idx], dtype=np.float32
            ),
            "satellite_embedding": np.asarray(gee["satellite_embedding"][gee_idx], dtype=np.float32),
            "satellite_embedding_mask": np.asarray(gee["satellite_embedding_mask"][gee_idx], dtype=np.float32),
            "satellite_embedding_quality": np.asarray(gee["satellite_embedding_quality"][gee_idx], dtype=np.float32),
            "satellite_embedding_columns": np.asarray(gee["satellite_embedding_columns"]),
            "satellite_embedding_years": np.asarray(gee["satellite_embedding_years"], dtype=np.int32),
            "satellite_embedding_quality_columns": np.asarray(gee["satellite_embedding_quality_columns"]),
            "embedding_source_code": np.asarray(gee["embedding_source_code"][gee_idx], dtype=np.int8),
            "used_original_satellite_embedding": np.asarray(gee["used_original_satellite_embedding"][gee_idx], dtype=bool),
            "used_additional_satellite_embedding": np.asarray(gee["used_additional_satellite_embedding"][gee_idx], dtype=bool),
            "sentinel_sequence": np.asarray(raw_sentinel["sentinel_sequence"][row_idx], dtype=np.float32),
            "sentinel_sequence_mask": np.asarray(raw_sentinel["sentinel_sequence_mask"][row_idx], dtype=bool),
            "sentinel_sequence_dates_yyyymmdd": np.asarray(
                raw_sentinel["sentinel_sequence_dates_yyyymmdd"][row_idx], dtype=np.int32
            ),
            "sentinel_sequence_length": np.asarray(raw_sentinel["sentinel_sequence_length"][row_idx], dtype=np.int16),
            "sentinel_quality": np.asarray(raw_sentinel["sentinel_quality"][row_idx], dtype=np.float32),
            "sentinel_quality_columns": np.asarray(raw_sentinel["sentinel_quality_columns"]),
            "sentinel_feature_columns": np.asarray(raw_sentinel["sentinel_feature_columns"]),
            "raw_sentinel_source_code": np.asarray(raw_sentinel["raw_sentinel_source_code"][row_idx], dtype=np.int8),
            "missing_raw_sentinel": np.asarray(raw_sentinel["missing_raw_sentinel"][row_idx], dtype=bool),
            "tree_id": tree_id,
            "tree_centered_index": tree_id,
            "crop_index": crop_idx.astype(np.int64),
            "row_index": np.arange(len(shard_meta), dtype=np.int64),
            "source_metadata_row": row_idx.astype(np.int64),
            "source_structure_row": struct_idx.astype(np.int64),
            "source_sentinel_row": sentinel_idx.astype(np.int64),
            "source_gee_row": gee_idx.astype(np.int64),
        }
        if args.use_prism_daily_temperature:
            arrays.update(
                {
                    "sentinel_prism_daily": np.asarray(
                        raw_sentinel["sentinel_prism_daily"][row_idx], dtype=np.float32
                    ),
                    "sentinel_prism_daily_columns": np.asarray(
                        raw_sentinel["sentinel_prism_daily_columns"]
                    ),
                    "sentinel_prism_daily_mask": np.asarray(
                        raw_sentinel["sentinel_prism_daily_mask"][row_idx], dtype=bool
                    ),
                    "missing_prism_daily": np.asarray(
                        raw_sentinel["missing_prism_daily"][row_idx], dtype=bool
                    ),
                }
            )
        if prism is not None:
            assert prism_idx is not None
            arrays.update(
                {
                    "prism_normals": np.asarray(prism["prism_normals"][prism_idx], dtype=np.float32),
                    "prism_normals_feature_names": np.asarray(prism["prism_normals_feature_names"]),
                    "prism_normals_source_row": np.asarray(prism["prism_normals_source_row"][prism_idx], dtype=np.int64),
                    "missing_prism_normals": np.asarray(prism["missing_prism_normals"][prism_idx], dtype=bool),
                    "source_prism_row": prism_idx.astype(np.int64),
                }
            )

        input_path = out_dir / f"{source_key}_inputs.npz"
        metadata_path = out_dir / f"{source_key}_metadata.csv"
        if not args.dry_run and input_path.exists() and not args.force:
            raise FileExistsError(f"{input_path} exists; pass --force to overwrite")
        if not args.dry_run:
            write_npz(input_path, arrays, args.compress)
            shard_meta.to_csv(metadata_path, index=False)
        summaries.append(
            {
                "city_token": city,
                "source_key": source_key,
                "rows": int(len(shard_meta)),
                "input_path": str(input_path),
                "metadata_path": str(metadata_path),
            }
        )
        action = "would write" if args.dry_run else "wrote"
        print(f"  {action} {source_key}: rows={len(shard_meta):,}", flush=True)
    return summaries


def main() -> int:
    args = parse_args()
    require_clean_roots(args)
    cities = discover_cities(args)
    if not cities:
        raise SystemExit(f"No clean QA city files found under {args.qa_root}")
    print(f"Assembling clean complete shards for {len(cities):,} city/cities: {', '.join(cities)}", flush=True)
    if not args.dry_run:
        args.output_root.mkdir(parents=True, exist_ok=True)
    all_summaries: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    worker_count = max(1, min(int(args.parallel_workers), len(cities)))
    if worker_count == 1:
        for city in cities:
            try:
                all_summaries.extend(process_city(city, args))
            except Exception as exc:
                print(f"{city}: ERROR {exc}", flush=True)
                failures.append({"city_token": city, "error": str(exc)})
                if args.fail_fast:
                    break
    else:
        print(f"Processing cities with parallel_workers={worker_count}", flush=True)
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            future_to_city = {executor.submit(process_city, city, args): city for city in cities}
            for future in as_completed(future_to_city):
                city = future_to_city[future]
                try:
                    all_summaries.extend(future.result())
                except Exception as exc:
                    print(f"{city}: ERROR {exc}", flush=True)
                    failures.append({"city_token": city, "error": str(exc)})
    summary_path = args.output_root / "clean_tree_id_centered_complete_shards_summary.csv"
    if args.dry_run:
        print(f"Dry run: would write summary: {summary_path}", flush=True)
    else:
        summary = pd.DataFrame(all_summaries)
        if args.city_token is not None and summary_path.exists():
            existing = pd.read_csv(summary_path)
            rebuilt_cities = set(cities)
            existing = existing.loc[
                ~existing["city_token"].astype(str).map(normalize_token).isin(rebuilt_cities)
            ].copy()
            summary = pd.concat([existing, summary], ignore_index=True)
        summary.sort_values(["city_token", "source_key"], kind="stable").to_csv(
            summary_path,
            index=False,
        )
        if failures:
            failure_path = args.output_root / "clean_tree_id_centered_complete_shards_failures.csv"
            pd.DataFrame(failures).to_csv(failure_path, index=False)
            print(f"Wrote failures: {failure_path}", flush=True)
        config_path = args.output_root / "clean_tree_id_centered_complete_shards_config.json"
        config_path.write_text(json.dumps(vars(args), indent=2, default=str), encoding="utf-8")
        print(f"Wrote summary: {summary_path}", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
