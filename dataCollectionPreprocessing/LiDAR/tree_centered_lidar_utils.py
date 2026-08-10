"""Shared helpers for TNM LiDAR manifest and tree-centered chip processing."""

from __future__ import annotations

import csv
import json
import re
from collections import Counter
from pathlib import Path
from typing import Callable


HERE = Path(__file__).resolve().parent
DEFAULT_MANIFEST = HERE / "tnm_lidar_download_manifest.csv"
DEFAULT_CITY_SUMMARY = HERE / "tnm_lidar_download_city_summary.csv"
FOOT_TO_METRE = 0.3048


def normalize_token(value: object) -> str:
    text = str(value or "").lower().replace("alberquerque", "albuquerque")
    return re.sub(r"[^a-z0-9]+", "", text)


def class_code_set(value: str) -> set[int]:
    return {int(part) for part in re.split(r"[,; ]+", value.strip()) if part}


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise SystemExit(f"Missing required file: {path}")
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def load_z_scale_table(path: Path | None) -> list[dict[str, str]]:
    if path is None:
        return []
    if not path.exists():
        raise SystemExit(f"Missing --z-scale-table: {path}")
    return read_csv(path)


def expected_size(row: dict[str, str]) -> int | None:
    value = str(row.get("size_bytes", "")).strip()
    if not value:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def complete_lidar_file(path: Path, row: dict[str, str]) -> bool:
    if not path.exists() or path.stat().st_size == 0:
        return False
    size = expected_size(row)
    if size is None or path.stat().st_size == size:
        return True
    if row.get("status") not in {"downloaded", "skipped_existing"}:
        return False
    try:
        recorded_bytes = int(str(row.get("downloaded_bytes", "")).strip())
    except ValueError:
        return False
    return recorded_bytes == path.stat().st_size


def manifest_rows_by_city(rows: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        code = str(row.get("canonical_city_code") or "").upper()
        if code:
            grouped.setdefault(code, []).append(row)
    return grouped


def city_matches(row: dict[str, str], filters: set[str]) -> bool:
    if not filters:
        return True
    tokens = {
        normalize_token(row.get("city_token")),
        normalize_token(row.get("city_name")),
        normalize_token(row.get("city_code")),
        normalize_token(row.get("selected_project")),
    }
    return bool(tokens & filters)


def eligible_summary_rows(summary_rows: list[dict[str, str]], filters: set[str]) -> list[dict[str, str]]:
    cities = []
    for row in summary_rows:
        if not city_matches(row, filters):
            continue
        try:
            kept = int(float(row.get("kept_tile_count") or 0))
        except ValueError:
            kept = 0
        selected = str(row.get("selected_project") or "").strip().lower()
        if kept > 0 and selected not in {"", "none", "all"}:
            cities.append(row)
    return cities


def parse_candidate_project_stats(text: str) -> dict[str, dict[str, float | int | bool]]:
    stats: dict[str, dict[str, float | int | bool]] = {}
    for part in str(text or "").split(";"):
        if not part.strip() or ":" not in part:
            continue
        project, payload = part.split(":", 1)
        record: dict[str, float | int | bool] = {"coverage": 0.0, "year": 0, "event": False, "stale": False, "tiles": 0}
        for item in payload.split(","):
            if "=" not in item:
                continue
            key, value = (piece.strip() for piece in item.split("=", 1))
            try:
                if key == "coverage":
                    record[key] = float(value)
                elif key in {"year", "tiles", "footprints"}:
                    record[key] = int(float(value))
                elif key in {"event", "stale"}:
                    record[key] = bool(int(float(value)))
            except ValueError:
                continue
        stats[project.strip()] = record
    return stats


def project_preference_order(summary_row: dict[str, str], projects: set[str]) -> list[str]:
    selected = str(summary_row.get("selected_project") or "").strip()
    stats = parse_candidate_project_stats(summary_row.get("candidate_projects", ""))
    ordered = [selected] if selected in projects else []
    remaining = [project for project in projects if project not in ordered]
    remaining.sort(key=lambda project: (
        bool(stats.get(project, {}).get("stale", False)),
        bool(stats.get(project, {}).get("event", False)),
        -int(stats.get(project, {}).get("year", 0)),
        -float(stats.get(project, {}).get("coverage", 0.0)),
        project,
    ))
    return ordered + remaining


def safe_project_token(project: str) -> str:
    token = re.sub(r"[^A-Za-z0-9._-]+", "_", str(project).strip()).strip("._")
    return token or "unknown"


def transform_points(data, target_epsg: int, fallback_source_epsg: int):
    import numpy as np
    import pyproj

    x, y = np.asarray(data.x), np.asarray(data.y)
    try:
        source_crs = data.header.parse_crs()
    except Exception:
        source_crs = None
    if source_crs is None and fallback_source_epsg > 0:
        source_crs = pyproj.CRS.from_epsg(fallback_source_epsg)
    if source_crs is None:
        return x, y, "untransformed"
    source_crs = pyproj.CRS.from_user_input(source_crs)
    target_crs = pyproj.CRS.from_epsg(target_epsg)
    if source_crs == target_crs:
        return x, y, source_crs.to_string()
    tx, ty = pyproj.Transformer.from_crs(source_crs, target_crs, always_xy=True).transform(x, y)
    return np.asarray(tx), np.asarray(ty), f"{source_crs}->{target_crs}"


def open_or_create_memmaps(paths: dict[str, Path], shape: tuple[int, int, int], overwrite: bool):
    import numpy as np

    paths["bin_dir"].mkdir(parents=True, exist_ok=True)
    files = [paths[key] for key in ("ground_min", "surface_max", "ground_count", "surface_count", "all_count")]
    if overwrite:
        for path in files + [paths["bin_marker"], paths["bin_state"]]:
            path.unlink(missing_ok=True)
    if any(path.exists() for path in files) and not all(path.exists() for path in files):
        raise RuntimeError(f"Incomplete bin arrays in {paths['bin_dir']}; rerun with --overwrite")
    mode = "r+" if all(path.exists() for path in files) else "w+"
    arrays = (
        np.lib.format.open_memmap(files[0], mode=mode, dtype=np.float32, shape=shape),
        np.lib.format.open_memmap(files[1], mode=mode, dtype=np.float32, shape=shape),
        np.lib.format.open_memmap(files[2], mode=mode, dtype=np.uint32, shape=shape),
        np.lib.format.open_memmap(files[3], mode=mode, dtype=np.uint32, shape=shape),
        np.lib.format.open_memmap(files[4], mode=mode, dtype=np.uint32, shape=shape),
    )
    if mode == "w+":
        arrays[0][:], arrays[1][:] = np.inf, -np.inf
        for array in arrays[2:]:
            array[:] = 0
        for array in arrays:
            array.flush()
    return arrays


def close_memmap(array) -> None:
    try:
        array.flush()
    except Exception:
        pass
    mmap = getattr(array, "_mmap", None)
    if mmap is not None:
        mmap.close()


def write_bin_state(path: Path, state: dict) -> None:
    path.write_text(json.dumps(state, indent=2), encoding="utf-8")


def explicit_z_scale_for_job(job, args) -> tuple[float, str, str, str] | None:
    for row in getattr(args, "z_scale_rows", []):
        project = str(row.get("project") or "").strip()
        city_code = normalize_token(row.get("city_code", ""))
        city_token = normalize_token(row.get("city_token", ""))
        if (not project or project == job.project) and (not (city_code or city_token) or city_code == normalize_token(job.code) or city_token == normalize_token(job.token)):
            try:
                scale = float(row.get("z_scale") or args.z_scale)
            except ValueError:
                scale = float(args.z_scale)
            return scale, str(row.get("z_units") or args.z_units), str(row.get("confidence") or "override"), str(row.get("reason") or "z_scale_table")
    return None


def unit_kind(unit_name: str, conversion_factor: float | None = None) -> str:
    unit = str(unit_name or "").lower()
    if "foot" in unit or "feet" in unit or unit.strip() == "ft":
        return "feet"
    if "metre" in unit or "meter" in unit or unit.strip() in {"m", "metres", "meters"}:
        return "metres"
    if conversion_factor is not None and abs(float(conversion_factor) - FOOT_TO_METRE) < 0.002:
        return "feet"
    if conversion_factor is not None and abs(float(conversion_factor) - 1.0) < 0.0001:
        return "metres"
    return "unknown"


def audit_tile_units(path: Path) -> dict[str, object]:
    import laspy

    with laspy.open(str(path)) as reader:
        try:
            crs = reader.header.parse_crs()
        except Exception:
            crs = None
        vertical, horizontal = [], []
        if crs is not None:
            for candidate in list(getattr(crs, "sub_crs_list", []) or []) or [crs]:
                for axis in list(getattr(candidate, "axis_info", []) or []):
                    name = str(getattr(axis, "unit_name", "") or "")
                    factor = getattr(axis, "unit_conversion_factor", None)
                    kind = unit_kind(name, factor)
                    direction = str(getattr(axis, "direction", "") or "").lower()
                    (vertical if getattr(candidate, "is_vertical", False) or direction in {"up", "down"} else horizontal).append(kind)
        return {"vertical_kinds": vertical, "horizontal_kinds": horizontal, "filename": path.name}


def infer_z_scale_for_job(job, args, tile_paths_fn: Callable) -> tuple[float, str, str, str]:
    explicit = explicit_z_scale_for_job(job, args)
    if explicit is not None:
        return explicit
    if args.auto_z_scale:
        records = []
        for path in tile_paths_fn(job, args)[: max(1, int(args.z_scale_audit_tiles))]:
            try:
                records.append(audit_tile_units(path))
            except Exception:
                continue
        for field, confidence in (("vertical_kinds", "high"), ("horizontal_kinds", "medium")):
            kinds = Counter(kind for record in records for kind in record[field] if kind != "unknown")
            if kinds:
                kind, count = kinds.most_common(1)[0]
                scale = FOOT_TO_METRE if kind == "feet" else 1.0
                return scale, "meters", confidence, f"{field}_{kind}:{count}/{sum(kinds.values())}"
    return float(args.z_scale), str(args.z_units), "fallback", "default_or_cli_z_scale"


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["city_token", "city_name", "city_code", "candidate_project", "preference_rank", "stage", "status", "reason", "tree_crop_rows", "manifest_rows", "tile_count", "started_at", "finished_at", "bin_marker", "dtm_path", "dsm_path", "chm_path", "lidar_index_path"]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
