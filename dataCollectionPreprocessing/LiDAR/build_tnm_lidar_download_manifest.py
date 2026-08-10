#!/usr/bin/env python3
"""Build a deduplicated TNM LiDAR download manifest for study cities.

This is the discovery half of the LiDAR workflow. It queries the same USGS
National Map LPC/LAZ catalog used for the Albuquerque test, estimates which
catalog tiles cover expanded Sentinel-2 chip footprints, selects one best
LiDAR project per city, deduplicates tiles across cities,
and writes a restart-friendly manifest for the downloader.

The manifest keeps one physical download row per unique URL. If a tile covers
multiple study cities, all cities are recorded in the row and the file is
assigned to the city with the most sampled chip-footprint hits.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import ssl
import time
import urllib.parse
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

from identify_tnm_lidar_city_coverage import (
    TNM_PRODUCTS_URL,
    catalog_tiles,
    cell_center_xy,
    city_bbox,
    city_cell_tables,
    normalize_token,
    read_unique_cells,
    sample_cells,
    ssl_context,
    utm_to_lonlat,
)


HERE = Path(__file__).resolve().parent
DEFAULT_CELL_MAP_DIR = HERE.parents[0] / "Sentinel2" / "mccoy_sentinel_10m_cells_utm"
DEFAULT_CITY_MANIFEST = HERE.parents[0] / "NAIP" / "naip_county_manifest.json"
DEFAULT_OUTPUT_CSV = HERE / "tnm_lidar_download_manifest.csv"
DEFAULT_OUTPUT_JSON = HERE / "tnm_lidar_download_manifest.json"
DEFAULT_CITY_SUMMARY_CSV = HERE / "tnm_lidar_download_city_summary.csv"
DEFAULT_LIDAR_ROOT = Path(r"E:\LiDAR")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--cell-map-dir", type=Path, default=DEFAULT_CELL_MAP_DIR)
    parser.add_argument("--city-manifest", type=Path, default=DEFAULT_CITY_MANIFEST)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--city-summary-csv", type=Path, default=DEFAULT_CITY_SUMMARY_CSV)
    parser.add_argument("--lidar-root", type=Path, default=DEFAULT_LIDAR_ROOT)
    parser.add_argument("--city-token", action="append", default=[], help="Optional city folder/name/code filter. Repeatable.")
    parser.add_argument("--project", action="append", default=[], help="Optional TNM project filter. Repeatable.")
    parser.add_argument(
        "--project-selection",
        choices=("best", "all"),
        default="best",
        help="Keep only the best project per city by default, or keep all matching projects.",
    )
    parser.add_argument(
        "--project-coverage-tie-fraction",
        type=float,
        default=1.0,
        help="Projects covering at least this fraction of the max footprint coverage can win by newer year. Default requires equal coverage.",
    )
    parser.add_argument(
        "--event-project-keyword",
        action="append",
        default=["flood", "wildfire", "hurricane", "storm", "sandy"],
        help="Project/title keyword treated as event-specific and avoided when a good non-event project exists. Repeatable.",
    )
    parser.add_argument(
        "--allow-event-projects",
        action="store_true",
        help="Allow event-specific projects such as flood LiDAR to be selected normally.",
    )
    parser.add_argument(
        "--min-non-event-coverage",
        type=float,
        default=0.90,
        help="Minimum footprint coverage needed before a non-event project can displace an event-specific project.",
    )
    parser.add_argument(
        "--min-project-year",
        type=int,
        default=2012,
        help="Minimum acquisition/project year to select by default. Older projects are skipped unless --allow-stale-projects is used.",
    )
    parser.add_argument(
        "--preferred-project-year",
        type=int,
        default=2017,
        help="Prefer projects at least this recent when they meet --min-preferred-coverage.",
    )
    parser.add_argument(
        "--min-preferred-coverage",
        type=float,
        default=0.85,
        help="Minimum footprint coverage needed for a preferred-year project to beat older fuller-coverage projects.",
    )
    parser.add_argument(
        "--allow-stale-projects",
        action="store_true",
        help="Allow projects older than --min-project-year to be selected.",
    )
    parser.add_argument("--bbox-buffer-metres", type=float, default=500.0)
    parser.add_argument("--sentinel-cell-size", type=float, default=10.0)
    parser.add_argument("--sentinel-origin-x", type=float, default=0.0)
    parser.add_argument("--sentinel-origin-y", type=float, default=0.0)
    parser.add_argument("--chip-half-size-metres", type=float, default=10.0, help="Half-width of the LiDAR/NAIP chip footprint.")
    parser.add_argument("--coverage-margin-metres", type=float, default=15.0, help="Extra margin around each chip footprint when keeping border tiles.")
    parser.add_argument("--max-results", type=int, default=500)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--sleep", type=float, default=0.25, help="Pause between city catalog queries.")
    parser.add_argument("--max-coverage-points", type=int, default=5000, help="Subsample cell footprints for coverage scoring. 0 means all.")
    parser.add_argument("--max-tiles-per-city", type=int, default=0, help="0 means all matching catalog tiles.")
    parser.add_argument("--include-bbox-only", action="store_true", help="Keep selected-project tiles that hit the city bbox even if no sampled chip footprint intersects them.")
    parser.add_argument("--no-verify-tls", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Do not write output files.")
    return parser.parse_args()


def load_city_lookup(path: Path) -> dict[str, dict[str, str]]:
    lookup: dict[str, dict[str, str]] = {}
    if not path.exists():
        return lookup
    data = json.loads(path.read_text(encoding="utf-8"))
    for city in data.get("cities", []):
        name = str(city.get("city", "")).strip()
        code = str(city.get("code", "")).strip().upper()
        if not name or not code:
            continue
        record = {"city_name": name, "city_code": code}
        lookup[normalize_token(name)] = record
        lookup[normalize_token(code)] = record
    return lookup


def city_label(folder_token: str, city_lookup: dict[str, dict[str, str]]) -> tuple[str, str]:
    lookup = city_lookup.get(normalize_token(folder_token), {})
    city_name = lookup.get("city_name") or folder_token
    city_code = lookup.get("city_code") or normalize_token(folder_token)[:3].upper()
    if normalize_token(folder_token) == "albuquerque":
        city_code = "ABQ"
    return city_name, city_code


def download_filename(url: str) -> str:
    filename = Path(urllib.parse.unquote(urllib.parse.urlparse(url).path)).name
    if not filename.lower().endswith(".laz"):
        raise ValueError(f"Expected LAZ URL, got {url}")
    return filename


def download_host(url: str) -> str:
    return urllib.parse.urlparse(url).netloc.lower()


def size_from_tile(tile: dict) -> str:
    for key in ("size_bytes", "sizeInBytes", "size", "fileSize"):
        value = tile.get(key)
        if value not in (None, ""):
            try:
                return str(int(value))
            except (TypeError, ValueError):
                return str(value)
    return ""


def sample_footprints(cells: list[dict], args: argparse.Namespace) -> list[tuple[float, float, float, float]]:
    half_size = args.chip_half_size_metres + args.coverage_margin_metres
    footprints = []
    for cell in sample_cells(cells, args.max_coverage_points):
        center_x, center_y = cell_center_xy(cell, args)
        epsg = int(cell["cell_epsg"])
        corners = [
            (center_x - half_size, center_y - half_size),
            (center_x - half_size, center_y + half_size),
            (center_x + half_size, center_y - half_size),
            (center_x + half_size, center_y + half_size),
        ]
        lonlats = [utm_to_lonlat(x, y, epsg) for x, y in corners]
        lons = [lon for lon, _ in lonlats]
        lats = [lat for _, lat in lonlats]
        footprints.append((min(lons), min(lats), max(lons), max(lats)))
    return footprints


def bbox_intersects(left: tuple[float, float, float, float], right: tuple[float, float, float, float]) -> bool:
    left_min_x, left_min_y, left_max_x, left_max_y = left
    right_min_x, right_min_y, right_max_x, right_max_y = right
    return not (
        left_max_x < right_min_x
        or left_min_x > right_max_x
        or left_max_y < right_min_y
        or left_min_y > right_max_y
    )


def tile_bbox(tile: dict) -> tuple[float, float, float, float] | None:
    try:
        values = (tile["min_x"], tile["min_y"], tile["max_x"], tile["max_y"])
        if any(value in (None, "") for value in values):
            return None
        return tuple(float(value) for value in values)
    except (KeyError, TypeError, ValueError):
        return None


def tile_footprint_hit_indices(tile: dict, footprints: list[tuple[float, float, float, float]]) -> set[int]:
    bbox = tile_bbox(tile)
    if bbox is None:
        return set()
    return {index for index, footprint in enumerate(footprints) if bbox_intersects(bbox, footprint)}


def project_year(tile: dict) -> int:
    max_year = datetime.now().year + 1
    project_text = " ".join(str(tile.get(key, "") or "") for key in ("project", "title"))
    project_years = [
        int(match)
        for match in re.findall(r"(?<!\d)((?:19|20)\d{2})(?!\d)", project_text)
        if 1980 <= int(match) <= max_year
    ]
    if project_years:
        return max(project_years)
    catalog_text = " ".join(str(tile.get(key, "") or "") for key in ("date_created", "last_updated"))
    catalog_years = [
        int(match)
        for match in re.findall(r"(?<!\d)((?:19|20)\d{2})(?!\d)", catalog_text)
        if 1980 <= int(match) <= max_year
    ]
    return max(catalog_years) if catalog_years else 0


def is_event_project(tile: dict, args: argparse.Namespace) -> bool:
    if args.allow_event_projects:
        return False
    text = " ".join(str(tile.get(key, "") or "") for key in ("project", "title")).lower()
    return any(str(keyword or "").lower() in text for keyword in args.event_project_keyword)


def is_stale_project(year: int, args: argparse.Namespace) -> bool:
    return not args.allow_stale_projects and 0 < year < args.min_project_year


def project_summary_text(stats: dict[str, dict], footprint_count: int) -> str:
    parts = []
    for project, record in sorted(
        stats.items(),
        key=lambda item: (-len(item[1]["footprints"]), -int(item[1]["year"]), str(item[0])),
    ):
        coverage = len(record["footprints"]) / footprint_count if footprint_count else 0.0
        parts.append(
            f"{project}:tiles={record['tile_count']},footprints={len(record['footprints'])},"
            f"coverage={coverage:.3f},year={record['year']},"
            f"event={int(bool(record.get('event_project')))},stale={int(bool(record.get('stale_project')))}"
        )
    return ";".join(parts)


def select_project(
    candidates: list[tuple[dict, int, set[int]]],
    footprint_count: int,
    args: argparse.Namespace,
) -> tuple[str, dict[str, dict], str]:
    stats: dict[str, dict] = defaultdict(
        lambda: {
            "tile_count": 0,
            "total_hits": 0,
            "footprints": set(),
            "year": 0,
            "event_project": False,
            "stale_project": False,
        }
    )
    for tile, hits, hit_indices in candidates:
        project = str(tile.get("project", "") or "unknown")
        stats[project]["tile_count"] += 1
        stats[project]["total_hits"] += hits
        stats[project]["footprints"].update(hit_indices)
        stats[project]["year"] = max(int(stats[project]["year"]), project_year(tile))
        stats[project]["event_project"] = bool(stats[project]["event_project"]) or is_event_project(tile, args)

    if not stats:
        return "", stats, "no candidate projects"

    for record in stats.values():
        record["stale_project"] = is_stale_project(int(record["year"]), args)

    eligible_stats = stats
    stale_projects = [project for project, record in stats.items() if record.get("stale_project")]
    current_projects = [project for project, record in stats.items() if not record.get("stale_project")]
    if stale_projects:
        if current_projects:
            eligible_stats = {project: stats[project] for project in current_projects}
        else:
            newest_stale_year = max(int(record["year"]) for record in stats.values())
            return (
                "",
                stats,
                f"no eligible project >= {args.min_project_year}; newest candidate year={newest_stale_year}; "
                "use --allow-stale-projects to include older LiDAR",
            )

    preferred_projects = [
        project
        for project, record in eligible_stats.items()
        if int(record["year"]) >= args.preferred_project_year
        and (len(record["footprints"]) / footprint_count if footprint_count else 0.0) >= args.min_preferred_coverage
    ]
    if preferred_projects:
        eligible_stats = {project: eligible_stats[project] for project in preferred_projects}

    event_projects = [project for project, record in stats.items() if record.get("event_project")]
    non_event_projects = [project for project, record in eligible_stats.items() if not record.get("event_project")]
    if event_projects and non_event_projects:
        best_non_event_coverage = max(len(stats[project]["footprints"]) for project in non_event_projects)
        non_event_fraction = best_non_event_coverage / footprint_count if footprint_count else 0.0
        if non_event_fraction >= args.min_non_event_coverage:
            eligible_stats = {project: stats[project] for project in non_event_projects}

    max_coverage = max(len(record["footprints"]) for record in eligible_stats.values())
    tied_projects = [
        project
        for project, record in eligible_stats.items()
        if max_coverage == 0
        or len(record["footprints"]) >= max_coverage * args.project_coverage_tie_fraction
    ]
    selected = max(
        tied_projects,
        key=lambda project: (
            int(stats[project]["year"]),
            len(stats[project]["footprints"]),
            int(stats[project]["total_hits"]),
            -int(stats[project]["tile_count"]),
            project,
        ),
    )
    selected_record = stats[selected]
    coverage = len(selected_record["footprints"]) / footprint_count if footprint_count else 0.0
    reason = (
        f"selected {selected} by footprint_coverage={coverage:.3f}, "
        f"covered_footprints={len(selected_record['footprints'])}/{footprint_count}, "
        f"year={selected_record['year']}, tiles={selected_record['tile_count']}, "
        f"event_project={int(bool(selected_record.get('event_project')))}, "
        f"stale_project={int(bool(selected_record.get('stale_project')))}"
    )
    if event_projects and selected not in event_projects:
        reason += f", avoided_event_projects={','.join(sorted(event_projects))}"
    if stale_projects:
        reason += f", avoided_stale_projects={','.join(sorted(stale_projects))}"
    if preferred_projects:
        reason += (
            f", preferred_year_filter={args.preferred_project_year}+"
            f"@{args.min_preferred_coverage:.3f}"
        )
    return selected, stats, reason


def complete_existing(path: Path, expected_size: str) -> tuple[bool, int]:
    if not path.exists() or path.stat().st_size == 0:
        return False, path.stat().st_size if path.exists() else 0
    size = path.stat().st_size
    if expected_size:
        try:
            return size == int(expected_size), size
        except ValueError:
            return True, size
    return True, size


def merge_tile(
    rows_by_url: dict[str, dict],
    tile: dict,
    city_token: str,
    city_name: str,
    city_code: str,
    hit_count: int,
    args: argparse.Namespace,
    lidar_root: Path,
) -> None:
    url = str(tile["url"])
    filename = download_filename(url)
    existing = rows_by_url.get(url)
    if existing is None:
        host = download_host(url)
        relative_path = str(Path(city_code) / filename)
        expected_size = size_from_tile(tile)
        local_path = lidar_root / relative_path
        complete, existing_bytes = complete_existing(local_path, expected_size)
        existing = {
            "url": url,
            "filename": filename,
            "host": host,
            "project": str(tile.get("project", "") or "unknown"),
            "canonical_city_token": city_token,
            "canonical_city_name": city_name,
            "canonical_city_code": city_code,
            "city_tokens": city_token,
            "city_names": city_name,
            "city_codes": city_code,
            "city_hit_counts": f"{city_token}:{hit_count}",
            "city_count": "1",
            "total_sample_hits": str(hit_count),
            "max_city_sample_hits": str(hit_count),
            "coverage_filter": "expanded_chip_footprint",
            "chip_half_size_metres": str(args.chip_half_size_metres),
            "coverage_margin_metres": str(args.coverage_margin_metres),
            "min_project_year": str(args.min_project_year),
            "preferred_project_year": str(args.preferred_project_year),
            "min_preferred_coverage": str(args.min_preferred_coverage),
            "relative_path": relative_path,
            "local_path": str(local_path),
            "status": "skipped_existing" if complete else "queued",
            "existing_bytes": str(existing_bytes),
            "size_bytes": expected_size,
            "title": str(tile.get("title", "")),
            "source_id": str(tile.get("source_id", "")),
            "date_created": str(tile.get("date_created", "")),
            "last_updated": str(tile.get("last_updated", "")),
            "min_x": str(tile.get("min_x", "")),
            "min_y": str(tile.get("min_y", "")),
            "max_x": str(tile.get("max_x", "")),
            "max_y": str(tile.get("max_y", "")),
            "last_error": "",
        }
        rows_by_url[url] = existing
        return

    tokens = existing["city_tokens"].split(";") if existing["city_tokens"] else []
    names = existing["city_names"].split(";") if existing["city_names"] else []
    codes = existing["city_codes"].split(";") if existing["city_codes"] else []
    if city_token not in tokens:
        tokens.append(city_token)
        names.append(city_name)
        codes.append(city_code)
        existing["city_tokens"] = ";".join(tokens)
        existing["city_names"] = ";".join(names)
        existing["city_codes"] = ";".join(codes)
        existing["city_count"] = str(len(tokens))
    counts = existing["city_hit_counts"].split(";") if existing["city_hit_counts"] else []
    counts = [item for item in counts if not item.startswith(f"{city_token}:")]
    counts.append(f"{city_token}:{hit_count}")
    existing["city_hit_counts"] = ";".join(counts)
    existing["total_sample_hits"] = str(int(existing["total_sample_hits"] or 0) + hit_count)

    if hit_count > int(existing["max_city_sample_hits"] or 0):
        existing["canonical_city_token"] = city_token
        existing["canonical_city_name"] = city_name
        existing["canonical_city_code"] = city_code
        existing["max_city_sample_hits"] = str(hit_count)
        relative_path = str(Path(city_code) / existing["filename"])
        existing["relative_path"] = relative_path
        existing["local_path"] = str(lidar_root / relative_path)


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    city_lookup = load_city_lookup(args.city_manifest)
    city_filters = {normalize_token(value) for value in args.city_token}
    context = ssl_context(args.no_verify_tls)

    rows_by_url: dict[str, dict] = {}
    city_summaries: list[dict] = []

    for folder_token, table in city_cell_tables(args.cell_map_dir):
        city_name, city_code = city_label(folder_token, city_lookup)
        filter_tokens = {normalize_token(folder_token), normalize_token(city_name), normalize_token(city_code)}
        if city_filters and not (city_filters & filter_tokens):
            continue

        cells = read_unique_cells(table)
        if not cells:
            continue
        bbox = city_bbox(cells, args)
        bbox_text = ",".join(f"{value:.8f}" for value in bbox)
        footprints = sample_footprints(cells, args)

        print(
            f"{folder_token} ({city_code}): {len(cells):,} cell(s), "
            f"footprint_sample={len(footprints):,}, "
            f"chip_half={args.chip_half_size_metres:g}m, "
            f"margin={args.coverage_margin_metres:g}m, bbox={bbox_text}"
        )
        tiles = catalog_tiles(bbox, args, context)
        candidates = []
        for tile in tiles:
            hit_indices = tile_footprint_hit_indices(tile, footprints)
            hits = len(hit_indices)
            if hits > 0 or args.include_bbox_only:
                candidates.append((tile, hits, hit_indices))

        selected_project = ""
        project_stats: dict[str, dict] = {}
        selection_reason = "project_selection=all"
        if args.project_selection == "best":
            selected_project, project_stats, selection_reason = select_project(candidates, len(footprints), args)
            kept = [(tile, hits) for tile, hits, _ in candidates if str(tile.get("project", "") or "unknown") == selected_project]
        else:
            for tile, hits, hit_indices in candidates:
                project = str(tile.get("project", "") or "unknown")
                project_stats.setdefault(
                    project,
                    {"tile_count": 0, "total_hits": 0, "footprints": set(), "year": 0, "event_project": False, "stale_project": False},
                )
                project_stats[project]["tile_count"] += 1
                project_stats[project]["total_hits"] += hits
                project_stats[project]["footprints"].update(hit_indices)
                project_stats[project]["year"] = max(int(project_stats[project]["year"]), project_year(tile))
                project_stats[project]["event_project"] = bool(project_stats[project]["event_project"]) or is_event_project(tile, args)
            for record in project_stats.values():
                record["stale_project"] = is_stale_project(int(record["year"]), args)
            kept = [(tile, hits) for tile, hits, _ in candidates]
        kept.sort(key=lambda item: (-item[1], str(item[0].get("project", "")), download_filename(str(item[0]["url"]))))
        if args.max_tiles_per_city:
            kept = kept[: args.max_tiles_per_city]

        for tile, hits in kept:
            merge_tile(rows_by_url, tile, folder_token, city_name, city_code, hits, args, args.lidar_root)

        projects = Counter(str(tile.get("project", "") or "unknown") for tile, _ in kept)
        hosts = Counter(download_host(str(tile["url"])) for tile, _ in kept)
        city_summaries.append(
            {
                "city_token": folder_token,
                "city_name": city_name,
                "city_code": city_code,
                "cell_count": str(len(cells)),
                "sampled_cell_footprints": str(len(footprints)),
                "chip_half_size_metres": str(args.chip_half_size_metres),
                "coverage_margin_metres": str(args.coverage_margin_metres),
                "min_project_year": str(args.min_project_year),
                "preferred_project_year": str(args.preferred_project_year),
                "min_preferred_coverage": str(args.min_preferred_coverage),
                "bbox": bbox_text,
                "catalog_tile_hits": str(len(tiles)),
                "candidate_tile_count": str(len(candidates)),
                "kept_tile_count": str(len(kept)),
                "project_selection": args.project_selection,
                "selected_project": selected_project or ("all" if args.project_selection == "all" else "none"),
                "project_selection_reason": selection_reason,
                "project_count": str(len(projects)),
                "top_projects": ";".join(f"{name}:{count}" for name, count in projects.most_common(5)),
                "candidate_projects": project_summary_text(project_stats, len(footprints)),
                "host_count": str(len(hosts)),
                "hosts": ";".join(f"{name}:{count}" for name, count in hosts.most_common()),
            }
        )
        print(f"  {selection_reason}")
        print(f"  kept={len(kept):,}/{len(tiles):,}, candidates={len(candidates):,}, projects={len(projects):,}, hosts={dict(hosts)}")
        time.sleep(args.sleep)

    rows = sorted(
        rows_by_url.values(),
        key=lambda row: (
            0 if row["status"] == "queued" else 1,
            -int(row["max_city_sample_hits"] or 0),
            row["canonical_city_code"],
            row["filename"],
        ),
    )
    for index, row in enumerate(rows, start=1):
        row["priority_rank"] = str(index)

    manifest_fields = [
        "priority_rank",
        "status",
        "url",
        "filename",
        "host",
        "project",
        "canonical_city_token",
        "canonical_city_name",
        "canonical_city_code",
        "city_tokens",
        "city_names",
        "city_codes",
        "city_hit_counts",
        "city_count",
        "total_sample_hits",
        "max_city_sample_hits",
        "coverage_filter",
        "chip_half_size_metres",
        "coverage_margin_metres",
        "min_project_year",
        "preferred_project_year",
        "min_preferred_coverage",
        "relative_path",
        "local_path",
        "existing_bytes",
        "size_bytes",
        "title",
        "source_id",
        "date_created",
        "last_updated",
        "min_x",
        "min_y",
        "max_x",
        "max_y",
        "last_error",
    ]
    summary_fields = [
        "city_token",
        "city_name",
        "city_code",
        "cell_count",
        "sampled_cell_footprints",
        "chip_half_size_metres",
        "coverage_margin_metres",
        "min_project_year",
        "preferred_project_year",
        "min_preferred_coverage",
        "bbox",
        "catalog_tile_hits",
        "candidate_tile_count",
        "kept_tile_count",
        "project_selection",
        "selected_project",
        "project_selection_reason",
        "project_count",
        "top_projects",
        "candidate_projects",
        "host_count",
        "hosts",
    ]

    queued = sum(1 for row in rows if row["status"] == "queued")
    skipped = sum(1 for row in rows if row["status"] == "skipped_existing")
    host_counts = Counter(row["host"] for row in rows)
    project_counts = Counter(row["project"] for row in rows)
    print(f"Manifest rows={len(rows):,}; queued={queued:,}; skipped_existing={skipped:,}")
    print(f"Hosts: {dict(host_counts)}")
    print(f"Top projects: {dict(project_counts.most_common(10))}")

    if args.dry_run:
        for row in rows[:10]:
            print(f"  {row['status']} {row['canonical_city_code']} {row['filename']} {row['url']}")
        if len(rows) > 10:
            print(f"  ... {len(rows) - 10:,} more")
        return 0

    write_csv(args.output_csv, rows, manifest_fields)
    args.output_json.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    write_csv(args.city_summary_csv, city_summaries, summary_fields)
    print(f"Wrote {args.output_csv}")
    print(f"Wrote {args.output_json}")
    print(f"Wrote {args.city_summary_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
