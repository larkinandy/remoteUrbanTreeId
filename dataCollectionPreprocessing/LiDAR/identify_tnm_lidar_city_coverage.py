#!/usr/bin/env python3
"""Identify which study cities have USGS TNM LiDAR Point Cloud coverage.

This uses the same USGS/The National Map products API used by the Albuquerque
LiDAR downloader, but it does not download tiles. For each Sentinel-2 city
folder, it:

  1. reads unique Sentinel cell centers,
  2. converts their UTM coordinates to WGS84 lon/lat,
  3. queries TNM Lidar Point Cloud (LPC) LAZ products for the city bbox,
  4. reports matching tile/project counts and approximate cell-center coverage.

Catalog bounding boxes are a coarse pre-screen, not proof that every 20 m chip
contains LiDAR points. Use the downstream chip/raster coverage test before
deriving products.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import ssl
import time
import urllib.parse
import urllib.request
from collections import Counter
from pathlib import Path


HERE = Path(__file__).resolve().parent
DEFAULT_CELL_MAP_DIR = HERE.parents[0] / "Sentinel2" / "mccoy_sentinel_10m_cells_utm"
DEFAULT_OUTPUT_CSV = HERE / "tnm_lidar_city_coverage.csv"
DEFAULT_OUTPUT_JSON = HERE / "tnm_lidar_city_coverage.json"
TNM_PRODUCTS_URL = "https://tnmaccess.nationalmap.gov/api/v1/products"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--cell-map-dir", type=Path, default=DEFAULT_CELL_MAP_DIR)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--city-token", default="", help="Optional city folder/name filter.")
    parser.add_argument("--project", action="append", default=[], help="Optional TNM project filter. Repeatable.")
    parser.add_argument("--bbox-buffer-metres", type=float, default=500.0)
    parser.add_argument("--sentinel-cell-size", type=float, default=10.0)
    parser.add_argument("--sentinel-origin-x", type=float, default=0.0)
    parser.add_argument("--sentinel-origin-y", type=float, default=0.0)
    parser.add_argument("--max-results", type=int, default=500)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--sleep", type=float, default=0.2, help="Pause between city API queries.")
    parser.add_argument("--max-coverage-points", type=int, default=5000, help="Subsample city cell centers for bbox coverage estimates. 0 means all.")
    parser.add_argument("--no-verify-tls", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Print city bboxes but do not query TNM.")
    return parser.parse_args()


def normalize_token(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value or "").lower().replace("alberquerque", "albuquerque"))


def parse_epsg(cell_id: str) -> int:
    match = re.search(r"epsg(\d+)", cell_id.lower())
    if not match:
        raise ValueError(f"Could not parse EPSG from cell_id={cell_id!r}")
    return int(match.group(1))


def city_cell_tables(cell_map_dir: Path):
    if cell_map_dir.is_file():
        yield cell_map_dir.parent.name, cell_map_dir
        return
    for path in sorted(cell_map_dir.rglob("tree_to_sentinel10m_cell.csv")):
        yield path.parent.name, path


def read_unique_cells(path: Path) -> list[dict]:
    cells: dict[int, dict] = {}
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"reduced_id", "cell_id", "cell_col", "cell_row"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
        for row in reader:
            reduced_id = int(row["reduced_id"])
            if reduced_id in cells:
                continue
            cell_id = row["cell_id"]
            cells[reduced_id] = {
                "reduced_id": reduced_id,
                "cell_id": cell_id,
                "cell_col": int(row["cell_col"]),
                "cell_row": int(row["cell_row"]),
                "cell_epsg": parse_epsg(cell_id),
            }
    return [cells[key] for key in sorted(cells)]


def utm_to_lonlat(x: float, y: float, epsg: int) -> tuple[float, float]:
    """Convert UTM WGS84/NAD83 coordinates to lon/lat."""
    zone = epsg % 100
    if not 1 <= zone <= 60:
        raise ValueError(f"Cannot infer UTM zone from EPSG:{epsg}")
    northern = epsg in range(32601, 32661) or epsg in range(26901, 26924)
    a = 6378137.0
    e = 0.08181919084262149
    e1sq = e * e / (1.0 - e * e)
    k0 = 0.9996
    x = x - 500000.0
    if not northern:
        y = y - 10000000.0
    lon_origin = (zone - 1) * 6 - 180 + 3
    m = y / k0
    mu = m / (a * (1 - e**2 / 4 - 3 * e**4 / 64 - 5 * e**6 / 256))
    e1 = (1 - (1 - e**2) ** 0.5) / (1 + (1 - e**2) ** 0.5)
    j1 = 3 * e1 / 2 - 27 * e1**3 / 32
    j2 = 21 * e1**2 / 16 - 55 * e1**4 / 32
    j3 = 151 * e1**3 / 96
    j4 = 1097 * e1**4 / 512
    fp = mu + j1 * math.sin(2 * mu) + j2 * math.sin(4 * mu) + j3 * math.sin(6 * mu) + j4 * math.sin(8 * mu)
    c1 = e1sq * math.cos(fp) ** 2
    t1 = math.tan(fp) ** 2
    r1 = a * (1 - e**2) / ((1 - e**2 * math.sin(fp) ** 2) ** 1.5)
    n1 = a / ((1 - e**2 * math.sin(fp) ** 2) ** 0.5)
    d = x / (n1 * k0)
    lat = fp - (n1 * math.tan(fp) / r1) * (
        d**2 / 2
        - (5 + 3 * t1 + 10 * c1 - 4 * c1**2 - 9 * e1sq) * d**4 / 24
        + (61 + 90 * t1 + 298 * c1 + 45 * t1**2 - 252 * e1sq - 3 * c1**2) * d**6 / 720
    )
    lon = math.radians(lon_origin) + (
        d
        - (1 + 2 * t1 + c1) * d**3 / 6
        + (5 - 2 * c1 + 28 * t1 - 3 * c1**2 + 8 * e1sq + 24 * t1**2) * d**5 / 120
    ) / math.cos(fp)
    return math.degrees(lon), math.degrees(lat)


def cell_center_xy(cell: dict, args: argparse.Namespace) -> tuple[float, float]:
    x = args.sentinel_origin_x + (cell["cell_col"] + 0.5) * args.sentinel_cell_size
    y = args.sentinel_origin_y + (cell["cell_row"] + 0.5) * args.sentinel_cell_size
    return x, y


def cell_center_lonlat(cell: dict, args: argparse.Namespace) -> tuple[float, float]:
    x, y = cell_center_xy(cell, args)
    return utm_to_lonlat(x, y, int(cell["cell_epsg"]))


def city_bbox(cells: list[dict], args: argparse.Namespace) -> tuple[float, float, float, float]:
    by_epsg: dict[int, list[tuple[float, float]]] = {}
    for cell in cells:
        by_epsg.setdefault(int(cell["cell_epsg"]), []).append(cell_center_xy(cell, args))

    lonlats = []
    for epsg, xys in by_epsg.items():
        xs = [xy[0] for xy in xys]
        ys = [xy[1] for xy in xys]
        min_x = min(xs) - args.bbox_buffer_metres
        max_x = max(xs) + args.bbox_buffer_metres
        min_y = min(ys) - args.bbox_buffer_metres
        max_y = max(ys) + args.bbox_buffer_metres
        for x, y in [(min_x, min_y), (min_x, max_y), (max_x, min_y), (max_x, max_y)]:
            lonlats.append(utm_to_lonlat(x, y, epsg))
    lons = [lon for lon, _ in lonlats]
    lats = [lat for _, lat in lonlats]
    return min(lons), min(lats), max(lons), max(lats)


def ssl_context(no_verify_tls: bool) -> ssl.SSLContext | None:
    if not no_verify_tls:
        return None
    context = ssl.create_default_context()
    context.check_hostname = False
    context.verify_mode = ssl.CERT_NONE
    return context


def request_json(url: str, timeout: int, context: ssl.SSLContext | None) -> dict:
    request = urllib.request.Request(url, headers={"User-Agent": "remoteUrbanTreeId LiDAR coverage checker"})
    with urllib.request.urlopen(request, timeout=timeout, context=context) as response:
        return json.loads(response.read().decode("utf-8"))


def project_from_url(url: str) -> str:
    legacy = re.search(r"/Projects/legacy/([^/]+)/LAZ/", url)
    if legacy:
        return legacy.group(1)
    staged = re.search(r"/Projects/([^/]+)/", url)
    if staged:
        return staged.group(1)
    return ""


def item_bbox(item: dict) -> tuple[float | None, float | None, float | None, float | None]:
    bbox = item.get("boundingBox") or {}
    try:
        return float(bbox["minX"]), float(bbox["minY"]), float(bbox["maxX"]), float(bbox["maxY"])
    except (KeyError, TypeError, ValueError):
        return None, None, None, None


def size_from_item(item: dict) -> int | None:
    for key in ("sizeInBytes", "size", "fileSize"):
        value = item.get(key)
        if value in (None, ""):
            continue
        try:
            return int(value)
        except (TypeError, ValueError):
            continue
    return None


def catalog_tiles(
    bbox: tuple[float, float, float, float],
    args: argparse.Namespace,
    context: ssl.SSLContext | None,
) -> list[dict]:
    bbox_text = ",".join(f"{value:.8f}" for value in bbox)
    wanted_projects = {project.lower() for project in args.project}
    tiles = []
    seen_urls = set()
    offset = 0
    total = None
    while total is None or offset < total:
        params = urllib.parse.urlencode(
            {
                "datasets": "Lidar Point Cloud (LPC)",
                "bbox": bbox_text,
                "prodFormats": "LAZ",
                "max": args.max_results,
                "offset": offset,
            }
        )
        url = f"{TNM_PRODUCTS_URL}?{params}"
        data = request_json(url, args.timeout, context)
        total = int(data.get("total") or 0)
        items = data.get("items") or []
        if not items:
            break
        for item in items:
            download_url = item.get("downloadURL") or (item.get("urls") or {}).get("LAZ") or ""
            if not download_url or download_url in seen_urls:
                continue
            project = project_from_url(download_url)
            if wanted_projects and project.lower() not in wanted_projects:
                continue
            min_x, min_y, max_x, max_y = item_bbox(item)
            seen_urls.add(download_url)
            tiles.append(
                {
                    "title": item.get("title", ""),
                    "project": project,
                    "url": download_url,
                    "source_id": item.get("sourceId", ""),
                    "date_created": item.get("dateCreated", ""),
                    "last_updated": item.get("lastUpdated", ""),
                    "size_bytes": size_from_item(item),
                    "min_x": min_x,
                    "min_y": min_y,
                    "max_x": max_x,
                    "max_y": max_y,
                }
            )
        offset += args.max_results
    return tiles


def point_in_tile(lon: float, lat: float, tile: dict) -> bool:
    if None in (tile["min_x"], tile["min_y"], tile["max_x"], tile["max_y"]):
        return False
    return float(tile["min_x"]) <= lon <= float(tile["max_x"]) and float(tile["min_y"]) <= lat <= float(tile["max_y"])


def sample_cells(cells: list[dict], max_points: int) -> list[dict]:
    if max_points <= 0 or len(cells) <= max_points:
        return cells
    if max_points == 1:
        return [cells[len(cells) // 2]]
    step = (len(cells) - 1) / (max_points - 1)
    return [cells[int(round(i * step))] for i in range(max_points)]


def coverage_fraction(points: list[tuple[float, float]], tiles: list[dict]) -> float:
    if not points:
        return 0.0
    covered = sum(1 for lon, lat in points if any(point_in_tile(lon, lat, tile) for tile in tiles))
    return covered / len(points)


def main() -> int:
    args = parse_args()
    city_filter = normalize_token(args.city_token)
    context = ssl_context(args.no_verify_tls)
    records = []
    details = []

    for city_token, table in city_cell_tables(args.cell_map_dir):
        if city_filter and city_filter not in normalize_token(city_token):
            continue
        cells = read_unique_cells(table)
        if not cells:
            continue
        bbox = city_bbox(cells, args)
        bbox_text = ",".join(f"{value:.8f}" for value in bbox)
        print(f"{city_token}: {len(cells):,} cell(s), bbox={bbox_text}")
        if args.dry_run:
            tiles = []
        else:
            tiles = catalog_tiles(bbox, args, context)
            time.sleep(args.sleep)

        sampled_cells = sample_cells(cells, args.max_coverage_points)
        point_sample = [cell_center_lonlat(cell, args) for cell in sampled_cells] if tiles else []
        project_counts = Counter(tile["project"] or "unknown" for tile in tiles)
        coverage = coverage_fraction(point_sample, tiles) if tiles else 0.0
        top_projects = ";".join(f"{project}:{count}" for project, count in project_counts.most_common(5))
        record = {
            "city_token": city_token,
            "cell_count": len(cells),
            "bbox": bbox_text,
            "tile_count": len(tiles),
            "project_count": len(project_counts),
            "top_projects": top_projects,
            "sampled_cell_centers": len(point_sample),
            "catalog_bbox_center_coverage_fraction": round(coverage, 6),
            "has_lidar_catalog_hits": int(bool(tiles)),
        }
        records.append(record)
        details.append({**record, "tiles": tiles})
        print(
            f"  tiles={len(tiles):,}, projects={len(project_counts):,}, "
            f"sample-center coverage={coverage:.1%}, top={top_projects or 'none'}"
        )

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", encoding="utf-8", newline="") as stream:
        fieldnames = [
            "city_token",
            "cell_count",
            "bbox",
            "tile_count",
            "project_count",
            "top_projects",
            "sampled_cell_centers",
            "catalog_bbox_center_coverage_fraction",
            "has_lidar_catalog_hits",
        ]
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
    args.output_json.write_text(json.dumps(details, indent=2), encoding="utf-8")
    print(f"Wrote {args.output_csv}")
    print(f"Wrote {args.output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
