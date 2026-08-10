#!/usr/bin/env python3
"""Download PRISM climate grids for study Sentinel-2 cells.

This script is designed for the McCoy/remoteUrbanTreeId Sentinel-cell workflow.
It can:

  1. read unique Sentinel-2 cells where trees are located,
  2. download selected PRISM products/variables/dates, and
  3. optionally sample the downloaded raster grids at those Sentinel cell
     centers when rasterio is available.

The public PRISM URL conventions are exposed as templates so the script can be
adjusted if PRISM changes a product path. By default, downloads use PRISM's
direct zip paths, for example:

    https://ftp.prism.oregonstate.edu/normals/us/4km/ppt/monthly/prism_ppt_us_25m_202001_avg_30y.zip
    https://ftp.prism.oregonstate.edu/time_series/us/an/4km/ppt/daily/2020/prism_ppt_us_25m_20200101.zip

where ``date`` is typically ``01``-``12`` for monthly normals. Use
``--url-template`` to override these conventions.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import ssl
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Iterable


HERE = Path(__file__).resolve().parent
LIDAR_HELPER_DIR = HERE.parent / "LiDAR"
DEFAULT_CELL_MAP_DIR = HERE.parent / "Sentinel2" / "mccoy_sentinel_10m_cells_utm"
DEFAULT_OUTPUT_DIR = Path(r"E:\PRISM\sentinel_cells")
DEFAULT_URL_TEMPLATES = {
    "daily": (
        "https://ftp.prism.oregonstate.edu/time_series/{region}/{time_series_kind}/{resolution}/{variable}/daily/{year}/"
        "prism_{variable}_{region}_{grid_token}_{date}.zip"
    ),
    "monthly": (
        "https://ftp.prism.oregonstate.edu/time_series/{region}/{time_series_kind}/{resolution}/{variable}/monthly/{year}/"
        "prism_{variable}_{region}_{grid_token}_{date}.zip"
    ),
    "annual": (
        "https://ftp.prism.oregonstate.edu/time_series/{region}/{time_series_kind}/{resolution}/{variable}/annual/"
        "prism_{variable}_{region}_{grid_token}_{date}.zip"
    ),
    "normal-monthly": (
        "https://ftp.prism.oregonstate.edu/normals/{region}/{resolution}/{variable}/monthly/"
        "prism_{variable}_{region}_{grid_token}_2020{date}_avg_30y.zip"
    ),
    "normal-annual": (
        "https://ftp.prism.oregonstate.edu/normals/{region}/{resolution}/{variable}/monthly/"
        "prism_{variable}_{region}_{grid_token}_2020_avg_30y.zip"
    ),
}
DEFAULT_VARIABLES = ("ppt", "tmin", "tmax", "tmean", "vpdmin", "vpdmax")
PRODUCT_DATE_FORMATS = {
    "daily": "%Y%m%d",
    "monthly": "%Y%m",
    "annual": "%Y",
    "normal-monthly": "%m",
    "normal-annual": "annual",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    city_group = parser.add_mutually_exclusive_group(required=True)
    city_group.add_argument("--city-token", action="append", default=[], help="Study city token/name. Repeatable.")
    city_group.add_argument("--all-cities", action="store_true", help="Use every city in --cell-map-dir.")
    parser.add_argument("--exclude-city-token", action="append", default=[], help="City token/name to skip. Repeatable.")

    parser.add_argument("--cell-map-dir", type=Path, default=DEFAULT_CELL_MAP_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--sentinel-cell-size", type=float, default=10.0)
    parser.add_argument("--sentinel-origin-x", type=float, default=0.0)
    parser.add_argument("--sentinel-origin-y", type=float, default=0.0)
    parser.add_argument(
        "--product",
        choices=sorted(PRODUCT_DATE_FORMATS),
        default="normal-monthly",
        help="PRISM product/date family.",
    )
    parser.add_argument(
        "--variables",
        default=",".join(DEFAULT_VARIABLES),
        help="Comma-separated PRISM variables, e.g. ppt,tmin,tmax,tmean,vpdmax.",
    )
    parser.add_argument("--start-date", help="Start date for daily/monthly/annual products.")
    parser.add_argument("--end-date", help="End date for daily/monthly/annual products.")
    parser.add_argument("--years", help="Comma-separated years or ranges, e.g. 2018,2020-2022.")
    parser.add_argument("--months", default="1-12", help="Comma-separated months or ranges for monthly normals/time series.")
    parser.add_argument(
        "--dates",
        help="Explicit comma-separated product dates. Examples: 20200101,202001,2020,01,annual.",
    )
    parser.add_argument(
        "--url-template",
        help=(
            "Override URL template. Supports {variable}, {date}, {year}, {month}, {product}, "
            "{region}, {resolution}, {grid_token}, {time_series_kind}, {stability}."
        ),
    )
    parser.add_argument("--region", default="us", help="PRISM region folder, usually us or ak.")
    parser.add_argument("--resolution", default="4km", help="PRISM resolution folder, e.g. 4km or 800m where available.")
    parser.add_argument("--grid-token", default="25m", help="Filename grid token used by PRISM, e.g. 25m for 4 km.")
    parser.add_argument(
        "--time-series-kind",
        choices=("an", "lt"),
        default="an",
        help="PRISM time-series type: an=all networks, lt=long-term temporal consistency.",
    )
    parser.add_argument(
        "--stability",
        default="stable",
        help="PRISM time-series file status token used by default daily/monthly/annual templates.",
    )
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--sleep", type=float, default=0.2)
    parser.add_argument("--chunk-mb", type=float, default=8.0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-verify-tls", action="store_true")
    parser.add_argument("--query-only", action="store_true", help="Write expected download manifest without downloading.")
    parser.add_argument("--dry-run", action="store_true", help="Print planned work without writing or downloading.")
    parser.add_argument("--extract-values", action="store_true", help="Sample downloaded grids at Sentinel cell centers.")
    parser.add_argument("--keep-extracted-rasters", action="store_true")
    parser.add_argument("--max-cells-per-city", type=int, default=0, help="Debug/sample cap. 0 means all cells.")
    parser.add_argument("--manifest-name", default="prism_download_manifest")
    parser.add_argument("--cell-output-name", default="prism_sentinel_cell_values.csv")
    return parser.parse_args()


def normalize_token(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value or "").lower().replace("alberquerque", "albuquerque"))


def ssl_context(no_verify_tls: bool) -> ssl.SSLContext | None:
    if not no_verify_tls:
        return None
    context = ssl.create_default_context()
    context.check_hostname = False
    context.verify_mode = ssl.CERT_NONE
    return context


def load_city_helpers():
    sys.path.insert(0, str(LIDAR_HELPER_DIR))
    try:
        from identify_tnm_lidar_city_coverage import cell_center_lonlat, city_cell_tables, read_unique_cells
    finally:
        try:
            sys.path.remove(str(LIDAR_HELPER_DIR))
        except ValueError:
            pass
    return cell_center_lonlat, city_cell_tables, read_unique_cells


def iter_city_cells(args: argparse.Namespace) -> Iterable[tuple[str, list[dict]]]:
    cell_center_lonlat, city_cell_tables, read_unique_cells = load_city_helpers()
    wanted = {normalize_token(token) for token in args.city_token}
    excluded = {normalize_token(token) for token in args.exclude_city_token}
    found = set()
    for city_name, path in city_cell_tables(args.cell_map_dir):
        city = normalize_token(city_name)
        if wanted and city not in wanted:
            continue
        if city in excluded:
            continue
        cells = read_unique_cells(path)
        if args.max_cells_per_city:
            cells = cells[: args.max_cells_per_city]
        for cell in cells:
            lon, lat = cell_center_lonlat(cell, args)
            cell["lon"] = lon
            cell["lat"] = lat
        found.add(city)
        yield city, cells
    missing = sorted(wanted - found)
    if missing:
        raise FileNotFoundError(f"No Sentinel cell map found for city token(s): {', '.join(missing)}")


def parse_number_list(text: str, min_value: int, max_value: int) -> list[int]:
    values: set[int] = set()
    for part in str(text or "").split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_text, end_text = part.split("-", 1)
            start = int(start_text)
            end = int(end_text)
            values.update(range(start, end + 1))
        else:
            values.add(int(part))
    bad = sorted(value for value in values if value < min_value or value > max_value)
    if bad:
        raise ValueError(f"Values out of range {min_value}-{max_value}: {bad}")
    return sorted(values)


def parse_date(value: str) -> date:
    for fmt in ("%Y-%m-%d", "%Y%m%d", "%Y-%m", "%Y%m", "%Y"):
        try:
            parsed = datetime.strptime(value, fmt)
            return parsed.date()
        except ValueError:
            continue
    raise ValueError(f"Could not parse date {value!r}")


def month_iter(start: date, end: date) -> Iterable[date]:
    current = date(start.year, start.month, 1)
    final = date(end.year, end.month, 1)
    while current <= final:
        yield current
        year = current.year + (1 if current.month == 12 else 0)
        month = 1 if current.month == 12 else current.month + 1
        current = date(year, month, 1)


def year_iter(start: date, end: date) -> Iterable[date]:
    for year in range(start.year, end.year + 1):
        yield date(year, 1, 1)


def build_dates(args: argparse.Namespace) -> list[str]:
    if args.dates:
        return [part.strip() for part in args.dates.split(",") if part.strip()]

    if args.product == "normal-annual":
        return ["annual"]
    if args.product == "normal-monthly":
        return [f"{month:02d}" for month in parse_number_list(args.months, 1, 12)]

    if args.years:
        years = parse_number_list(args.years, 1, 9999)
        if args.product == "monthly":
            months = parse_number_list(args.months, 1, 12)
            return [f"{year}{month:02d}" for year in years for month in months]
        return [str(year) for year in years]

    if not args.start_date or not args.end_date:
        raise ValueError("--start-date and --end-date are required unless --dates, --years, or normals are used.")

    start = parse_date(args.start_date)
    end = parse_date(args.end_date)
    if start > end:
        raise ValueError("--start-date must be <= --end-date")

    if args.product == "daily":
        values = []
        current = start
        while current <= end:
            values.append(current.strftime("%Y%m%d"))
            current += timedelta(days=1)
        return values
    if args.product == "monthly":
        return [value.strftime("%Y%m") for value in month_iter(start, end)]
    if args.product == "annual":
        return [value.strftime("%Y") for value in year_iter(start, end)]
    raise ValueError(f"Unsupported product: {args.product}")


def product_url(variable: str, product_date: str, args: argparse.Namespace) -> str:
    template = args.url_template
    if not template:
        template = DEFAULT_URL_TEMPLATES[args.product]
    year = product_date[:4] if len(product_date) >= 4 and product_date[:4].isdigit() else ""
    month = ""
    if len(product_date) == 2 and product_date.isdigit():
        month = product_date
    elif len(product_date) >= 6 and product_date[4:6].isdigit():
        month = product_date[4:6]
    return template.format(
        variable=variable,
        date=product_date,
        year=year,
        month=month,
        product=args.product,
        stability=args.stability,
        region=args.region,
        resolution=args.resolution,
        grid_token=args.grid_token,
        time_series_kind=args.time_series_kind,
    )


def request_filename(response: urllib.request.addinfourl, fallback: str) -> str:
    header = response.headers.get("Content-Disposition", "")
    match = re.search(r'filename="?([^";]+)"?', header)
    if match:
        return Path(match.group(1)).name
    url_name = Path(urllib.parse.unquote(urllib.parse.urlparse(response.url).path)).name
    if url_name and "." in url_name:
        return url_name
    return fallback


def fallback_filename(product: str, variable: str, product_date: str, args: argparse.Namespace) -> str:
    if product == "normal-monthly":
        return f"prism_{variable}_{args.region}_{args.grid_token}_2020{product_date}_avg_30y.zip"
    if product == "normal-annual":
        return f"prism_{variable}_{args.region}_{args.grid_token}_2020_avg_30y.zip"
    return f"prism_{variable}_{args.region}_{args.grid_token}_{product_date}.zip"


def is_valid_zip(path: Path) -> bool:
    if not path.exists() or path.stat().st_size <= 0:
        return False
    try:
        return zipfile.is_zipfile(path)
    except OSError:
        return False


def response_looks_like_zip(response: urllib.request.addinfourl) -> bool:
    content_type = (response.headers.get("Content-Type") or "").lower()
    if "zip" in content_type or "octet-stream" in content_type:
        return True
    # PRISM's server may omit a helpful content type, so the final zip-file
    # validation remains the source of truth after the download completes.
    return not any(token in content_type for token in ("html", "json", "text/plain"))


def download_one(row: dict, args: argparse.Namespace, context: ssl.SSLContext | None) -> dict:
    raw_dir = args.output_dir / "raw" / str(row["product"]) / str(row["variable"])
    raw_dir.mkdir(parents=True, exist_ok=True)
    destination = raw_dir / str(row["filename"])
    row["destination"] = str(destination)
    if destination.exists() and destination.stat().st_size > 0 and not args.overwrite:
        if is_valid_zip(destination):
            row["status"] = "skipped_existing"
            row["downloaded_bytes"] = destination.stat().st_size
            return row
        row["last_error"] = "existing file is not a valid zip; redownloading"

    partial = destination.with_suffix(destination.suffix + ".partial")
    chunk_size = max(1, int(args.chunk_mb * 1024 * 1024))
    for attempt in range(1, args.retries + 1):
        try:
            request = urllib.request.Request(row["url"], headers={"User-Agent": "remoteUrbanTreeId PRISM downloader"})
            with urllib.request.urlopen(request, timeout=args.timeout, context=context) as response:
                if not response_looks_like_zip(response):
                    raise RuntimeError(
                        f"unexpected content type {response.headers.get('Content-Type')!r} from {response.url}"
                    )
                resolved_name = request_filename(response, row["filename"])
                if resolved_name != row["filename"]:
                    destination = raw_dir / resolved_name
                    partial = destination.with_suffix(destination.suffix + ".partial")
                    row["filename"] = resolved_name
                    row["destination"] = str(destination)
                    if destination.exists() and destination.stat().st_size > 0 and not args.overwrite:
                        if is_valid_zip(destination):
                            row["status"] = "skipped_existing"
                            row["downloaded_bytes"] = destination.stat().st_size
                            return row
                        row["last_error"] = "existing file is not a valid zip; redownloading"
                with partial.open("wb") as stream:
                    while True:
                        chunk = response.read(chunk_size)
                        if not chunk:
                            break
                        stream.write(chunk)
            if partial.stat().st_size <= 0:
                raise RuntimeError("download produced an empty file")
            if not is_valid_zip(partial):
                preview = partial.read_bytes()[:200]
                try:
                    preview_text = preview.decode("utf-8", errors="replace").replace("\r", " ").replace("\n", " ")
                except OSError:
                    preview_text = ""
                raise RuntimeError(f"download is not a valid zip; first bytes={preview_text!r}")
            partial.replace(destination)
            row["status"] = "downloaded"
            row["downloaded_bytes"] = destination.stat().st_size
            row["last_error"] = ""
            return row
        except (OSError, TimeoutError, RuntimeError, urllib.error.URLError, urllib.error.HTTPError) as exc:
            row["status"] = "failed"
            row["last_error"] = f"attempt {attempt}/{args.retries}: {exc}"
            print(f"  failed {row['variable']} {row['date']}: {row['last_error']}")
            time.sleep(args.sleep * attempt)
    return row


def write_manifest(rows: list[dict], args: argparse.Namespace) -> None:
    if args.dry_run:
        return
    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / f"{args.manifest_name}.csv"
    json_path = args.output_dir / f"{args.manifest_name}.json"
    fields = [
        "product",
        "variable",
        "date",
        "url",
        "filename",
        "destination",
        "status",
        "downloaded_bytes",
        "last_error",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    with json_path.open("w", encoding="utf-8") as stream:
        json.dump(rows, stream, indent=2)
    print(f"Wrote manifest: {csv_path}")


def extract_bil_from_zip(zip_path: Path, extract_root: Path) -> Path:
    target_dir = extract_root / zip_path.stem
    bils = sorted(target_dir.glob("*.bil"))
    if bils:
        return bils[0]
    target_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(target_dir)
    bils = sorted(target_dir.glob("*.bil"))
    if not bils:
        raise FileNotFoundError(f"No .bil raster found inside {zip_path}")
    return bils[0]


def sample_raster_to_cells(row: dict, city_cells: dict[str, list[dict]], args: argparse.Namespace, writer: csv.DictWriter) -> None:
    try:
        import rasterio
        from rasterio.warp import transform
    except ImportError as exc:
        raise RuntimeError("--extract-values requires rasterio in the active Python environment") from exc

    destination = Path(row["destination"])
    if not destination.exists():
        return
    extract_root = args.output_dir / "extracted" / str(row["product"]) / str(row["variable"])
    bil_path = extract_bil_from_zip(destination, extract_root)
    with rasterio.open(bil_path) as dataset:
        for city, cells in city_cells.items():
            xs = [float(cell["lon"]) for cell in cells]
            ys = [float(cell["lat"]) for cell in cells]
            if dataset.crs and str(dataset.crs).upper() not in {"EPSG:4326", "EPSG:4269"}:
                xs, ys = transform("EPSG:4326", dataset.crs, xs, ys)
            values = list(dataset.sample(list(zip(xs, ys))))
            nodata = dataset.nodata
            for cell, sample in zip(cells, values):
                value = float(sample[0])
                if nodata is not None and value == nodata:
                    value_text = ""
                else:
                    value_text = f"{value:.6g}"
                writer.writerow(
                    {
                        "city": city,
                        "reduced_id": cell["reduced_id"],
                        "cell_id": cell["cell_id"],
                        "cell_col": cell["cell_col"],
                        "cell_row": cell["cell_row"],
                        "lon": f"{float(cell['lon']):.8f}",
                        "lat": f"{float(cell['lat']):.8f}",
                        "product": row["product"],
                        "variable": row["variable"],
                        "date": row["date"],
                        "value": value_text,
                        "source_file": str(destination),
                    }
                )
    if not args.keep_extracted_rasters:
        # Leave the directory in place if cleanup fails; the raw zip remains the source of truth.
        for path in sorted(bil_path.parent.glob("*")):
            try:
                path.unlink()
            except OSError:
                pass
        try:
            bil_path.parent.rmdir()
        except OSError:
            pass


def main() -> int:
    args = parse_args()
    variables = [part.strip() for part in args.variables.split(",") if part.strip()]
    dates = build_dates(args)
    context = ssl_context(args.no_verify_tls)

    city_cells = dict(iter_city_cells(args))
    total_cells = sum(len(cells) for cells in city_cells.values())
    print(
        f"PRISM target cells: cities={len(city_cells):,}; cells={total_cells:,}; "
        f"product={args.product}; variables={len(variables):,}; dates={len(dates):,}"
    )

    rows: list[dict] = []
    for variable in variables:
        for product_date in dates:
            url = product_url(variable, product_date, args)
            rows.append(
                {
                    "product": args.product,
                    "variable": variable,
                    "date": product_date,
                    "url": url,
                    "filename": fallback_filename(args.product, variable, product_date, args),
                    "destination": "",
                    "status": "planned",
                    "downloaded_bytes": "",
                    "last_error": "",
                }
            )

    if args.dry_run:
        for row in rows[:20]:
            print(f"planned: {row['variable']} {row['date']} -> {row['url']}")
        if len(rows) > 20:
            print(f"... {len(rows) - 20:,} additional planned request(s)")
        return 0

    write_manifest(rows, args)
    if not args.query_only:
        for index, row in enumerate(rows, start=1):
            print(f"[{index:,}/{len(rows):,}] {row['variable']} {row['date']}")
            download_one(row, args, context)
            if index % 10 == 0 or index == len(rows):
                write_manifest(rows, args)

    if args.extract_values and not args.query_only:
        value_path = args.output_dir / args.cell_output_name
        fields = [
            "city",
            "reduced_id",
            "cell_id",
            "cell_col",
            "cell_row",
            "lon",
            "lat",
            "product",
            "variable",
            "date",
            "value",
            "source_file",
        ]
        with value_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for row in rows:
                if row["status"] not in {"downloaded", "skipped_existing"}:
                    continue
                print(f"extracting cell values: {row['variable']} {row['date']}")
                sample_raster_to_cells(row, city_cells, args, writer)
        print(f"Wrote cell values: {value_path}")

    status_counts: dict[str, int] = {}
    for row in rows:
        status = str(row.get("status") or "")
        status_counts[status] = status_counts.get(status, 0) + 1
    print("Status counts:", ", ".join(f"{key}={value}" for key, value in sorted(status_counts.items())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
