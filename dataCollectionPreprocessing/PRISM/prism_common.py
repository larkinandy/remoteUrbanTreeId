"""Shared parsing and Sentinel-cell helpers for the PRISM workflow."""

from __future__ import annotations

import csv
from datetime import date, datetime, timedelta
from pathlib import Path
import re
from typing import Iterable

from pyproj import Transformer


CITY_ALIASES = {
    "abq": "albuquerque",
    "ana": "anaheim",
    "arl": "arlington",
    "atl": "atlanta",
    "bal": "baltimore",
    "buf": "buffalo",
    "cos": "coloradosprings",
    "csp": "coloradosprings",
    "dca": "washingtondc",
    "dc": "washingtondc",
    "den": "denver",
    "ggv": "gardengrove",
    "hnb": "huntingtonbeach",
    "la": "losangeles",
    "lax": "losangeles",
    "nyc": "newyork",
    "sf": "sanfrancisco",
}


def normalize_city(value: object) -> str:
    """Return the canonical compact token used for study-city filenames."""
    token = re.sub(r"[^a-z0-9]+", "", str(value or "").lower())
    token = token.replace("alberquerque", "albuquerque")
    return CITY_ALIASES.get(token, token)


def parse_iso_date(value: str, label: str = "date") -> date:
    try:
        return datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError as exc:
        raise ValueError(f"{label} must use YYYY-MM-DD; got {value!r}") from exc


def parse_flexible_date(value: str) -> date:
    for fmt in ("%Y-%m-%d", "%Y%m%d", "%Y-%m", "%Y%m", "%Y"):
        try:
            return datetime.strptime(value, fmt).date()
        except ValueError:
            continue
    raise ValueError(f"Could not parse date {value!r}")


def inclusive_date_range(start: date, end: date) -> list[date]:
    if end < start:
        raise ValueError("--end-date must be on or after --start-date")
    return [start + timedelta(days=offset) for offset in range((end - start).days + 1)]


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
            if end < start:
                raise ValueError(f"Descending range is not supported: {part!r}")
            values.update(range(start, end + 1))
        else:
            values.add(int(part))
    bad = sorted(value for value in values if value < min_value or value > max_value)
    if bad:
        raise ValueError(f"Values out of range {min_value}-{max_value}: {bad}")
    return sorted(values)


def parse_epsg(cell_id: str) -> int:
    match = re.search(r"epsg(\d+)", cell_id.lower())
    if not match:
        raise ValueError(f"Could not parse EPSG from cell_id={cell_id!r}")
    return int(match.group(1))


def city_cell_tables(cell_map_dir: Path) -> Iterable[tuple[str, Path]]:
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


def cell_center_lonlat(cell: dict, args: object) -> tuple[float, float]:
    x = args.sentinel_origin_x + (cell["cell_col"] + 0.5) * args.sentinel_cell_size
    y = args.sentinel_origin_y + (cell["cell_row"] + 0.5) * args.sentinel_cell_size
    transformer = Transformer.from_crs(int(cell["cell_epsg"]), 4326, always_xy=True)
    lon, lat = transformer.transform(x, y)
    return float(lon), float(lat)
