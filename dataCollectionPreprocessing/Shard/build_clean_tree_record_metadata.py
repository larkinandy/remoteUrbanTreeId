#!/usr/bin/env python3
"""Build clean tree-record metadata keyed by tree_id.

This script creates the standalone tree-record files used by the rebuilt
tree-centered pipeline. It reads raw inventory CSVs, calculates exact lon/lat
coordinate duplication rates by city, removes every record whose exact
coordinate pair appears more than once, and writes one file per city keyed by
tree_id. Sentinel/GEE/PRISM cell IDs are intentionally not assigned here;
those belong after matching a tree record to a detected crown.

Additional provenance/taxon columns are retained for downstream QA/modeling,
but tree_id is the primary record key.
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_INVENTORY_ROOT = Path(r"C:\Users\larki\Desktop\PollenSense\training\McCoy")
DEFAULT_OUTPUT_ROOT = Path(r"H:\TreeCenteredModelInputs\tree_record_metadata_clean")

TARGET_GENUS_LABELS = {
    "quercus": "quercus",
    "acer": "acer",
    "betula": "betula",
    "ulmus": "ulmus",
    "fraxinus": "fraxinus",
    "populus": "populus",
    "platanus": "platanus",
    "lagerstroemia": "lagerstroemia",
    "liquidambar": "liquidambar",
    "pistacia": "pistacia",
}
PINACEAE_GENERA = {"abies", "cedrus", "larix", "picea", "pinus", "pseudotsuga", "tsuga"}
CONIFER_GENERA = PINACEAE_GENERA | {
    "calocedrus", "chamaecyparis", "cryptomeria", "cupressus", "juniperus",
    "metasequoia", "platycladus", "sequoia", "sequoiadendron", "taxodium", "thuja",
}
PALM_GENERA = {
    "arecastrum", "bismarckia", "brahea", "butia", "chamaedorea", "dypsis",
    "howea", "livistona", "phoenix", "sabal", "syagrus", "trachycarpus", "washingtonia",
}
ROSACEAE_ORNAMENTAL_GENERA = {"amelanchier", "crataegus", "malus", "prunus", "pyrus"}
GENUS_ALIASES = {"gingko": "ginkgo"}
LOW_CONFIDENCE_EXCLUDED_GENERA = {
    "pyracantha", "rhododendron", "conus", "solanum", "background", "n", "na", "nan",
    "no", "no_info", "none", "not", "null", "tree", "unk", "unassigned", "undefined",
    "unidentified", "unknown",
}
UNKNOWN_TOKENS = {
    "", "background", "na", "n/a", "nan", "none", "null", "not", "not a tree",
    "not tree", "unassigned", "undefined", "unknown", "unk", "tree", "unidentified", "no_info",
}


def extract_genus(scientific_name: object) -> str:
    if scientific_name is None:
        return ""
    text = str(scientific_name).strip()
    if text.lower() in UNKNOWN_TOKENS:
        return ""
    match = re.search(r"[A-Za-z]+", text)
    genus = match.group(0).lower() if match else ""
    return GENUS_ALIASES.get(genus, genus)


def map_taxon(scientific_name: object, common_name: object) -> str | None:
    """Map an inventory taxon to the current broad model label.

    ``common_name`` is retained in the signature for source-data compatibility;
    the current mapping deliberately requires a usable scientific-name genus.
    """
    del common_name
    genus = extract_genus(scientific_name)
    if not genus or genus in LOW_CONFIDENCE_EXCLUDED_GENERA:
        return None
    if genus in TARGET_GENUS_LABELS:
        return TARGET_GENUS_LABELS[genus]
    if genus in ROSACEAE_ORNAMENTAL_GENERA:
        return "rosaceae_ornamental"
    if genus in PALM_GENERA:
        return "other_palm"
    if genus in PINACEAE_GENERA:
        return "pinaceae"
    if genus in CONIFER_GENERA:
        return "other_conifer"
    return "other_broadleaf"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--inventory-root", type=Path, default=DEFAULT_INVENTORY_ROOT)
    parser.add_argument("--inventory-pattern", default="*_Final_*.csv")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--city-token", action="append", default=None, help="Optional city token(s). Repeatable.")
    parser.add_argument("--exclude-city-token", action="append", default=[], help="City token(s) to skip. Repeatable.")
    parser.add_argument("--longitude-column", default="longitude_coordinate")
    parser.add_argument("--latitude-column", default="latitude_coordinate")
    parser.add_argument("--scientific-column", default="scientific_name")
    parser.add_argument("--common-column", default="common_name")
    parser.add_argument("--min-unique-coordinate-percent", type=float, default=50.0)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def city_token_from_path(path: Path) -> str:
    stem = re.sub(r"_Final_.*$", "", path.stem, flags=re.IGNORECASE)
    return re.sub(r"[^a-z0-9]+", "", stem.lower())


def valid_coordinate_mask(frame: pd.DataFrame, lon_col: str, lat_col: str) -> pd.Series:
    lon = pd.to_numeric(frame[lon_col], errors="coerce")
    lat = pd.to_numeric(frame[lat_col], errors="coerce")
    return lon.between(-180, 180) & lat.between(-90, 90)


def require_columns(frame: pd.DataFrame, columns: list[str], path: Path) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise RuntimeError(f"{path} is missing required column(s): {missing}")


def build_city(path: Path, args: argparse.Namespace) -> dict[str, object]:
    city_token = city_token_from_path(path)
    output_dir = args.output_root / city_token
    output_csv = output_dir / f"{city_token}_tree_record_metadata_clean.csv"
    if output_csv.exists() and not args.force:
        existing = pd.read_csv(output_csv, usecols=["tree_id"], low_memory=False)
        return {
            "city_token": city_token,
            "source_file": path.name,
            "status": "exists",
            "output_csv": str(output_csv),
            "n_singleton_coordinate_tree_records_written": int(len(existing)),
        }

    header = pd.read_csv(path, nrows=0).columns.tolist()
    required = [args.longitude_column, args.latitude_column, args.scientific_column, args.common_column]
    require_columns(pd.DataFrame(columns=header), required, path)
    optional = [column for column in ["city", "state"] if column in header]
    usecols = list(dict.fromkeys(required + optional))
    frame = pd.read_csv(path, usecols=usecols, low_memory=False)
    n_inventory_rows = int(len(frame))
    valid = valid_coordinate_mask(frame, args.longitude_column, args.latitude_column)
    with_coords = frame.loc[valid].copy()
    with_coords["source_row"] = np.flatnonzero(valid.to_numpy()) + 2
    with_coords["tree_lon"] = pd.to_numeric(with_coords[args.longitude_column], errors="coerce")
    with_coords["tree_lat"] = pd.to_numeric(with_coords[args.latitude_column], errors="coerce")

    coord_counts = with_coords.groupby(["tree_lon", "tree_lat"], sort=False).size().rename("exact_coord_count")
    with_counts = with_coords.merge(coord_counts.reset_index(), on=["tree_lon", "tree_lat"], how="left", validate="many_to_one")
    singleton = with_counts.loc[with_counts["exact_coord_count"].eq(1)].copy()
    singleton = singleton.reset_index(drop=True)

    n_valid_coords = int(len(with_coords))
    n_distinct_coordinate_pairs = int(len(coord_counts))
    n_singleton = int(len(singleton))
    n_duplicate_coord_records = int(n_valid_coords - n_singleton)
    pct_distinct_locations = n_distinct_coordinate_pairs / n_valid_coords * 100.0 if n_valid_coords else 0.0
    pct_singleton_records = n_singleton / n_valid_coords * 100.0 if n_valid_coords else 0.0
    fails_unique_coordinate_threshold = pct_singleton_records < float(args.min_unique_coordinate_percent)

    if singleton.empty:
        output_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            columns=[
                "tree_id",
                "tree_lat",
                "tree_lon",
            ]
        ).to_csv(output_csv, index=False)
    else:
        output = pd.DataFrame(
            {
                "tree_id": np.arange(1, len(singleton) + 1, dtype=np.int64),
                "source_file": path.name,
                "source_row": singleton["source_row"].to_numpy(dtype=np.int64),
                "tree_lat": singleton["tree_lat"].to_numpy(float),
                "tree_lon": singleton["tree_lon"].to_numpy(float),
                "scientific_name": singleton[args.scientific_column].astype(str).replace("nan", "").to_numpy(),
                "common_name": singleton[args.common_column].astype(str).replace("nan", "").to_numpy(),
                "taxon_label": [
                    map_taxon(scientific, common)
                    for scientific, common in zip(singleton[args.scientific_column], singleton[args.common_column])
                ],
                "exact_coord_count": singleton["exact_coord_count"].to_numpy(dtype=np.int64),
            }
        )
        if "city" in singleton.columns:
            output["inventory_city"] = singleton["city"].to_numpy()
        if "state" in singleton.columns:
            output["inventory_state"] = singleton["state"].to_numpy()
        output_dir.mkdir(parents=True, exist_ok=True)
        output.to_csv(output_csv, index=False)

    return {
        "city_token": city_token,
        "source_file": path.name,
        "status": "completed",
        "output_csv": str(output_csv),
        "n_inventory_rows": n_inventory_rows,
        "n_valid_coordinate_rows": n_valid_coords,
        "n_distinct_coordinate_pairs": n_distinct_coordinate_pairs,
        "n_singleton_coordinate_tree_records": n_singleton,
        "n_duplicate_coordinate_tree_records_removed": n_duplicate_coord_records,
        "percent_distinct_coordinate_locations": pct_distinct_locations,
        "percent_singleton_coordinate_tree_records": pct_singleton_records,
        "min_unique_coordinate_percent": float(args.min_unique_coordinate_percent),
        "fails_unique_coordinate_threshold": bool(fails_unique_coordinate_threshold),
    }


def main() -> int:
    args = parse_args()
    files = sorted(args.inventory_root.glob(args.inventory_pattern))
    if not files:
        raise SystemExit(f"No inventory files matched {args.inventory_pattern!r} under {args.inventory_root}")
    selected = {token.strip().lower() for token in args.city_token or [] if token.strip()}
    excluded = {token.strip().lower() for token in args.exclude_city_token if token.strip()}
    if selected:
        files = [path for path in files if city_token_from_path(path) in selected]
    files = [path for path in files if city_token_from_path(path) not in excluded]
    if not files:
        raise SystemExit("No inventory files selected.")

    args.output_root.mkdir(parents=True, exist_ok=True)
    rows = []
    for index, path in enumerate(files, start=1):
        city_token = city_token_from_path(path)
        print(f"[{index:,}/{len(files):,}] {city_token}: {path.name}", flush=True)
        try:
            row = build_city(path, args)
            rows.append(row)
            if row.get("status") == "completed":
                print(
                    f"  singleton_unique={int(row['n_singleton_coordinate_tree_records']):,}/"
                    f"{int(row['n_valid_coordinate_rows']):,} "
                    f"({float(row['percent_singleton_coordinate_tree_records']):.2f}%); "
                    f"fails_lt_{args.min_unique_coordinate_percent:g}%={row['fails_unique_coordinate_threshold']}",
                    flush=True,
                )
            else:
                print(f"  {row.get('status')}: {row.get('output_csv')}", flush=True)
        except Exception as error:
            print(f"  FAILED: {error}", flush=True)
            rows.append({"city_token": city_token, "source_file": path.name, "status": "failed", "error": str(error)})

    summary = pd.DataFrame(rows).sort_values("city_token", kind="stable")
    summary_path = args.output_root / "tree_record_metadata_clean_summary.csv"
    summary.to_csv(summary_path, index=False, quoting=csv.QUOTE_MINIMAL)
    print("\nSummary status counts:", flush=True)
    print(summary["status"].value_counts(dropna=False).to_string(), flush=True)
    if "fails_unique_coordinate_threshold" in summary.columns:
        failed_threshold = summary["fails_unique_coordinate_threshold"].fillna(False).astype(bool)
        print(f"Cities below {args.min_unique_coordinate_percent:g}% singleton coordinate records: {int(failed_threshold.sum()):,}", flush=True)
    print(f"Wrote summary: {summary_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
