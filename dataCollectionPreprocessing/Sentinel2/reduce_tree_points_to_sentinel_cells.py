"""Reduce tree points to unique, aligned 10 m Sentinel-style grid cells.

The input may be a point shapefile, a longitude/latitude CSV, or a directory
containing spatial input files. Coordinates are transformed to a projected
meter-based CRS and snapped to a 10 m grid with a default origin of (0, 0).

Examples:
    python dataCollectionPreprocessing/Sentinel2/reduce_tree_points_to_sentinel_cells.py ^
        H:/NAIP/ANA/tree_predictions/tree_10m_from_cell.shp ^
        --target-crs EPSG:32617

    python dataCollectionPreprocessing/Sentinel2/reduce_tree_points_to_sentinel_cells.py ^
        C:/path/to/Albuquerque_Final_2022-06-18.csv ^
        --target-crs EPSG:32613

    python dataCollectionPreprocessing/Sentinel2/reduce_tree_points_to_sentinel_cells.py ^
        H:/NAIP/ANA/tree_predictions/partitioned_1km/tree_10m_from_cell ^
        --output-dir H:/NAIP/ANA/tree_predictions/sentinel_10m_cells ^
        --target-crs EPSG:32617

Required packages:
    pip install geopandas pyogrio pandas numpy
"""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd


CELL_SIZE_METERS = 10.0
DEFAULT_OUTPUT_DIR_NAME = "sentinel_10m_cells"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Assign tree points to aligned 10 m cells and create a unique "
            "cell-center dataset plus tree-to-cell join tables."
        )
    )
    parser.add_argument(
        "input",
        type=Path,
        help="A point shapefile, longitude/latitude CSV, or input directory.",
    )
    parser.add_argument(
        "--pattern",
        default="*.shp",
        help="File pattern when input is a directory (default: *.shp).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory. Defaults beside the input.",
    )
    parser.add_argument(
        "--tree-id-field",
        help=(
            "Existing unique tree-ID field. If omitted, the script tries "
            "uniqueID, tree_id, treeID, and id before generating IDs."
        ),
    )
    parser.add_argument(
        "--target-crs",
        help=(
            "Optional projected target CRS, such as EPSG:26917. Required if "
            "inputs do not share a projected meter-based CRS."
        ),
    )
    parser.add_argument("--origin-x", type=float, default=0.0)
    parser.add_argument("--origin-y", type=float, default=0.0)
    parser.add_argument("--cell-size", type=float, default=CELL_SIZE_METERS)
    parser.add_argument("--longitude-field", default="longitude_coordinate")
    parser.add_argument("--latitude-field", default="latitude_coordinate")
    return parser.parse_args()


def discover_input_files(input_path: Path, pattern: str) -> list[Path]:
    if input_path.is_file():
        if input_path.suffix.lower() not in {".shp", ".csv"}:
            raise ValueError(f"Input file must be a shapefile or CSV: {input_path}")
        return [input_path]

    if input_path.is_dir():
        input_files = sorted(input_path.glob(pattern))
        if not input_files:
            raise FileNotFoundError(
                f"No files matching {pattern!r} found in {input_path}"
            )
        invalid = [path for path in input_files if path.suffix.lower() not in {".shp", ".csv"}]
        if invalid:
            raise ValueError(f"Unsupported input file: {invalid[0]}")
        return input_files

    raise FileNotFoundError(f"Input does not exist: {input_path}")


def default_output_dir(input_path: Path) -> Path:
    parent = input_path.parent if input_path.is_file() else input_path
    return parent / DEFAULT_OUTPUT_DIR_NAME


def find_id_field(columns, requested: str | None) -> str | None:
    column_lookup = {str(column).lower(): str(column) for column in columns}
    if requested:
        match = column_lookup.get(requested.lower())
        if not match:
            raise ValueError(f"Tree ID field {requested!r} was not found")
        return match

    for candidate in ("uniqueid", "tree_id", "treeid", "id"):
        if candidate in column_lookup:
            return column_lookup[candidate]
    return None


def crs_token(crs) -> str:
    epsg = crs.to_epsg()
    if epsg:
        return f"epsg{epsg}"
    return re.sub(r"[^a-z0-9]+", "_", crs.to_string().lower()).strip("_")[:24]


def validate_projected_meter_crs(crs) -> None:
    if crs is None:
        raise ValueError("Input has no CRS; assign its CRS before running this script")
    if not crs.is_projected:
        raise ValueError(
            f"The working CRS must be projected, but got {crs}. "
            "Pass --target-crs with an appropriate meter-based CRS."
        )

    axis_units = {axis.unit_name.lower() for axis in crs.axis_info if axis.unit_name}
    if axis_units and not any("metre" in unit or "meter" in unit for unit in axis_units):
        raise ValueError(f"The working CRS must use meters, but got units {axis_units}")


def load_tree_points(
    input_files: list[Path],
    target_crs: str | None,
    requested_id_field: str | None,
    longitude_field: str,
    latitude_field: str,
) -> gpd.GeoDataFrame:
    frames = []
    working_crs = target_crs

    for input_file in input_files:
        print(f"Reading {input_file}")
        if input_file.suffix.lower() == ".csv":
            table = pd.read_csv(input_file, low_memory=False)
            missing = [
                field
                for field in (longitude_field, latitude_field)
                if field not in table.columns
            ]
            if missing:
                raise ValueError(f"{input_file} is missing coordinate fields: {missing}")

            longitude = pd.to_numeric(table[longitude_field], errors="coerce")
            latitude = pd.to_numeric(table[latitude_field], errors="coerce")
            valid = (
                longitude.between(-180, 180)
                & latitude.between(-90, 90)
                & longitude.notna()
                & latitude.notna()
            )
            if (~valid).any():
                print(f"  Skipping {int((~valid).sum()):,} invalid coordinates")
            table = table.loc[valid].copy()
            longitude = longitude.loc[valid]
            latitude = latitude.loc[valid]
            frame = gpd.GeoDataFrame(
                table,
                geometry=gpd.points_from_xy(longitude, latitude),
                crs="EPSG:4326",
            )
        else:
            frame = gpd.read_file(input_file, engine="pyogrio")

        if frame.crs is None:
            raise ValueError(f"Input has no CRS: {input_file}")

        if working_crs:
            frame = frame.to_crs(working_crs)
        elif frames and frame.crs != frames[0].crs:
            raise ValueError(
                "Input shapefiles have different CRSs. Pass --target-crs to "
                "transform them to one projected CRS."
            )

        invalid_geometry = frame.geometry.isna() | frame.geometry.is_empty
        if invalid_geometry.any():
            print(f"  Skipping {int(invalid_geometry.sum()):,} empty geometries")
            frame = frame.loc[~invalid_geometry].copy()

        non_points = frame.geometry.geom_type != "Point"
        if non_points.any():
            raise ValueError(
                f"{input_file} contains {int(non_points.sum()):,} non-point geometries"
            )

        id_field = find_id_field(frame.columns, requested_id_field)
        source_rows = frame.index.to_numpy()
        if id_field:
            tree_ids = frame[id_field].astype("string")
            missing_ids = tree_ids.isna() | (tree_ids.str.strip() == "")
            generated = pd.Series(
                [f"{input_file.stem}:{row}" for row in source_rows],
                index=frame.index,
                dtype="string",
            )
            tree_ids = tree_ids.mask(missing_ids, generated)
        else:
            tree_ids = pd.Series(
                [f"{input_file.stem}:{row}" for row in source_rows],
                index=frame.index,
                dtype="string",
            )

        loaded = gpd.GeoDataFrame(
            {
                "tree_uid": tree_ids.to_numpy(),
                "source_file": input_file.name,
                "source_row": source_rows,
            },
            geometry=frame.geometry.to_numpy(),
            crs=frame.crs,
        )
        frames.append(loaded)

    if not frames:
        raise ValueError("No valid point features were loaded")

    combined = gpd.GeoDataFrame(
        pd.concat(frames, ignore_index=True),
        geometry="geometry",
        crs=frames[0].crs,
    )
    if combined["tree_uid"].duplicated().any():
        duplicates = int(combined["tree_uid"].duplicated(keep=False).sum())
        raise ValueError(
            f"Found {duplicates:,} rows with duplicate tree IDs. Choose a "
            "globally unique --tree-id-field or let the script generate IDs."
        )
    return combined


def assign_cells(
    trees: gpd.GeoDataFrame,
    cell_size: float,
    origin_x: float,
    origin_y: float,
) -> tuple[gpd.GeoDataFrame, pd.DataFrame, pd.DataFrame]:
    validate_projected_meter_crs(trees.crs)
    if not math.isfinite(cell_size) or cell_size <= 0:
        raise ValueError("--cell-size must be a positive finite number")

    x = trees.geometry.x.to_numpy(dtype=np.float64)
    y = trees.geometry.y.to_numpy(dtype=np.float64)
    cell_col = np.floor((x - origin_x) / cell_size).astype(np.int64)
    cell_row = np.floor((y - origin_y) / cell_size).astype(np.int64)
    token = crs_token(trees.crs)

    assignment = trees.drop(columns="geometry").copy()
    assignment["cell_col"] = cell_col
    assignment["cell_row"] = cell_row
    assignment["cell_id"] = [
        f"{token}_c{col}_r{row}" for col, row in zip(cell_col, cell_row)
    ]

    cell_counts = (
        assignment.groupby(["cell_id", "cell_col", "cell_row"], sort=True)
        .size()
        .rename("tree_count")
        .reset_index()
    )
    cell_counts.insert(0, "reduced_id", np.arange(1, len(cell_counts) + 1))

    join_table = assignment.merge(
        cell_counts,
        on=["cell_id", "cell_col", "cell_row"],
        how="left",
        validate="many_to_one",
    )
    join_table = join_table[
        [
            "tree_uid",
            "source_file",
            "source_row",
            "reduced_id",
            "cell_id",
            "cell_col",
            "cell_row",
            "tree_count",
        ]
    ].sort_values(["reduced_id", "tree_uid"], kind="stable")

    centers_x = origin_x + (cell_counts["cell_col"].to_numpy() + 0.5) * cell_size
    centers_y = origin_y + (cell_counts["cell_row"].to_numpy() + 0.5) * cell_size
    unique_cells = gpd.GeoDataFrame(
        cell_counts,
        geometry=gpd.points_from_xy(centers_x, centers_y),
        crs=trees.crs,
    )

    multi_tree_cells = cell_counts.loc[cell_counts["tree_count"] > 1].copy()
    return unique_cells, join_table, multi_tree_cells


def write_outputs(
    output_dir: Path,
    unique_cells: gpd.GeoDataFrame,
    join_table: pd.DataFrame,
    multi_tree_cells: pd.DataFrame,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    unique_path = output_dir / "sentinel10m_unique_cells.shp"
    join_path = output_dir / "tree_to_sentinel10m_cell.csv"
    multi_path = output_dir / "sentinel10m_multi_tree_cells.csv"

    # Shapefile field names are limited to 10 characters.
    shapefile_cells = unique_cells.rename(
        columns={
            "reduced_id": "reduced_id",
            "tree_count": "tree_count",
            "cell_col": "cell_col",
            "cell_row": "cell_row",
        }
    )
    shapefile_cells.to_file(unique_path, driver="ESRI Shapefile", engine="pyogrio")
    join_table.to_csv(join_path, index=False)
    multi_tree_cells.to_csv(multi_path, index=False)

    print(f"Unique cell points: {unique_path}")
    print(f"Tree-to-cell join table: {join_path}")
    print(f"Multi-tree cell summary: {multi_path}")


def main() -> int:
    args = parse_args()
    input_files = discover_input_files(args.input, args.pattern)
    output_dir = args.output_dir or default_output_dir(args.input)
    trees = load_tree_points(
        input_files,
        args.target_crs,
        args.tree_id_field,
        args.longitude_field,
        args.latitude_field,
    )
    unique_cells, join_table, multi_tree_cells = assign_cells(
        trees,
        args.cell_size,
        args.origin_x,
        args.origin_y,
    )
    write_outputs(output_dir, unique_cells, join_table, multi_tree_cells)

    print(f"Input trees: {len(trees):,}")
    print(f"Unique occupied 10m cells: {len(unique_cells):,}")
    print(f"Cells containing multiple trees: {len(multi_tree_cells):,}")
    print(f"Trees removed from GEE sampling set: {len(trees) - len(unique_cells):,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
