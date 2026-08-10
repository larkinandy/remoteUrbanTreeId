#!/usr/bin/env python3
"""Run UrbanTreeDetector and build the crown-center CSV used by this pipeline.

UrbanTreeDetector writes one GeoJSON point file per input raster. This wrapper
runs its official inference module, consolidates those points for one city, and
writes ``<city>_tree_centers.csv`` for the inventory-to-crown spatial join.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--detector-repo", type=Path, required=True, help="Clone of jonathanventura/urban-tree-detection.")
    parser.add_argument("--input", type=Path, required=True, help="Input NAIP TIFF or directory of TIFFs.")
    parser.add_argument("--city-token", required=True, help="Lowercase city identifier used by downstream products.")
    parser.add_argument("--log-dir", type=Path, required=True, help="Detector model directory containing weights.best.h5.")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--geojson-dir", type=Path, default=None, help="Intermediate detector GeoJSON directory.")
    parser.add_argument("--python-executable", default=sys.executable, help="Python from the detector conda environment.")
    parser.add_argument("--bands", choices=("RGB", "RGBN"), default="RGBN")
    parser.add_argument("--tile-size", type=int, default=2048)
    parser.add_argument("--overlap", type=int, default=32)
    parser.add_argument("--cell-epsg", type=int, default=None, help="Fallback EPSG when GeoJSON CRS metadata is absent.")
    parser.add_argument("--skip-inference", action="store_true", help="Only consolidate existing GeoJSON files.")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not args.detector_repo.is_dir():
        raise SystemExit(f"Detector repository not found: {args.detector_repo}")
    if not args.input.exists():
        raise SystemExit(f"Input not found: {args.input}")
    if not args.log_dir.is_dir():
        raise SystemExit(f"Detector log directory not found: {args.log_dir}")
    weights = args.log_dir / "weights.best.h5"
    if not weights.is_file():
        raise SystemExit(f"Detector weights not found: {weights}")
    if args.tile_size <= 0 or args.overlap <= 0:
        raise SystemExit("--tile-size and --overlap must be positive.")


def detector_output_path(input_path: Path, geojson_dir: Path) -> Path:
    return geojson_dir if input_path.is_dir() else geojson_dir / f"{input_path.stem}.json"


def run_detector(args: argparse.Namespace, geojson_dir: Path) -> None:
    output = detector_output_path(args.input, geojson_dir)
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        str(args.python_executable),
        "-m",
        "scripts.inference",
        str(args.input.resolve()),
        str(output.resolve()),
        str(args.log_dir.resolve()),
        "--bands",
        args.bands,
        "--tile_size",
        str(args.tile_size),
        "--overlap",
        str(args.overlap),
    ]
    print("Running UrbanTreeDetector:", " ".join(command), flush=True)
    subprocess.run(command, cwd=args.detector_repo, check=True)


def epsg_from_crs(crs: Any) -> int | None:
    if crs is None:
        return None
    match = re.search(r"(?:EPSG[:/]|::)(\d{4,6})", json.dumps(crs), flags=re.IGNORECASE)
    return int(match.group(1)) if match else None


def read_points(path: Path, fallback_epsg: int | None) -> list[dict[str, object]]:
    with path.open("r", encoding="utf-8") as stream:
        document = json.load(stream)
    epsg = epsg_from_crs(document.get("crs")) or fallback_epsg
    if epsg is None:
        raise RuntimeError(f"Cannot determine projected EPSG for {path}; supply --cell-epsg.")
    rows: list[dict[str, object]] = []
    for feature_index, feature in enumerate(document.get("features", [])):
        geometry = feature.get("geometry") or {}
        coordinates = geometry.get("coordinates") or []
        if geometry.get("type") != "Point" or len(coordinates) < 2:
            continue
        properties = feature.get("properties") or {}
        confidence = properties.get("confidence")
        is_proxy = confidence is None
        rows.append(
            {
                "approx_x": float(coordinates[0]),
                "approx_y": float(coordinates[1]),
                # Official UrbanTreeDetector GeoJSON currently omits peak scores.
                # A value of 1 keeps detections compatible with the downstream
                # confidence filter; confidence_is_proxy prevents misinterpretation.
                "confidence": 1.0 if is_proxy else float(confidence),
                "confidence_is_proxy": is_proxy,
                "cell_epsg": epsg,
                "source_raster": path.stem,
                "source_feature_index": feature_index,
            }
        )
    return rows


def consolidate(geojson_dir: Path, output_csv: Path, fallback_epsg: int | None) -> int:
    paths = sorted({*geojson_dir.glob("*.json"), *geojson_dir.glob("*.geojson")})
    if not paths:
        raise FileNotFoundError(f"No detector GeoJSON files found under {geojson_dir}")
    rows: list[dict[str, object]] = []
    seen: set[tuple[int, float, float]] = set()
    for path in paths:
        for row in read_points(path, fallback_epsg):
            key = (int(row["cell_epsg"]), round(float(row["approx_x"]), 6), round(float(row["approx_y"]), 6))
            if key not in seen:
                seen.add(key)
                rows.append(row)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "approx_x", "approx_y", "confidence", "confidence_is_proxy", "cell_epsg",
        "source_raster", "source_feature_index",
    ]
    with output_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def main() -> int:
    args = parse_args()
    validate_args(args)
    city = args.city_token.strip().lower()
    if not city or not re.fullmatch(r"[a-z0-9_-]+", city):
        raise SystemExit("--city-token may contain only lowercase letters, digits, underscores, and hyphens.")
    geojson_dir = args.geojson_dir or args.output_root / "detector_geojson" / city
    output_csv = args.output_root / f"{city}_tree_centers.csv"
    if output_csv.exists() and not args.force:
        raise SystemExit(f"Output already exists (use --force to replace): {output_csv}")
    if not args.skip_inference:
        run_detector(args, geojson_dir)
    count = consolidate(geojson_dir, output_csv, args.cell_epsg)
    print(f"Wrote {count:,} detected crown center(s): {output_csv}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
