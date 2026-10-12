#!/usr/bin/env python3
"""Freeze path-independent NAIP provenance from retained extraction outputs."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--crop-root", type=Path, required=True, help="Root containing per-city *_config.json and *_metadata.csv files.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--boundaries", type=Path, required=True, help="County GeoJSON used by the study.")
    return parser.parse_args()


def source_name(value: str) -> str:
    return Path(str(value).replace("\\", "/")).name


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    configs = sorted(args.crop_root.rglob("*_config.json"))
    if not configs:
        raise FileNotFoundError(f"No retained extraction configurations found under {args.crop_root}")

    city_rows: list[dict[str, object]] = []
    raster_counts: Counter[tuple[str, str, str, str, str]] = Counter()
    city_counties: dict[str, set[tuple[str, str]]] = {}
    city_codes: dict[str, str] = {}
    boundaries = json.loads(args.boundaries.read_text(encoding="utf-8"))
    county_names = {
        str(feature.get("id") or feature["properties"].get("GEO_ID", "")[-5:]).zfill(5): feature["properties"].get("NAME", "")
        for feature in boundaries["features"]
    }
    missing_metadata: list[str] = []
    for config_path in configs:
        config = json.loads(config_path.read_text(encoding="utf-8"))
        city = str(config["city_token"])
        index_value = str(config.get("paired_index_csv") or config.get("fiveband_index_csv") or "")
        index_name = source_name(index_value)
        city_codes[city] = index_name.split("_", 1)[0].upper() if index_name else {"baltimore": "BAL", "washingtondc": "DCA"}.get(city, "")
        city_counties[city] = set()
        source_type = "single_multiband" if config.get("fiveband_index_csv") or config.get("band_source") else "paired_rgb_cir"
        city_rows.append({
            "city_token": city,
            "source_format": source_type,
            "crop_size_pixels": config.get("crop_size", ""),
            "crop_footprint_metres": config.get("crop_metres", ""),
            "selection_mode": config.get("sid_selection_mode", ""),
            "record_count": config.get("record_count", ""),
            "failed_count": config.get("failed_count", ""),
        })
        metadata_candidates = sorted(config_path.parent.glob("*_metadata.csv"))
        if not metadata_candidates:
            missing_metadata.append(city)
            continue
        with metadata_candidates[0].open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                if source_type == "single_multiband":
                    raster = source_name(row.get("crop_source_sid", ""))
                    bands = ",".join(row.get(name, "") for name in ("crop_red_band", "crop_green_band", "crop_blue_band", "crop_nir_band"))
                    if raster:
                        year_match = re.search(r"(?:^|\D)(20\d{2})(?:\D|$)", raster)
                        raster_counts[(city, source_type, raster, bands, year_match.group(1) if year_match else "")] += 1
                        match = re.search(r"_s_([a-z]{2})(\d{3})_", raster, re.I)
                        if match:
                            city_counties[city].add((match.group(1).upper(), match.group(2)))
                else:
                    rgb = source_name(row.get("crop_natural_sid", "") or row.get("crop_source_rgb_sid", "") or row.get("crop_source_natural_sid", ""))
                    nir = source_name(row.get("crop_companion_sid", "") or row.get("crop_source_nir_sid", "") or row.get("crop_source_companion_sid", ""))
                    if rgb:
                        year_match = re.search(r"(?:^|\D)(20\d{2})(?:\D|$)", rgb)
                        raster_counts[(city, "paired_natural", rgb, "1,2,3", year_match.group(1) if year_match else "")] += 1
                        match = re.search(r"_s_([a-z]{2})(\d{3})_", rgb, re.I)
                        if match:
                            city_counties[city].add((match.group(1).upper(), match.group(2)))
                    if nir:
                        year_match = re.search(r"(?:^|\D)(20\d{2})(?:\D|$)", nir)
                        raster_counts[(city, "paired_companion", nir, "1", year_match.group(1) if year_match else "")] += 1

    city_fields = ["city_token", "source_format", "crop_size_pixels", "crop_footprint_metres", "selection_mode", "record_count", "failed_count"]
    with (args.output_dir / "study_naip_city_sources.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=city_fields)
        writer.writeheader()
        writer.writerows(sorted(city_rows, key=lambda row: str(row["city_token"])))

    with (args.output_dir / "study_naip_source_rasters.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["city_token", "source_role", "source_raster_filename", "acquisition_year", "bands_one_based", "selected_record_count"])
        for (city, role, raster, bands, year), count in sorted(raster_counts.items()):
            writer.writerow([city, role, raster, year, bands, count])

    summary_fields = [
        "city_token", "source_format", "acquisition_year_min", "acquisition_year_max",
        "unique_source_rasters", "natural_or_multiband_rasters", "companion_rasters",
        "records_submitted_for_cropping", "successful_crops", "failed_crops", "crop_success_percent",
    ]
    with (args.output_dir / "frozen_study_naip_city_summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_fields)
        writer.writeheader()
        for row in sorted(city_rows, key=lambda item: str(item["city_token"])):
            city = str(row["city_token"])
            sources = [(role, raster, year) for (source_city, role, raster, _bands, year) in raster_counts if source_city == city]
            years = sorted(int(year) for _role, _raster, year in sources if year)
            submitted = int(row["record_count"] or 0)
            failed = int(row["failed_count"] or 0)
            successful = submitted - failed
            writer.writerow({
                "city_token": city,
                "source_format": row["source_format"],
                "acquisition_year_min": min(years) if years else "",
                "acquisition_year_max": max(years) if years else "",
                "unique_source_rasters": len({raster for _role, raster, _year in sources}),
                "natural_or_multiband_rasters": len({raster for role, raster, _year in sources if role != "paired_companion"}),
                "companion_rasters": len({raster for role, raster, _year in sources if role == "paired_companion"}),
                "records_submitted_for_cropping": submitted,
                "successful_crops": successful,
                "failed_crops": failed,
                "crop_success_percent": round(100.0 * successful / submitted, 3) if submitted else "",
            })

    # Washington, DC used a locally named multiband raster without a FIPS token.
    city_counties.setdefault("washingtondc", set()).add(("DC", "001"))
    manifest_cities = []
    for row in sorted(city_rows, key=lambda item: str(item["city_token"])):
        city = str(row["city_token"])
        counties = []
        for state, county_fips in sorted(city_counties[city]):
            state_fips = {"DC": "11", "MD": "24"}.get(state)
            if state_fips is None:
                # Raster names preserve state abbreviations; recover the unique matching
                # GEOID through the original manifest builder's state table.
                from build_county_manifest import STATE_ABBR
                state_fips = next(key for key, value in STATE_ABBR.items() if value == state)
            geoid = state_fips + county_fips
            counties.append({"geoid": geoid, "state": state, "county_fips": county_fips, "name": county_names.get(geoid, "")})
        manifest_cities.append({
            "city": city,
            "code": city_codes.get(city, ""),
            "retained_crop_records": row["record_count"],
            "counties": counties,
        })
    (args.output_dir / "study_naip_county_manifest.json").write_text(
        json.dumps({"cities": manifest_cities, "basis": "County tokens in the exact source rasters selected by retained crop metadata."}, indent=2) + "\n",
        encoding="utf-8",
    )

    summary = {
        "retained_configuration_count": len(configs),
        "city_count": len(city_rows),
        "source_formats": dict(sorted(Counter(str(row["source_format"]) for row in city_rows).items())),
        "unique_city_source_raster_combinations": len(raster_counts),
        "cities_missing_retained_metadata": missing_metadata,
        "path_handling": "Source filenames are retained; workstation-specific directory prefixes are removed.",
        "original_box_selection_manifest_retained": False,
        "archive_provenance_basis": "Exact SID raster filenames and bands recorded in the completed crop metadata; current Box URLs are intentionally not substituted for the missing historical selection manifest."
    }
    (args.output_dir / "study_naip_provenance_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
