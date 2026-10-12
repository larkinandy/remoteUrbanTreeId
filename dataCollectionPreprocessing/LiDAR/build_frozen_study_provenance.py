#!/usr/bin/env python3
"""Freeze LiDAR inputs and project selections from retained study products.

This provenance utility is not part of ordinary preprocessing. It reconstructs
portable, reviewable tables from completed bin-state files and final LiDAR
indexes. It was used to preserve the exact tile filenames and per-record
project selections after the mutable TNM download manifest had served its
download-management purpose.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


PROJECT_YEAR_OVERRIDES = {
    "Sandy_Supplemental_NCR_VA_MD_DC_QL2_LiDAR": 2014,
    "NY_New_York_CMGP_SANDY_LiDAR_15": 2013,
    "San_Bernadino_County_Flood_Control_Lidar": 2017,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--chm-root", type=Path, required=True)
    parser.add_argument("--lidar-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def year_from_project(project: str) -> int | str:
    if project in PROJECT_YEAR_OVERRIDES:
        return PROJECT_YEAR_OVERRIDES[project]
    match = re.search(r"(?:19|20)\d{2}", project)
    if match:
        return int(match.group())
    shorthand = re.search(r"(?:^|_)[A-Z](\d{2})(?:_|$)", project, flags=re.IGNORECASE)
    return 2000 + int(shorthand.group(1)) if shorthand else ""


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    candidates: dict[str, list[dict]] = defaultdict(list)
    manifest: list[dict] = []

    for complete_path in sorted(args.candidate_root.rglob("*_bin_complete.json")):
        complete = json.loads(complete_path.read_text(encoding="utf-8"))
        state_path = complete_path.with_name(complete_path.name.replace("_bin_complete.json", "_bin_state.json"))
        if not state_path.exists():
            continue
        state = json.loads(state_path.read_text(encoding="utf-8"))
        city = str(complete["city"])
        code = str(complete["city_code"])
        project = str(complete["project"])
        rank = int(complete["preference_rank"])
        filenames = list(state.get("processed_tiles", []))
        candidates[city].append(
            {
                "city_token": city,
                "city_name": city,
                "city_code": code,
                "project": project,
                "preference_rank": rank,
                "tile_count": len(filenames),
                "crop_rows": int(complete["shape"][0]),
            }
        )
        for filename in filenames:
            local_path = args.lidar_root / code / filename
            manifest.append(
                {
                    "priority_rank": rank,
                    "status": "frozen_study_input",
                    "url": "",
                    "filename": filename,
                    "host": "",
                    "project": project,
                    "canonical_city_token": city,
                    "canonical_city_name": city,
                    "canonical_city_code": code,
                    "relative_path": str(Path(code) / filename),
                    "size_bytes": local_path.stat().st_size if local_path.exists() else "",
                    "source_note": (
                        "Exact tile used by the retained study binning stage; URL omitted because "
                        "the mutable download manifest was not retained."
                    ),
                }
            )

    summary: list[dict] = []
    projects: list[dict] = []
    for city_dir in sorted(path for path in args.chm_root.iterdir() if path.is_dir()):
        index_paths = sorted(city_dir.glob("*_lidar_index.csv"))
        city_candidates = sorted(candidates.get(city_dir.name, []), key=lambda row: row["preference_rank"])
        if not index_paths or not city_candidates:
            continue
        counts: Counter[str] = Counter()
        examples: dict[str, dict[str, str]] = {}
        with index_paths[0].open(encoding="utf-8-sig", newline="") as stream:
            for row in csv.DictReader(stream):
                project = row.get("selected_project", "")
                counts[project] += 1
                examples.setdefault(project, row)
        total_records = sum(counts.values())
        selected_project = counts.most_common(1)[0][0]
        candidate_text = ";".join(
            f"{row['project']}:coverage=0,year={year_from_project(row['project']) or 0},"
            f"event=0,stale=0,tiles={row['tile_count']},footprints=0"
            for row in city_candidates
        )
        primary = city_candidates[0]
        summary.append(
            {
                "city_token": city_dir.name,
                "city_name": city_dir.name,
                "city_code": primary["city_code"],
                "cell_count": primary["crop_rows"],
                "sampled_cell_footprints": primary["crop_rows"],
                "chip_half_size_metres": 10,
                "coverage_margin_metres": 15,
                "min_project_year": 2012,
                "preferred_project_year": 2017,
                "min_preferred_coverage": 0.85,
                "bbox": "",
                "catalog_tile_hits": "",
                "candidate_tile_count": sum(row["tile_count"] for row in city_candidates),
                "kept_tile_count": sum(row["tile_count"] for row in city_candidates),
                "project_selection": "frozen_study_products",
                "selected_project": selected_project,
                "project_selection_reason": "Recovered from retained final CHM index and completed bin state",
                "project_count": len(city_candidates),
                "top_projects": ";".join(row["project"] for row in city_candidates),
                "candidate_projects": candidate_text,
                "host_count": "",
                "hosts": "",
            }
        )
        candidate_lookup = {row["project"]: row for row in city_candidates}
        for project, record_count in counts.most_common():
            example = examples[project]
            projects.append(
                {
                    "city_token": city_dir.name,
                    "city_code": primary["city_code"],
                    "selected_project": project,
                    "acquisition_year": year_from_project(project),
                    "records": record_count,
                    "record_share": f"{record_count / total_records:.6f}",
                    "tile_count": candidate_lookup.get(project, {}).get("tile_count", ""),
                    "z_scale": example.get("z_scale", ""),
                    "z_units": example.get("z_units", ""),
                    "z_scale_confidence": example.get("z_scale_confidence", ""),
                    "z_scale_reason": example.get("z_scale_reason", ""),
                }
            )

    write_csv(
        args.output_dir / "frozen_study_lidar_tile_manifest.csv",
        sorted(manifest, key=lambda row: (row["canonical_city_code"], row["priority_rank"], row["project"], row["filename"])),
        list(manifest[0]),
    )
    write_csv(
        args.output_dir / "frozen_study_lidar_city_summary.csv",
        sorted(summary, key=lambda row: row["city_code"]),
        list(summary[0]),
    )
    write_csv(
        args.output_dir / "frozen_study_lidar_project_table.csv",
        sorted(projects, key=lambda row: (row["city_code"], row["selected_project"])),
        list(projects[0]),
    )
    print(f"Wrote {len(manifest):,} tile rows, {len(summary):,} city rows, and {len(projects):,} city-project rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
