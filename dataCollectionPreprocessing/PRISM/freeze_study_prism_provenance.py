#!/usr/bin/env python3
"""Freeze path-independent PRISM sidecar and source-archive provenance."""

import argparse
import csv
import hashlib
import json
import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sidecar-root", type=Path, required=True)
    parser.add_argument("--archive-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--allow-missing-archive-root", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    args = arguments()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    source_summary = pd.read_csv(args.sidecar_root / "prism_daily_sidecar_summary.csv")
    source_summary = source_summary.drop(columns=[column for column in ("output",) if column in source_summary.columns])
    source_summary["mapping_percent"] = np.where(
        source_summary["trees"] > 0,
        100.0 * source_summary["valid_tree_mappings"] / source_summary["trees"],
        np.nan,
    ).round(3)
    source_summary["coverage_status"] = np.where(
        source_summary["trees"] == 0,
        "no_records",
        np.where(
            source_summary["valid_tree_mappings"] == 0,
            "no_coverage",
            np.where(source_summary["missing_tree_mappings"] > 0, "partial", "complete"),
        ),
    )
    source_summary.to_csv(args.output_dir / "frozen_study_prism_city_summary.csv", index=False)

    global_values = np.load(args.sidecar_root / "prism_daily_values.npz", allow_pickle=False)
    config = json.loads(global_values["config_json"].item())
    grid = config["reference_raster"]
    frozen_global = {
        "prism_pixels": int(global_values["values"].shape[0]),
        "dates": int(global_values["values"].shape[1]),
        "variables": global_values["variable_names"].tolist(),
        "first_date": int(global_values["dates_yyyymmdd"][0]),
        "last_date": int(global_values["dates_yyyymmdd"][-1]),
        "valid_value_percent": round(float(global_values["valid_mask"].mean() * 100.0), 6),
        "reference_raster": grid,
        "city_count": int(len(source_summary)),
        "tree_records": int(source_summary["trees"].sum()),
        "valid_tree_mappings": int(source_summary["valid_tree_mappings"].sum()),
        "missing_tree_mappings": int(source_summary["missing_tree_mappings"].sum()),
    }
    (args.output_dir / "frozen_study_prism_global_summary.json").write_text(json.dumps(frozen_global, indent=2) + "\n", encoding="utf-8")

    rows = []
    if args.archive_root.exists():
        for path in sorted(args.archive_root.rglob("*.zip")):
            match = re.search(r"prism_([^_]+)_.*_(\d{8})\.zip$", path.name, re.I)
            variable = path.parent.name.lower()
            date_code = match.group(2) if match else ""
            try:
                with zipfile.ZipFile(path) as archive:
                    bad_member = archive.testzip()
                integrity = "valid" if bad_member is None else f"bad_member:{bad_member}"
            except (OSError, zipfile.BadZipFile) as error:
                integrity = f"invalid:{type(error).__name__}"
            rows.append({
                "variable": variable,
                "date": date_code,
                "filename": path.name,
                "source_url": f"https://ftp.prism.oregonstate.edu/time_series/us/an/4km/{variable}/daily/{date_code[:4]}/{path.name}" if date_code else "",
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
                "zip_integrity": integrity,
                "used_by_final_sidecar": date_code and 20210101 <= int(date_code) <= 20231231 and variable in {"ppt", "tmean", "vpdmax"},
            })
    elif not args.allow_missing_archive_root:
        raise FileNotFoundError(args.archive_root)

    fields = ["variable", "date", "filename", "source_url", "bytes", "sha256", "zip_integrity", "used_by_final_sidecar"]
    with (args.output_dir / "frozen_study_prism_archive_inventory.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)
    status = {
        "archive_root_available": args.archive_root.exists(),
        "archive_count": len(rows),
        "checksum_algorithm": "SHA-256",
        "note": "Rerun without --allow-missing-archive-root when the retained archive volume is mounted." if not rows else "Exact retained archive inventory completed."
    }
    (args.output_dir / "frozen_study_prism_provenance_status.json").write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({**frozen_global, **status}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
