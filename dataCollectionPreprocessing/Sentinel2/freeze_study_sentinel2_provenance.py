#!/usr/bin/env python3
"""Create path-independent summaries of final Sentinel-2 phenology sidecars."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sidecar-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def scalar_count(values: np.ndarray, target: int) -> int:
    return int(np.count_nonzero(np.asarray(values) == target))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    args = parse_args()
    paths = sorted(args.sidecar_root.glob("*/*_tree_id_centered_sentinel_phenology.npz"))
    if not paths:
        raise FileNotFoundError(f"No final Sentinel-2 sidecars found under {args.sidecar_root}")

    rows = []
    feature_names: list[str] | None = None
    for path in paths:
        with np.load(path, allow_pickle=True) as data:
            names = [str(value) for value in data["sentinel_phenology_columns"].tolist()]
            if feature_names is None:
                feature_names = names
            elif names != feature_names:
                raise RuntimeError(f"Phenology schema differs in {path.name}")
            missing = np.asarray(data["missing_sentinel_phenology"], dtype=bool)
            missing_link = np.asarray(data["missing_clean_sentinel_link"], dtype=bool)
            source = np.asarray(data["sentinel_phenology_source_code"])
            distance = np.asarray(data["sentinel_timeseries_match_distance_m"], dtype=float)
            finite_distance = distance[np.isfinite(distance)]
            rows.append(
                {
                    "city_token": path.parent.name,
                    "records": int(missing.size),
                    "valid_phenology": int((~missing).sum()),
                    "missing_phenology": int(missing.sum()),
                    "missing_link": int(missing_link.sum()),
                    "original_source": scalar_count(source, 1),
                    "supplemental_source": scalar_count(source, 2),
                    "median_match_distance_m": (
                        round(float(np.median(finite_distance)), 6) if finite_distance.size else None
                    ),
                    "sidecar_filename": path.name,
                    "sidecar_bytes": int(path.stat().st_size),
                    "sidecar_sha256": sha256(path),
                }
            )

    frame = pd.DataFrame(rows).sort_values("city_token")
    frame["valid_phenology_percent"] = (
        100.0 * frame["valid_phenology"] / frame["records"].replace(0, np.nan)
    ).round(3)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output_dir / "frozen_study_sentinel2_city_summary.csv", index=False)
    global_summary = {
        "city_count": int(len(frame)),
        "records": int(frame["records"].sum()),
        "valid_phenology": int(frame["valid_phenology"].sum()),
        "missing_phenology": int(frame["missing_phenology"].sum()),
        "missing_link": int(frame["missing_link"].sum()),
        "original_source": int(frame["original_source"].sum()),
        "supplemental_source": int(frame["supplemental_source"].sum()),
        "phenology_feature_count": len(feature_names or []),
        "phenology_features": feature_names or [],
    }
    (args.output_dir / "frozen_study_sentinel2_global_summary.json").write_text(
        json.dumps(global_summary, indent=2) + "\n", encoding="utf-8"
    )
    status = {
        "final_sidecar_inventory_complete": True,
        "raw_gee_export_inventory_complete": False,
        "note": (
            "Final sidecars were inventoried with SHA-256. The historical raw GEE export "
            "volume was not mounted during repository cleanup; preserve and inventory it "
            "separately when available."
        ),
    }
    (args.output_dir / "frozen_study_sentinel2_provenance_status.json").write_text(
        json.dumps(status, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(global_summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
