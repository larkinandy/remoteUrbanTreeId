#!/usr/bin/env python3
"""Detect crown-center points in indexed NAIP chips using UrbanTreeDetector.

This workflow detects likely tree centers from an SFANet confidence surface. It
does not segment or delineate individual crown polygons.
"""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import yaml

from tree_crown_inference_utils import attach_index_metadata, extract_peaks, normalize_chip_shape, predict_batches
from tree_crown_common import normalize_city_token


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--detector-repo", type=Path, required=True, help="UrbanTreeDetector checkout at the frozen provenance commit.")
    parser.add_argument("--model-dir", type=Path, required=True, help="Directory containing weights.best.h5 and params.yaml.")
    parser.add_argument("--input-root", type=Path, required=True, help="City folders containing *_chips.npy and *_index.csv.")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--city-token", action="append", default=[])
    parser.add_argument("--bands", choices=("RGB", "RGBN"), default="RGBN")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--pixel-size", type=float, default=0.6)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_peak_parameters(model_dir: Path) -> dict[str, object]:
    """Load the frozen local-maximum parameters stored with the model."""
    with (model_dir / "params.yaml").open("r", encoding="utf-8") as stream:
        parameters = yaml.safe_load(stream)
    mode = parameters.get("mode", "rel")
    return {
        "min_distance": int(parameters["min_distance"]),
        "threshold_abs": float(parameters["threshold_abs"]) if mode == "abs" else None,
        "threshold_rel": float(parameters["threshold_rel"]) if mode == "rel" else None,
    }


def build_model(detector_repo: Path, model_dir: Path, input_shape: tuple[int, int, int], bands: str):
    """Import SFANet from the pinned checkout and load frozen weights."""
    sys.path.insert(0, str(detector_repo.resolve()))
    sfanet = importlib.import_module("models.SFANet")
    preprocess_module = importlib.import_module("utils.preprocess")
    preprocess = getattr(preprocess_module, f"preprocess_{bands}")
    training_model, inference_model = sfanet.build_model(input_shape, preprocess_fn=preprocess)
    training_model.load_weights(model_dir / "weights.best.h5")
    return inference_model


def discover_jobs(root: Path, selected: list[str]) -> list[tuple[str, Path, Path]]:
    """Find one chip stack and index table for each selected city."""
    requested = {normalize_city_token(token) for token in selected if normalize_city_token(token)}
    jobs: list[tuple[str, Path, Path]] = []
    for city_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        city = normalize_city_token(city_dir.name)
        if requested and city not in requested:
            continue
        chips, indexes = sorted(city_dir.glob("*_chips.npy")), sorted(city_dir.glob("*_index.csv"))
        if len(chips) != 1 or len(indexes) != 1:
            raise RuntimeError(f"{city}: expected one *_chips.npy and one *_index.csv")
        jobs.append((city, chips[0], indexes[0]))
    return jobs


def detect_city(city: str, chips_path: Path, index_path: Path, args: argparse.Namespace, peak_parameters: dict[str, object]) -> dict[str, object]:
    """Run inference for one city and write its stable crown-center table."""
    output = args.output_root / f"{city}_tree_centers.csv"
    if output.exists() and not args.overwrite:
        return {"city_token": city, "status": "exists", "output_name": output.name}
    chips = normalize_chip_shape(np.load(chips_path, mmap_mode="r"), len(args.bands))
    sample_height = max(64, int(np.ceil(chips.shape[1] / 32) * 32))
    sample_width = max(64, int(np.ceil(chips.shape[2] / 32) * 32))
    model = build_model(args.detector_repo, args.model_dir, (sample_height, sample_width, len(args.bands)), args.bands)
    frames: list[pd.DataFrame] = []
    for start, confidence in predict_batches(model, chips, args.batch_size):
        peaks = extract_peaks(confidence, **peak_parameters)
        if not peaks.empty:
            peaks["chip_index"] += start
            frames.append(peaks)
        print(f"{city}: processed {start + len(confidence):,}/{len(chips):,} chips", flush=True)
    peaks = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["chip_index", "x", "y", "confidence"])
    peaks = attach_index_metadata(peaks, index_path, chips.shape[1:3], args.pixel_size)
    output.parent.mkdir(parents=True, exist_ok=True)
    peaks.to_csv(output, index=False)
    return {"city_token": city, "status": "completed", "chips": len(chips), "detections": len(peaks), "output_name": output.name}


def validate_args(args: argparse.Namespace) -> None:
    """Fail before model loading when inputs or frozen assets are missing."""
    for path in (args.detector_repo, args.model_dir, args.input_root):
        if not path.is_dir():
            raise FileNotFoundError(path)
    for name in ("weights.best.h5", "params.yaml"):
        if not (args.model_dir / name).is_file():
            raise FileNotFoundError(args.model_dir / name)
    if args.batch_size <= 0 or args.pixel_size <= 0:
        raise ValueError("--batch-size and --pixel-size must be positive")


def main() -> int:
    args = parse_args()
    validate_args(args)
    args.output_root.mkdir(parents=True, exist_ok=True)
    peak_parameters = load_peak_parameters(args.model_dir)
    jobs = discover_jobs(args.input_root, args.city_token)
    if not jobs:
        raise SystemExit("No city chip stacks selected.")
    results = [detect_city(city, chips, index, args, peak_parameters) for city, chips, index in jobs]
    summary = pd.DataFrame(results).sort_values("city_token")
    summary.to_csv(args.output_root / "tree_crown_detection_summary.csv", index=False)
    (args.output_root / "tree_crown_detection_config.json").write_text(
        json.dumps({"bands": args.bands, "batch_size": args.batch_size, "pixel_size": args.pixel_size, "peak_parameters": peak_parameters}, indent=2),
        encoding="utf-8",
    )
    print(summary.to_string(index=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
