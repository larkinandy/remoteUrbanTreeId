#!/usr/bin/env python3
"""Validate final NPZ shards against the canonical schema and paired metadata."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from dataCollectionPreprocessing.Shard.shard_schema import validate_shard_arrays


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--input-pattern", default="*/*_tree_centered_complete_inputs.npz")
    parser.add_argument("--city-token", action="append", default=None)
    return parser.parse_args()


def metadata_path_for(shard_path: Path) -> Path:
    return shard_path.with_name(shard_path.name.replace("_inputs.npz", "_metadata.csv"))


def validate_file(shard_path: Path) -> int:
    metadata_path = metadata_path_for(shard_path)
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing paired metadata: {metadata_path}")
    metadata = pd.read_csv(metadata_path, low_memory=False)
    with np.load(shard_path, allow_pickle=True) as arrays:
        validate_shard_arrays(arrays, metadata)
    return len(metadata)


def main() -> int:
    args = parse_args()
    selected = None if args.city_token is None else {value.strip().lower() for value in args.city_token}
    shards = [
        path for path in sorted(args.input_root.glob(args.input_pattern))
        if selected is None or path.parent.name.lower() in selected
    ]
    if not shards:
        raise SystemExit("No shards matched the requested input pattern and city selection.")
    total_rows = 0
    for index, shard_path in enumerate(shards, start=1):
        rows = validate_file(shard_path)
        total_rows += rows
        print(f"[{index}/{len(shards)}] valid: {shard_path.relative_to(args.input_root)} ({rows:,} rows)", flush=True)
    print(f"Validated {len(shards):,} shard(s) and {total_rows:,} records.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
