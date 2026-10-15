#!/usr/bin/env python3
"""Count retained QA-warning records in a fixed analytical split manifest."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--qa-root", type=Path, required=True)
    return parser.parse_args()


def as_boolean(values: pd.Series) -> pd.Series:
    return values.astype(str).str.strip().str.lower().isin({"true", "1", "yes"})


def main() -> int:
    args = parse_args()
    manifest = pd.read_csv(args.split_manifest, usecols=["record_key"])
    wanted = set(manifest["record_key"].astype(str))
    matched = warning = low_height = other_warning = 0

    for path in sorted(args.qa_root.glob("*_tree_centered_qa_metadata.csv")):
        city = path.name.removesuffix("_tree_centered_qa_metadata.csv")
        frame = pd.read_csv(
            path,
            usecols=[
                "tree_id",
                "crop_index",
                "qa_flag_low_vegetated_height_with_lidar",
                "qa_any_warning_flag",
            ],
        )
        keys = city + "|" + frame["tree_id"].astype(str) + "|" + frame["crop_index"].astype(str)
        selected = frame.loc[keys.isin(wanted)]
        any_warning = as_boolean(selected["qa_any_warning_flag"])
        low = as_boolean(selected["qa_flag_low_vegetated_height_with_lidar"])
        matched += len(selected)
        warning += int(any_warning.sum())
        low_height += int(low.sum())
        other_warning += int((any_warning & ~low).sum())

    total = len(manifest)
    if matched != total:
        raise RuntimeError(f"Matched {matched:,} of {total:,} manifest records")
    no_warning = total - warning
    print(f"total={total:,}")
    print(f"no_warning={no_warning:,} ({no_warning / total:.2%})")
    print(f"any_warning={warning:,} ({warning / total:.2%})")
    print(f"low_vegetated_height={low_height:,} ({low_height / total:.2%})")
    print(f"other_warning={other_warning:,} ({other_warning / total:.2%})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
