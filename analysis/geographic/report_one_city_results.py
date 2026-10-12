#!/usr/bin/env python3
"""Regenerate one-city publication summaries from a prepared source-model table."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from geographic_reporting import write_one_city_reports


def parse_args() -> argparse.Namespace:
    """Parse prepared-table input and publication-output paths."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-model-summaries", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    """Regenerate compact tables without loading model predictions."""
    arguments = parse_args()
    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    summaries = pd.read_csv(arguments.source_model_summaries)
    overall, comparison = write_one_city_reports(summaries, arguments.output_dir)
    print(overall.to_string(index=False))
    print(f"\nSource comparisons: {len(comparison)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
