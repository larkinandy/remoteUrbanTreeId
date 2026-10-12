#!/usr/bin/env python3
"""Reusable command-line argument groups for stage-two entry points."""

from __future__ import annotations

import argparse
from pathlib import Path


DEFAULT_SHARD_PATTERN = "*_part*_tree_centered_complete_inputs.npz"


def add_shard_io_arguments(parser: argparse.ArgumentParser) -> None:
    """Add portable input-shard and output-root arguments."""
    parser.add_argument("--complete-shard-root", type=Path, required=True)
    parser.add_argument("--complete-shard-pattern", default=DEFAULT_SHARD_PATTERN)
    parser.add_argument("--output-dir", type=Path, required=True)


def add_split_manifest_argument(parser: argparse.ArgumentParser, *, required: bool) -> None:
    """Add the inherited split-manifest argument."""
    parser.add_argument(
        "--split-manifest",
        type=Path,
        required=required,
        default=None,
        help=(
            "CSV or CSV.GZ containing unique record_key and split columns. "
            "The final manuscript workflow requires its frozen external artifact."
        ),
    )


def add_overwrite_argument(parser: argparse.ArgumentParser) -> None:
    """Add the repository-standard overwrite switch."""
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Permit writing into an existing non-empty output directory.",
    )
