"""Shared command-line definitions for NAIP download and extraction entry points."""

import argparse
from pathlib import Path


def downloader_args(description: str, box_root: str) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=description, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--box-root-url", default=box_root)
    p.add_argument("--manifest", type=Path, required=True); p.add_argument("--output", type=Path, required=True)
    p.add_argument("--target-year", type=int, default=2022); p.add_argument("--year-window", type=int, default=6)
    p.add_argument("--city-code", action="append", default=[]); p.add_argument("--test-city", default="")
    p.add_argument("--state", action="append", default=[])
    p.add_argument("--natural-product", action="append", default=["hn", "nc"])
    p.add_argument("--companion-product", action="append", default=["hc"])
    p.add_argument("--index-cache", type=Path, required=True); p.add_argument("--selection-output", type=Path, required=True)
    p.add_argument("--skip-city-code", action="append", default=[])
    p.add_argument("--refresh-index", action="store_true"); p.add_argument("--scroll-rounds", type=int, default=80)
    p.add_argument("--settle-seconds", type=float, default=0.25); p.add_argument("--timeout-minutes", type=float, default=20)
    p.add_argument("--channel", default="msedge"); p.add_argument("--headed", action="store_true")
    p.add_argument("--retries", type=int, default=4); p.add_argument("--download-limit", type=int, default=0)
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def add_common_crop_arguments(p: argparse.ArgumentParser) -> None:
    """Add options shared by the paired and multiband crop extractors."""
    p.add_argument("--join-csv", type=Path, default=None)
    p.add_argument("--join-root", type=Path, default=None)
    p.add_argument("--join-pattern", default="{city}_tree_to_nearest_detected_crown_5m.csv")
    p.add_argument("--naip-dir", type=Path, required=True); p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--output-prefix", default=None); p.add_argument("--crop-size", type=int, default=64)
    p.add_argument("--crop-metres", type=float, default=38.0); p.add_argument("--source-epsg", default="")
    p.add_argument("--max-records", type=int, default=None); p.add_argument("--checkpoint-every", type=int, default=10000)
    p.add_argument("--sid-block-tile-metres", type=float, default=1000.0)
    p.add_argument("--sid-path-rewrite", action="append", default=[]); p.add_argument("--force", action="store_true")


def require_columns(frame, columns: list[str], path: Path) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise RuntimeError(f"{path} is missing required column(s): {missing}")


def parse_positive_bands(value: str, count: int, option: str) -> tuple[int, ...]:
    bands = tuple(int(part.strip()) for part in str(value).split(",") if part.strip())
    if len(bands) != count or min(bands, default=0) < 1:
        raise SystemExit(f"{option} must contain {count} positive one-based band numbers")
    return bands
