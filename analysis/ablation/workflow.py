"""Shared validation helpers for the frozen ablation workflow."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_split_manifest(path: Path) -> None:
    metadata_path = Path(__file__).resolve().parent / "provenance" / "split_manifest.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    observed = sha256(path)
    if observed != metadata["sha256"]:
        raise ValueError(
            f"Split manifest checksum mismatch for {path}: {observed}; "
            f"expected {metadata['sha256']}"
        )


def canonical_collapse_map() -> str:
    config = json.loads((ROOT / "config" / "final_grouping.json").read_text(encoding="utf-8"))
    return ",".join(str(value) for value in config["collapse_map_one_based"])
