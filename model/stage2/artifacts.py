#!/usr/bin/env python3
"""Load and validate the frozen stage-two manuscript artifacts."""

from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPOSITORY_ROOT / "config"
SPLIT_PROVENANCE = REPOSITORY_ROOT / "analysis" / "ablation" / "provenance" / "split_manifest.json"
FINAL_GROUPING_MANIFEST = CONFIG_DIR / "final_grouping.json"
FINAL_PARTITION = CONFIG_DIR / "final_k11_partition.npz"
FINAL_COLLAPSE = CONFIG_DIR / "final_k11_to_k6_collapse.npz"
FINAL_MODEL_CONFIG = Path(__file__).resolve().parent / "final_model_config.json"


def sha256(path: Path) -> str:
    """Return the lowercase SHA-256 digest for *path*."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_grouping_manifest(path: Path = FINAL_GROUPING_MANIFEST) -> dict:
    """Load the grouping manifest and validate its frozen artifact checksums."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    checks = {
        CONFIG_DIR / payload["partition_file"]: payload["partition_sha256"],
        CONFIG_DIR / payload["collapse_artifact_file"]: payload["collapse_artifact_sha256"],
    }
    for artifact, expected in checks.items():
        observed = sha256(artifact)
        if observed != expected:
            raise RuntimeError(
                f"Frozen artifact checksum mismatch for {artifact}: "
                f"expected {expected}, observed {observed}"
            )
    if int(payload["intermediate_group_count"]) != 11:
        raise RuntimeError("The final manuscript partition must contain 11 intermediate groups.")
    if int(payload["reported_group_count"]) != 6:
        raise RuntimeError("The final manuscript collapse must contain 6 reported groups.")
    return payload


def load_collapse(path: Path = FINAL_COLLAPSE) -> tuple[np.ndarray, list[str]]:
    """Return the validated zero-based 11-to-6 collapse and display labels."""
    manifest = load_grouping_manifest()
    expected_path = CONFIG_DIR / manifest["collapse_artifact_file"]
    if path.resolve() != expected_path.resolve():
        raise ValueError(
            "The manuscript evaluator only accepts the frozen collapse artifact: "
            f"{expected_path}"
        )
    with np.load(path, allow_pickle=False) as data:
        mapping = np.asarray(data["initial_group_assignment"], dtype=np.int64)
    expected = np.asarray(manifest["collapse_map_one_based"], dtype=np.int64) - 1
    if not np.array_equal(mapping, expected):
        raise RuntimeError("Collapse artifact does not match final_grouping.json.")
    if mapping.shape != (11,) or sorted(np.unique(mapping).tolist()) != list(range(6)):
        raise RuntimeError("Frozen collapse must map 11 intermediate groups onto 6 groups.")
    return mapping, [str(value) for value in manifest["reported_group_display_labels"]]


def validate_split_manifest(path: Path) -> dict:
    """Validate the externally distributed inherited split manifest."""
    provenance = json.loads(SPLIT_PROVENANCE.read_text(encoding="utf-8"))
    observed = sha256(path)
    expected = str(provenance["sha256"])
    if observed != expected:
        raise RuntimeError(
            f"Split-manifest checksum mismatch for {path}: expected {expected}, "
            f"observed {observed}"
        )
    return provenance


def npz_schema_sha256(path: Path) -> str:
    """Fingerprint NPZ names, shapes, and dtypes without loading array payloads.

    Complete legacy shards include object-typed metadata arrays. Reading only
    each embedded NPY header avoids deserializing those objects while still
    producing the same schema representation used for provenance checks.
    """
    schema = []
    with zipfile.ZipFile(path) as archive:
        members = sorted(name for name in archive.namelist() if name.endswith(".npy"))
        for member in members:
            with archive.open(member) as stream:
                version = np.lib.format.read_magic(stream)
                if version == (1, 0):
                    shape, _, dtype = np.lib.format.read_array_header_1_0(stream)
                elif version == (2, 0):
                    shape, _, dtype = np.lib.format.read_array_header_2_0(stream)
                else:
                    raise ValueError(
                        f"Unsupported NPY header version {version} in {path}:{member}"
                    )
            schema.append(
                {
                    "name": Path(member).stem,
                    "shape": list(shape),
                    "dtype": str(dtype),
                }
            )
    encoded = json.dumps(schema, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def provenance(
    split_manifest: Path | None = None,
    input_schema_sha256: str | None = None,
) -> dict[str, str | None]:
    """Return checksums sufficient to identify a stage-two reproduction run."""
    return {
        "grouping_manifest_sha256": sha256(FINAL_GROUPING_MANIFEST),
        "partition_sha256": sha256(FINAL_PARTITION),
        "collapse_sha256": sha256(FINAL_COLLAPSE),
        "final_model_config_sha256": sha256(FINAL_MODEL_CONFIG),
        "split_manifest_sha256": None if split_manifest is None else sha256(split_manifest),
        "input_schema_sha256": input_schema_sha256,
    }


def validate_checkpoint_provenance(
    checkpoint: dict,
    current: dict[str, str | None],
    *,
    checkpoint_name: str,
) -> None:
    """Reject a checkpoint whose recorded scientific artifacts have changed."""
    recorded = checkpoint.get("provenance")
    if not recorded:
        print(
            f"WARNING: {checkpoint_name} predates embedded provenance; validating "
            "the current artifacts but cannot compare them with that checkpoint.",
            flush=True,
        )
        return
    mismatches = {
        key: (recorded.get(key), value)
        for key, value in current.items()
        if recorded.get(key) is not None and recorded.get(key) != value
    }
    if mismatches:
        details = "; ".join(
            f"{key}: checkpoint={old}, current={new}"
            for key, (old, new) in mismatches.items()
        )
        raise RuntimeError(f"Checkpoint provenance mismatch for {checkpoint_name}: {details}")
