#!/usr/bin/env python3
"""Validate the frozen manuscript grouping configuration and taxon table.

The validator treats ``final_grouping.json`` as the manifest for four compact
NPZ artifacts. It verifies their checksums, schemas, shapes, indexing, taxon
ordering, core/rare composition, prototype consistency, and exact 11-to-6
collapse. It also confirms that the human-readable taxon table is a lossless
rendering of those binary inputs.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


CONFIG_DIR = Path(__file__).resolve().parent
DEFAULT_MANIFEST = CONFIG_DIR / "final_grouping.json"
DEFAULT_TABLE = CONFIG_DIR / "frozen_taxon_group_mapping.csv"
DEFAULT_ARTIFACT_MANIFEST = CONFIG_DIR / "artifact_manifest.csv"
TABLE_COLUMNS = [
    "taxon",
    "taxon_provenance",
    "k11_group",
    "collapsed_k6_group",
    "core_validation_records",
    "rare_development_records",
    "rare_held_out_test_records",
]
ARTIFACT_MANIFEST_COLUMNS = ["file_name", "role", "bytes", "sha256"]
ARTIFACT_ROLES = {
    "final_grouping.json": "Grouping manifest and selection provenance",
    "core_k11_partition.npz": "Core-taxon assignments and validation centroids",
    "core_k11_group_prototypes.npz": "Frozen core-derived k11 prototypes",
    "final_k11_partition.npz": "Expanded core-plus-rare k11 partition",
    "final_k11_to_k6_collapse.npz": "Frozen k11-to-collapsed_k6 mapping",
    "frozen_taxon_group_mapping.csv": "Human-readable taxon mapping",
}


def parse_args() -> argparse.Namespace:
    """Parse manifest, table, and deliberate table-regeneration options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
        help="Grouping manifest JSON to validate.",
    )
    parser.add_argument(
        "--table",
        type=Path,
        default=DEFAULT_TABLE,
        help="Human-readable frozen taxon mapping CSV.",
    )
    parser.add_argument(
        "--write-table",
        action="store_true",
        help="Generate the taxon mapping table from validated NPZ artifacts.",
    )
    parser.add_argument(
        "--artifact-manifest",
        type=Path,
        default=DEFAULT_ARTIFACT_MANIFEST,
        help="Frozen CSV containing artifact roles, sizes, and checksums.",
    )
    parser.add_argument(
        "--write-manifest",
        action="store_true",
        help="Generate the artifact manifest from the validated configuration files.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow --write-table to replace an existing table.",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    """Calculate a file's SHA-256 digest without loading it all into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_arrays(artifact: np.lib.npyio.NpzFile, required: set[str], name: str) -> None:
    """Reject an NPZ artifact that omits any required named array."""
    missing = required - set(artifact.files)
    if missing:
        raise ValueError(f"{name} is missing arrays: {sorted(missing)}")


def require_integer_groups(values: np.ndarray, count: int, name: str) -> np.ndarray:
    """Validate a complete zero-based group vector and return it as int64."""
    array = np.asarray(values)
    if not np.issubdtype(array.dtype, np.integer) or array.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional integer array")
    if np.any(array < 0) or np.any(array >= count):
        raise ValueError(f"{name} contains assignments outside [0, {count - 1}]")
    if set(array.tolist()) != set(range(count)):
        raise ValueError(f"{name} does not represent every expected group")
    return array.astype(np.int64, copy=False)


def require_dtype(values: np.ndarray, kind: str, name: str) -> np.ndarray:
    """Require a NumPy dtype kind without silently coercing malformed arrays."""
    array = np.asarray(values)
    if array.dtype.kind not in kind:
        raise ValueError(f"{name} has dtype {array.dtype}; expected kind in {kind!r}")
    return array


def resolve_artifact(manifest_path: Path, file_name: str) -> Path:
    """Resolve an artifact beside the manifest while preventing path escape."""
    base = manifest_path.resolve().parent
    candidate = (base / file_name).resolve()
    if candidate.parent != base:
        raise ValueError(f"Artifact must be directly inside the config directory: {file_name}")
    if not candidate.is_file():
        raise FileNotFoundError(candidate)
    return candidate


def expected_table_rows(data: dict[str, np.ndarray], collapse_map: np.ndarray) -> list[dict[str, str]]:
    """Construct the canonical human-readable rows from validated arrays."""
    labels = data["final_labels"]
    assignments = data["final_assignments"]
    core_count = int(data["core_count"])
    core_support = data["core_support"]
    rare_development = data["rare_development"]
    rare_test = data["rare_test"]
    rows: list[dict[str, str]] = []
    for index, (taxon, k11_group) in enumerate(zip(labels, assignments)):
        is_core = index < core_count
        rare_index = index - core_count
        rows.append({
            "taxon": str(taxon),
            "taxon_provenance": "core_ge500" if is_core else "rare_50_499",
            "k11_group": str(int(k11_group) + 1),
            "collapsed_k6_group": str(int(collapse_map[int(k11_group)]) + 1),
            "core_validation_records": str(int(core_support[index])) if is_core else "",
            "rare_development_records": "" if is_core else str(int(rare_development[rare_index])),
            "rare_held_out_test_records": "" if is_core else str(int(rare_test[rare_index])),
        })
    return rows


def write_table(path: Path, rows: list[dict[str, str]], overwrite: bool) -> None:
    """Write the deterministic CSV table, refusing replacement unless authorized."""
    if path.exists() and not overwrite:
        raise FileExistsError(f"Table already exists; pass --overwrite to replace it: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=TABLE_COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def validate_table(path: Path, expected: list[dict[str, str]]) -> None:
    """Confirm exact columns, ordering, and values in the frozen mapping table."""
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != TABLE_COLUMNS:
            raise ValueError(f"Unexpected table columns: {reader.fieldnames}")
        actual = list(reader)
    if actual != expected:
        mismatch = next(
            (index for index, pair in enumerate(zip(actual, expected)) if pair[0] != pair[1]),
            min(len(actual), len(expected)),
        )
        raise ValueError(
            f"Frozen taxon table differs from binary artifacts at row {mismatch + 2}; "
            f"actual_rows={len(actual)}, expected_rows={len(expected)}"
        )


def expected_artifact_manifest_rows(manifest_path: Path, table_path: Path) -> list[dict[str, str]]:
    """Build deterministic artifact-manifest rows from the frozen file set."""
    configuration = json.loads(manifest_path.read_text(encoding="utf-8"))
    names = [
        manifest_path.name,
        str(configuration["core_partition_file"]),
        str(configuration["core_group_prototypes_file"]),
        str(configuration["partition_file"]),
        str(configuration["collapse_artifact_file"]),
        table_path.name,
    ]
    rows: list[dict[str, str]] = []
    for name in names:
        path = resolve_artifact(manifest_path, name)
        rows.append({
            "file_name": name,
            "role": ARTIFACT_ROLES[name],
            "bytes": str(path.stat().st_size),
            "sha256": sha256(path),
        })
    return rows


def write_csv_manifest(path: Path, rows: list[dict[str, str]], overwrite: bool) -> None:
    """Write the deterministic artifact manifest with overwrite protection."""
    if path.exists() and not overwrite:
        raise FileExistsError(f"Manifest already exists; pass --overwrite to replace it: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=ARTIFACT_MANIFEST_COLUMNS,
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def validate_csv_manifest(path: Path, expected: list[dict[str, str]]) -> None:
    """Confirm exact artifact-manifest columns, order, sizes, roles, and hashes."""
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ARTIFACT_MANIFEST_COLUMNS:
            raise ValueError(f"Unexpected artifact-manifest columns: {reader.fieldnames}")
        actual = list(reader)
    if actual != expected:
        raise ValueError("Artifact manifest is stale or differs from the frozen file set")


def validate_configuration(manifest_path: Path) -> tuple[dict[str, Any], list[dict[str, str]]]:
    """Validate all frozen artifacts and return an audit plus canonical CSV rows."""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    intermediate_count = int(manifest["intermediate_group_count"])
    reported_count = int(manifest["reported_group_count"])
    embedding_dimension = int(manifest["embedding_dimension"])
    artifact_specs = {
        "final": ("partition_file", "partition_sha256"),
        "core": ("core_partition_file", "core_partition_sha256"),
        "prototypes": ("core_group_prototypes_file", "core_group_prototypes_sha256"),
        "collapse": ("collapse_artifact_file", "collapse_artifact_sha256"),
    }
    paths: dict[str, Path] = {}
    for name, (file_field, hash_field) in artifact_specs.items():
        path = resolve_artifact(manifest_path, str(manifest[file_field]))
        digest = sha256(path)
        if digest != manifest[hash_field]:
            raise ValueError(f"Checksum mismatch for {path.name}: {digest}")
        paths[name] = path

    with np.load(paths["core"], allow_pickle=False) as core:
        require_arrays(core, {"assignments", "label_columns", "validation_counts", "full_validation_centroids"}, "core partition")
        core_labels = require_dtype(core["label_columns"], "US", "core labels").astype(str)
        core_assignments = require_integer_groups(core["assignments"], intermediate_count, "core assignments")
        core_support = require_dtype(core["validation_counts"], "iu", "core validation counts")
        core_centroids = require_dtype(core["full_validation_centroids"], "f", "core centroids")
    with np.load(paths["prototypes"], allow_pickle=False) as prototypes:
        require_arrays(prototypes, {"group_centroids", "label_columns", "assignments", "validation_counts"}, "core prototypes")
        prototype_centroids = require_dtype(prototypes["group_centroids"], "f", "group prototypes")
        prototype_labels = require_dtype(prototypes["label_columns"], "US", "prototype labels").astype(str)
        prototype_assignments = require_integer_groups(prototypes["assignments"], intermediate_count, "prototype assignments")
        prototype_support = require_dtype(prototypes["validation_counts"], "iu", "prototype validation counts")
    with np.load(paths["final"], allow_pickle=False) as final:
        require_arrays(final, {"assignments", "group_centroids", "label_columns", "core_validation_counts", "rare_development_counts", "rare_test_counts", "core_taxon_count", "rare_extension_taxon_count"}, "final partition")
        final_labels = require_dtype(final["label_columns"], "US", "final labels").astype(str)
        final_assignments = require_integer_groups(final["assignments"], intermediate_count, "final assignments")
        final_centroids = require_dtype(final["group_centroids"], "f", "final group centroids")
        final_core_support = require_dtype(final["core_validation_counts"], "iu", "final core counts")
        rare_development = require_dtype(final["rare_development_counts"], "iu", "rare development counts")
        rare_test = require_dtype(final["rare_test_counts"], "iu", "rare held-out counts")
        core_count = int(np.asarray(final["core_taxon_count"]).reshape(-1)[0])
        rare_count = int(np.asarray(final["rare_extension_taxon_count"]).reshape(-1)[0])
    with np.load(paths["collapse"], allow_pickle=False) as collapse:
        require_arrays(collapse, {"initial_group_assignment", "initial_group_names", "label_columns", "label_group"}, "collapse artifact")
        collapse_map = require_integer_groups(collapse["initial_group_assignment"], reported_count, "collapse map")
        collapse_labels = require_dtype(collapse["label_columns"], "US", "collapse labels").astype(str)
        collapsed_taxa = require_integer_groups(collapse["label_group"], reported_count, "collapsed taxon assignments")
        group_names = require_dtype(collapse["initial_group_names"], "US", "initial group names")

    expected_core = int(manifest["core_taxon_count"])
    expected_rare = int(manifest["rare_extension_taxon_count"])
    expected_total = int(manifest["total_taxon_count"])
    if (core_count, rare_count, len(final_labels)) != (expected_core, expected_rare, expected_total):
        raise ValueError("Manifest and final-partition taxon counts disagree")
    expected_shapes = {
        "core_labels": (expected_core,),
        "core_assignments": (expected_core,),
        "core_support": (expected_core,),
        "core_centroids": (expected_core, embedding_dimension),
        "prototype_centroids": (intermediate_count, embedding_dimension),
        "final_labels": (expected_total,),
        "final_assignments": (expected_total,),
        "final_centroids": (intermediate_count, embedding_dimension),
        "rare_development": (expected_rare,),
        "rare_test": (expected_rare,),
        "collapse_map": (intermediate_count,),
        "collapsed_taxa": (expected_total,),
        "group_names": (intermediate_count,),
    }
    arrays = {
        "core_labels": core_labels,
        "core_assignments": core_assignments,
        "core_support": core_support,
        "core_centroids": core_centroids,
        "prototype_centroids": prototype_centroids,
        "final_labels": final_labels,
        "final_assignments": final_assignments,
        "final_centroids": final_centroids,
        "rare_development": rare_development,
        "rare_test": rare_test,
        "collapse_map": collapse_map,
        "collapsed_taxa": collapsed_taxa,
        "group_names": group_names,
    }
    for name, shape in expected_shapes.items():
        if arrays[name].shape != shape:
            raise ValueError(f"Unexpected shape for {name}: {arrays[name].shape}; expected {shape}")
    if not np.array_equal(final_labels[:expected_core], core_labels):
        raise ValueError("Core taxa are not the exact prefix of the final partition")
    if not np.array_equal(final_assignments[:expected_core], core_assignments):
        raise ValueError("Core assignments differ between core and final partitions")
    if len(np.unique(final_labels)) != expected_total or np.any(np.char.str_len(final_labels) == 0):
        raise ValueError("Final taxon labels must be nonempty and unique")
    if not np.array_equal(prototype_labels, core_labels):
        raise ValueError("Prototype taxon order differs from the core partition")
    if not np.array_equal(prototype_assignments, core_assignments):
        raise ValueError("Prototype assignments differ from the core partition")
    if not np.array_equal(prototype_support, core_support) or not np.array_equal(final_core_support, core_support):
        raise ValueError("Core validation support differs across artifacts")
    if not np.allclose(prototype_centroids, final_centroids, atol=0.0, rtol=0.0):
        raise ValueError("Final group centroids differ from frozen core prototypes")
    if not np.array_equal(collapse_labels, final_labels):
        raise ValueError("Collapse taxon order differs from the final partition")
    if not np.array_equal(collapsed_taxa, collapse_map[final_assignments]):
        raise ValueError("Stored taxon-level collapsed groups do not reproduce the collapse map")
    json_map = np.asarray(manifest["collapse_map_one_based"], dtype=np.int64) - 1
    if not np.array_equal(json_map, collapse_map):
        raise ValueError("JSON and NPZ collapse mappings disagree")
    if len(manifest["intermediate_group_display_labels"]) != intermediate_count:
        raise ValueError("Intermediate display-label count does not match k11")
    if len(manifest["reported_group_display_labels"]) != reported_count:
        raise ValueError("Reported display-label count does not match collapsed_k6")
    indexing = manifest["group_indexing"]
    if indexing["npz_arrays"] != "zero-based" or indexing["manuscript_displays"] != "one-based":
        raise ValueError("Unexpected group-indexing convention")

    table_data = {
        "final_labels": final_labels,
        "final_assignments": final_assignments,
        "core_count": np.asarray(core_count),
        "core_support": core_support,
        "rare_development": rare_development,
        "rare_test": rare_test,
    }
    rows = expected_table_rows(table_data, collapse_map)
    audit = {
        "schema_version": int(manifest["schema_version"]),
        "artifacts_validated": len(paths),
        "core_taxa": core_count,
        "rare_extension_taxa": rare_count,
        "total_taxa": len(final_labels),
        "embedding_dimension": embedding_dimension,
        "k11_group_taxon_counts": np.bincount(final_assignments, minlength=intermediate_count).tolist(),
        "collapsed_k6_group_taxon_counts": np.bincount(collapsed_taxa, minlength=reported_count).tolist(),
        "checksums_valid": True,
        "cross_artifact_consistency_valid": True,
    }
    return audit, rows


def main() -> int:
    """Validate artifacts, optionally regenerate the table, and print an audit."""
    arguments = parse_args()
    audit, rows = validate_configuration(arguments.manifest)
    if arguments.write_table:
        write_table(arguments.table, rows, arguments.overwrite)
    validate_table(arguments.table, rows)
    manifest = json.loads(arguments.manifest.read_text(encoding="utf-8"))
    expected_table_path = resolve_artifact(
        arguments.manifest, str(manifest["taxon_mapping_table_file"])
    )
    if arguments.table.resolve() != expected_table_path:
        raise ValueError("Validated table path differs from the table recorded in the manifest")
    table_digest = sha256(arguments.table)
    if table_digest != manifest["taxon_mapping_table_sha256"]:
        raise ValueError(f"Frozen taxon table checksum mismatch: {table_digest}")
    manifest_rows = expected_artifact_manifest_rows(arguments.manifest, arguments.table)
    if arguments.write_manifest:
        write_csv_manifest(arguments.artifact_manifest, manifest_rows, arguments.overwrite)
    validate_csv_manifest(arguments.artifact_manifest, manifest_rows)
    audit["frozen_taxon_table_valid"] = True
    audit["frozen_taxon_table_rows"] = len(rows)
    audit["artifact_manifest_valid"] = True
    audit["artifact_manifest_rows"] = len(manifest_rows)
    print(json.dumps(audit, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
