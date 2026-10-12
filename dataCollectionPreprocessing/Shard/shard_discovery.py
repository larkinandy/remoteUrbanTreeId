"""Portable product discovery for QA and final shard assembly."""

from __future__ import annotations

from argparse import Namespace
from pathlib import Path

from dataCollectionPreprocessing.Shard.shard_common import normalize_token, sidecar_path

QACityInputs = tuple[Path, Path, Path | None, Path | None, Path | None, Path | None]


def _first_existing(candidates: list[Path]) -> Path | None:
    return next((path for path in candidates if path.exists()), None)


def find_sentinel_phenology_path(args: Namespace, city: str) -> Path | None:
    root = Path(args.sentinel_phenology_dir)
    return _first_existing([
        root / city / f"{city}{args.sentinel_phenology_suffix}",
        root / f"{city}{args.sentinel_phenology_suffix}",
    ])


def find_embedding_path(args: Namespace, city: str) -> Path | None:
    root = Path(args.gee_dir)
    candidates = sorted((root / city).glob(args.gee_sidecar_pattern)) if (root / city).exists() else []
    candidates.extend(sorted(root.glob(f"{city}{args.gee_sidecar_pattern.lstrip('*')}")))
    return candidates[0] if candidates else None


def discover_qa_city_inputs(args: Namespace) -> dict[str, QACityInputs]:
    """Find complete city products needed by the record-level QA stage."""
    selected = None if args.city_token is None else {token.strip().lower() for token in args.city_token if token.strip()}
    excluded = {token.strip().lower() for token in args.exclude_city_token}
    found: dict[str, QACityInputs] = {}
    for city_dir in sorted(path for path in args.crop_root.iterdir() if path.is_dir()):
        city = city_dir.name.strip().lower()
        if (selected is not None and city not in selected) or city in excluded:
            continue
        metadata = sorted(city_dir.glob(args.metadata_pattern))
        crops = sorted(city_dir.glob(args.crop_npy_pattern))
        structure = args.structure_dir / f"{city}{args.structure_suffix}"
        lidar_dir = args.lidar_chm_root / city
        lidar_matches = sorted(lidar_dir.glob(args.lidar_chm_pattern)) if lidar_dir.exists() else []
        lidar_chm = lidar_matches[0] if lidar_matches else None
        sentinel = find_sentinel_phenology_path(args, city)
        embedding = find_embedding_path(args, city)
        has_required_chm = lidar_chm is not None or not args.require_lidar_chm_product
        if not metadata or not crops or not has_required_chm or (not structure.exists() and lidar_chm is None):
            print(
                f"SKIP {city}: metadata={bool(metadata)} crops={bool(crops)} "
                f"lidar_chm={lidar_chm is not None} structure={structure.exists()}",
                flush=True,
            )
            continue
        found[city] = (metadata[0], crops[0], structure if structure.exists() else None, lidar_chm, sentinel, embedding)
    return found


def assembly_city_paths(args: Namespace, city: str) -> dict[str, Path]:
    """Resolve all modality products for one assembly city."""
    return {
        "qa": args.qa_root / f"{city}_tree_centered_qa_metadata.csv",
        "crop": args.crop_root / city / f"{city}_tree_id_centered_nearest_64px_rgbnir_crops.npy",
        "structure": sidecar_path(args.structure_root, city, "_tree_id_centered_chm_structure_metrics.npz"),
        "sentinel": sidecar_path(args.sentinel_phenology_root, city, "_tree_id_centered_sentinel_phenology.npz"),
        "gee": sidecar_path(args.gee_root, city, "_tree_id_centered_gee_inputs.npz"),
        "prism": sidecar_path(args.prism_root, city, "_prism_normals.npz"),
    }


def discover_assembly_cities(args: Namespace) -> list[str]:
    """Discover QA-complete cities and validate any explicit selection."""
    selected = None if args.city_token is None else {normalize_token(value) for value in args.city_token}
    excluded = {normalize_token(value) for value in args.exclude_city_token}
    available: set[str] = set()
    cities: list[str] = []
    for path in sorted(Path(args.qa_root).glob("*_tree_centered_qa_metadata.csv")):
        city = normalize_token(path.name.replace("_tree_centered_qa_metadata.csv", ""))
        available.add(city)
        if (selected is not None and city not in selected) or city in excluded:
            continue
        cities.append(city)
    if selected is not None:
        missing = sorted(selected.difference(available))
        if missing:
            raise SystemExit(
                f"Requested --city-token value(s) were not found under {args.qa_root}: {', '.join(missing)}"
            )
    return cities
