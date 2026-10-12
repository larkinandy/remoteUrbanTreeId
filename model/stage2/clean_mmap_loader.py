"""Memory-mapped clean-shard store and dataset compatible with stage-two training."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any

import numpy as np

DISCRIMINATOR_DIR = Path(__file__).resolve().parents[1] / "discriminator"
if str(DISCRIMINATOR_DIR) not in sys.path:
    sys.path.insert(0, str(DISCRIMINATOR_DIR))

import train_taxon_discriminator as clean

torch = clean.train_base.torch


@dataclass
class MmapShardArrays:
    path: Path
    crops: np.ndarray
    chm: np.ndarray
    chm_interaction: np.ndarray
    structure: np.ndarray
    sentinel_phenology: np.ndarray | None
    sentinel_sequence: np.ndarray | None
    sentinel_sequence_mask: np.ndarray | None
    satellite_embedding: np.ndarray | None
    prism_normals: np.ndarray | None = None


class MmapShardStore:
    """Open standalone NPY sidecars lazily; OS page cache is shared by workers."""

    def __init__(self, source_paths: dict[str, Path], prepared_root: Path):
        self.source_paths = dict(source_paths)
        self.prepared_root = Path(prepared_root).resolve()
        self.sidecars = self._discover()
        self.cache: dict[str, MmapShardArrays] = {}

    def _discover(self) -> dict[str, dict[str, str]]:
        by_source: dict[str, dict[str, str]] = {}
        raster_manifests = list(self.prepared_root.rglob("*_prepared_rasters_manifest.json"))
        vector_manifests = list(self.prepared_root.rglob("*_mmap_vectors_manifest.json"))
        for path, kind in [(p, "raster") for p in raster_manifests] + [(p, "vector") for p in vector_manifests]:
            payload = json.loads(path.read_text(encoding="utf-8"))
            source = str(Path(payload["source_shard"]).resolve()).lower()
            entry = by_source.setdefault(source, {})
            if kind in entry:
                raise RuntimeError(f"Duplicate {kind} sidecar manifest for {source}")
            entry[kind] = str(path)
            entry.update(payload["outputs"])
        result: dict[str, dict[str, str]] = {}
        missing: list[str] = []
        for key, source_path in self.source_paths.items():
            source = str(Path(source_path).resolve()).lower()
            entry = by_source.get(source)
            required = {
                "raster", "vector", "naip", "chm_independent", "chm_interaction",
                "tree_centered_naip_chm_structure", "sentinel_phenology",
                "sentinel_sequence", "sentinel_sequence_mask", "satellite_embedding",
            }
            if entry is None or not required.issubset(entry) or not all(Path(entry[x]).exists() for x in required - {"raster", "vector"}):
                missing.append(f"{key}: {source_path}")
            else:
                result[key] = entry
        if missing:
            preview = "\n".join(missing[:10])
            raise RuntimeError(f"Missing/incomplete prepared mmap sidecars for {len(missing)} shard(s):\n{preview}")
        print(f"Prepared mmap sidecars verified: shards={len(result):,}; root={self.prepared_root}", flush=True)
        return result

    def __getstate__(self):
        state = self.__dict__.copy(); state["cache"] = {}; return state

    def get(self, source_key: str) -> MmapShardArrays:
        if source_key in self.cache:
            return self.cache[source_key]
        paths = self.sidecars[source_key]
        load = lambda name: np.load(paths[name], mmap_mode="r")
        arrays = MmapShardArrays(
            path=self.source_paths[source_key], crops=load("naip"),
            chm=load("chm_independent"), chm_interaction=load("chm_interaction"),
            structure=load("tree_centered_naip_chm_structure"),
            sentinel_phenology=load("sentinel_phenology"),
            sentinel_sequence=load("sentinel_sequence"),
            sentinel_sequence_mask=load("sentinel_sequence_mask"),
            satellite_embedding=load("satellite_embedding"),
        )
        n = len(arrays.crops)
        for name in ("chm", "chm_interaction", "structure", "sentinel_phenology", "sentinel_sequence", "sentinel_sequence_mask", "satellite_embedding"):
            if len(getattr(arrays, name)) != n:
                raise RuntimeError(f"{arrays.path}: prepared {name} rows do not match NAIP rows")
        self.cache[source_key] = arrays
        return arrays

    def clear(self) -> None:
        self.cache.clear()


class MmapTreeDataset(clean.CleanTreeDataset):
    """Build the same tensors as CleanTreeDataset from physically prepared arrays."""

    def _build_item(self, index: int) -> dict[str, Any]:
        source_key = self.source_keys[index]
        row = int(self.local_rows[index])
        arrays = self.store.get(source_key)
        transform = int(torch.randint(0, 6, ()).item()) if self.args.image_augmentation else 0

        image = torch.from_numpy(np.array(arrays.crops[row], copy=True))
        chm = torch.from_numpy(np.array(arrays.chm[row], copy=True))
        interaction = torch.from_numpy(np.array(arrays.chm_interaction[row], copy=True))
        if self.args.image_augmentation:
            image = self._apply_raster_transform(image, transform)
            chm = self._apply_raster_transform(chm, transform)
            interaction = self._apply_raster_transform(interaction, transform)
            if self.args.tree_image_channel_dropout > 0:
                keep = torch.rand(image.shape[0]) >= self.args.tree_image_channel_dropout
                if not bool(keep.any()):
                    keep[int(torch.randint(0, image.shape[0], ()).item())] = True
                image = image * keep.to(dtype=image.dtype).view(-1, 1, 1) / max(1.0 - self.args.tree_image_channel_dropout, 1.0e-6)

        item: dict[str, Any] = {
            "tree_image": image,
            "tree_chm_image": chm,
            "tree_centered_structure": torch.from_numpy(self._normalize_vector(arrays.structure[row], self.structure_mean, self.structure_std)),
            "label": torch.tensor(int(self.labels[index]), dtype=torch.long),
            "sample_weight": torch.tensor(float(self.sample_weights[index]), dtype=torch.float32),
            "city_token": str(self.manifest["city_token"].iloc[index]),
            "tree_centered_index": torch.tensor(int(self.tree_ids[index]), dtype=torch.long),
            "crop_index": torch.tensor(int(self.crop_indices[index]), dtype=torch.long),
        }
        if bool(getattr(self.args, "use_naip_chm_interaction_branch", False)):
            item["tree_chm_interaction_image"] = interaction
        if self.phenology_mean is not None:
            value = self._normalize_vector(arrays.sentinel_phenology[row], self.phenology_mean, self.phenology_std)
            zero_indices = tuple(int(x) for x in getattr(self.args, "phenology_zero_indices", ()))
            if zero_indices:
                value = value.copy(); value[list(zero_indices)] = 0.0
            item["sentinel_phenology"] = torch.from_numpy(value)
        if self.raw_sentinel_mean is not None:
            seq = np.asarray(arrays.sentinel_sequence[row], dtype=np.float32)
            seq = (seq - self.raw_sentinel_mean.reshape(1, -1)) / self.raw_sentinel_std.reshape(1, -1)
            seq = np.clip(np.nan_to_num(seq, nan=0, posinf=0, neginf=0), -12, 12).astype(np.float32)
            item["sentinel_sequence"] = torch.from_numpy(seq)
            item["sentinel_sequence_mask"] = torch.from_numpy(np.array(arrays.sentinel_sequence_mask[row], dtype=np.float32, copy=True))
        if self.satellite_embedding_mean is not None:
            value = self._normalize_vector(arrays.satellite_embedding[row], self.satellite_embedding_mean, self.satellite_embedding_std)
            if self.args.image_augmentation and self.args.satellite_embedding_value_dropout > 0:
                value = value * (np.random.random(value.shape) >= self.args.satellite_embedding_value_dropout).astype(np.float32)
            if self.args.image_augmentation and self.args.satellite_embedding_dropout > 0 and float(np.random.random()) < self.args.satellite_embedding_dropout:
                value = np.zeros_like(value, dtype=np.float32)
            item["satellite_embedding"] = torch.from_numpy(value)
        if self.prism_mean is not None:
            raise RuntimeError("Prepared mmap loader does not contain PRISM normals; this model uses daily PRISM in raw Sentinel.")
        return item
