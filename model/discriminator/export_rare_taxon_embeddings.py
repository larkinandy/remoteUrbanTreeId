#!/usr/bin/env python3
"""Export frozen-encoder embeddings for taxa with 50–499 records.

The checkpoint supplies the trained encoder and training-derived standardizers.
Records are split deterministically within taxon into development and held-out
test subsets. Held-out records are never used to create taxon centroids.
"""

from __future__ import annotations

import argparse
import json
from argparse import Namespace
from pathlib import Path

import numpy as np

import train_taxon_discriminator as trainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--complete-shard-root", type=Path, required=True)
    parser.add_argument("--complete-shard-pattern", default="*_part*_tree_centered_complete_inputs.npz")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-records", type=int, default=50)
    parser.add_argument("--maximum-records", type=int, default=499)
    parser.add_argument("--development-fraction", type=float, default=0.8)
    parser.add_argument("--batch-size", type=int, default=768)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-shard-cache", type=int, default=12)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def saved_array(checkpoint: dict, name: str) -> np.ndarray | None:
    value = checkpoint.get(name)
    if value is None:
        return None
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.float32)


def build_model(args: Namespace, checkpoint: dict, arrays: dict[str, np.ndarray | None], device):
    model = trainer.CleanTreeDiscriminator(
        class_count=len(checkpoint["label_to_index"]),
        structure_dim=int(arrays["structure_mean"].shape[0]),
        hidden_dim=args.hidden_dim,
        embedding_dim=args.embedding_dim,
        dropout=args.dropout,
        phenology_dim=0 if arrays["phenology_mean"] is None else int(arrays["phenology_mean"].shape[0]),
        raw_sentinel_dim=0 if arrays["raw_sentinel_mean"] is None else int(arrays["raw_sentinel_mean"].shape[0]),
        satellite_embedding_dim=0 if arrays["satellite_embedding_mean"] is None else int(arrays["satellite_embedding_mean"].shape[0]),
        prism_normals_dim=0 if arrays["prism_mean"] is None else int(arrays["prism_mean"].shape[0]),
        tree_image_branch_dropout=args.tree_image_branch_dropout,
        naip_encoder_normalization=args.naip_encoder_normalization,
        chm_encoder_normalization=args.chm_encoder_normalization,
        use_naip_chm_interaction_branch=args.use_naip_chm_interaction_branch,
        interaction_encoder_normalization=args.interaction_encoder_normalization,
        interaction_naip_patch_pixels=args.interaction_naip_patch_pixels,
        interaction_chm_patch_pixels=args.interaction_chm_patch_pixels,
        independent_naip_crop_pixels=args.image_center_crop_pixels,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model


def main() -> int:
    cli = parse_args()
    if cli.output.exists() and not cli.overwrite:
        raise SystemExit(f"Output exists; pass --overwrite to replace it: {cli.output}")
    device = trainer.train_base.torch.device(cli.device)
    checkpoint = trainer.train_base.torch.load(cli.checkpoint, map_location=device)
    saved = dict(checkpoint["args"])
    saved.update(
        complete_shard_root=cli.complete_shard_root,
        complete_shard_pattern=cli.complete_shard_pattern,
        city_token=None,
        exclude_city_token=[],
        max_shard_cache=cli.max_shard_cache,
        batch_size=cli.batch_size,
        num_workers=cli.num_workers,
        eval_num_workers=cli.num_workers,
        persistent_workers=False,
        cache_val_in_memory=False,
        cache_test_in_memory=False,
        image_augmentation=False,
        device=cli.device,
        seed=cli.seed,
    )
    saved["complete_shard_root"] = Path(saved["complete_shard_root"])
    args = Namespace(**saved)

    manifest, paths = trainer.discover_shards(args)
    manifest["discriminator_label"] = trainer.build_target_labels(manifest, args)
    manifest = manifest.loc[manifest["discriminator_label"].ne("")].copy()
    counts = manifest["discriminator_label"].value_counts()
    rare_taxa = sorted(counts[(counts >= cli.minimum_records) & (counts <= cli.maximum_records)].index)
    manifest = manifest.loc[manifest["discriminator_label"].isin(rare_taxa)].copy()
    if manifest.empty:
        raise SystemExit("No records satisfy the requested rare-taxon support interval")
    manifest["label_index"] = 0
    manifest["sample_weight"] = 1.0
    manifest["rare_split"] = "test"
    rng = np.random.default_rng(cli.seed)
    for indices in manifest.groupby("discriminator_label", sort=True).groups.values():
        indices = np.asarray(list(indices), dtype=np.int64)
        rng.shuffle(indices)
        development_count = max(1, min(len(indices) - 1, round(len(indices) * cli.development_fraction)))
        manifest.loc[indices[:development_count], "rare_split"] = "development"
    manifest = manifest.reset_index(drop=True)

    store = trainer.CleanShardStore(paths, max_cached_shards=cli.max_shard_cache)
    names = (
        "structure_mean", "structure_std", "phenology_mean", "phenology_std",
        "raw_sentinel_mean", "raw_sentinel_std", "satellite_embedding_mean",
        "satellite_embedding_std", "prism_mean", "prism_std",
    )
    arrays = {name: saved_array(checkpoint, name) for name in names}
    dataset = trainer.CleanTreeDataset(
        manifest, store, arrays["structure_mean"], arrays["structure_std"],
        arrays["phenology_mean"], arrays["phenology_std"], arrays["raw_sentinel_mean"],
        arrays["raw_sentinel_std"], arrays["satellite_embedding_mean"],
        arrays["satellite_embedding_std"], arrays["prism_mean"], arrays["prism_std"], args,
    )
    loader = trainer.make_loader(dataset, args, train=False)
    metrics = trainer.evaluate(build_model(args, checkpoint, arrays, device), loader, device, len(checkpoint["label_to_index"]), collect=True)
    cli.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cli.output,
        embeddings=metrics["embeddings"],
        taxon=manifest["discriminator_label"].to_numpy(dtype=str),
        rare_split=manifest["rare_split"].to_numpy(dtype=str),
        city_token=manifest["city_token"].to_numpy(dtype=str),
        tree_id=manifest["tree_id"].to_numpy(dtype=np.int64),
        crop_index=manifest["crop_index"].to_numpy(dtype=np.int64),
    )
    audit = {
        "records": int(len(manifest)), "taxa": int(len(rare_taxa)),
        "development_records": int(manifest["rare_split"].eq("development").sum()),
        "held_out_test_records": int(manifest["rare_split"].eq("test").sum()),
        "support_interval": [cli.minimum_records, cli.maximum_records],
        "checkpoint": str(cli.checkpoint), "output": str(cli.output),
    }
    cli.output.with_suffix(".audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
