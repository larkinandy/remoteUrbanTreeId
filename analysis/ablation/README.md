# Ablation and modality analyses

This directory reproduces the modality experiments retained for the manuscript.
All models use the frozen 11-class taxon partition in `../../config/` and the
same inherited train, validation, and independent-test record assignments. The
test set is not used for checkpoint selection, early stopping, pruning, or
modality selection.

## Frozen model specification

`final_experiment_config.json` records the architecture, optimization,
weighting, modality, and checkpoint-selection settings. Large input shards,
the prepared memory-mapped tensors, checkpoints, and the split manifest are
external data products and must be supplied explicitly on the command line.
The split manifest used for every comparison must be identical.

The complete model contains seven branches. Sequential backward elimination
removed `tree_centered_structure`; the resulting six-branch model is the final
pruned reference. Models are trained in the intermediate k=11 label space.
Probabilities from the five highest-scoring validation checkpoints are averaged
and only then collapsed to the six reported groups using
`../../config/final_k11_to_k6_collapse.npz`.

## Workflows

- `run_modality_experiments.py` trains complete-model leave-one-out or
  single-modality validation experiments.
- `run_final_pruned_ablation.py` retrains leave-one-out models relative to the
  final six-branch reference and evaluates their checkpoint ensembles.
- `run_single_modality_test_inference.py` evaluates completed single-modality
  checkpoints on the independent test set.
- `analyze_single_modality_validation.py` and
  `analyze_single_modality_test.py` create compact overall and per-group tables.
- `paired_bootstrap_metrics.py` performs a paired Poisson(1) record bootstrap
  on aligned prediction files.
- `reapply_frozen_collapse.py` deterministically rebuilds collapsed outputs
  from saved k=11 probabilities; it does not rerun inference.

Leave-one-out models measure marginal contribution conditional on the other
branches. Single-modality models measure sufficiency. Neither experiment, nor
the learned gate weights, should be interpreted as causal feature importance.

## Required inputs and example

The training runners require explicit locations for the complete shards,
prepared memory maps, split manifest, and output directory. For example:

```powershell
python analysis/ablation/run_modality_experiments.py `
  --experiment leave_one_out `
  --modality raw_sentinel `
  --complete-shard-root "$DATA_ROOT/tree_centered_complete_shards" `
  --prepared-mmap-root "$DATA_ROOT/tree_centered_prepared_float32_v1" `
  --split-manifest "$DATA_ROOT/reviewer_stage2_loader_manifest.csv.gz" `
  --output-dir "$DATA_ROOT/k11_modality_experiments"
```

Each comparison must use identical record keys (`city_token`, `tree_id`, and
`crop_index`), labels, class order, preprocessing, and checkpoint-selection
rules. Analysis scripts stop rather than compare misaligned prediction files.

## Frozen numerical results

Compact publication summaries are stored in `provenance/`. Raw predictions,
checkpoints, logs, and generated figures are intentionally excluded from Git.
The provenance README identifies the source outputs and canonical collapse.
