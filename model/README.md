# Manuscript model pipeline

This directory contains the two modeling stages used for the manuscript. The
checked-in partition and collapse configuration are frozen artifacts, not
parameters selected from the independent test set.

## Directory layout

<img src = https://github.com/larkinandy/remoteUrbanTreeId/blob/main/images/Discriminator.png width="1024">

<img src = https://github.com/larkinandy/remoteUrbanTreeId/blob/main/images/Architecture.png width="1024">

- `discriminator/` trains the fine-taxon discriminator, constructs the fixed
  intermediate groups from validation embeddings, extends the partition to
  taxa with 50–499 records, and evaluates the frozen mapping independently.
- `stage2/` trains the fixed-partition classifier and evaluates checkpoint
  ensembles.
- `model_components.py` contains shared PyTorch encoders and runtime helpers.
- `../config/final_k11_partition.npz` is the frozen taxon-to-intermediate-group
  mapping used by the manuscript model.
- `../config/final_grouping.json` records the fixed 11-to-6 collapse and the
  partition checksum.

Generated checkpoints, embeddings, prediction exports, and input shards are
not stored in Git.

## Stage 1: taxon discriminator

Train the discriminator from the complete model-input shards. GroupNorm is
used for the raster encoders and the NAIP--CHM interaction is an independent
modality. Validation embeddings are exported for label construction; test
evaluation should remain disabled until the mapping is frozen.

```powershell
python model/discriminator/train_taxon_discriminator.py `
  --complete-shard-root H:/TreeCenteredModelInputs/tree_centered_complete_sharded100k_prism_daily3_clean `
  --output-dir H:/TreeCenteredModelInputs/taxon_discrimination_clean `
  --run-name manuscript_discriminator `
  --min-class-train-samples 500 `
  --naip-encoder-normalization group `
  --chm-encoder-normalization group `
  --use-naip-chm-interaction-branch `
  --interaction-encoder-normalization group `
  --export-validation-embeddings `
  --no-evaluate-test
```

Use `select_group_count.py` on validation embeddings to compare candidate
values of `k`, then freeze the selected `k=11` assignments with 500 within-
taxon bootstrap replicates:

```powershell
python model/discriminator/bootstrap_fixed_k_consensus.py `
  --validation H:/path/to/manuscript_discriminator/val_embeddings.npz `
  --output-dir H:/path/to/fixed_k11_consensus `
  --groups 11 `
  --replicates 500
```

`extend_partition_with_rare_taxa.py` assigns taxa with 50--499 records using
development-only centroids. It never defines a taxon's label from that record's
own held-out embedding. The resulting mapping is the small artifact committed
as `config/final_k11_partition.npz`.

After all assignments are frozen, use `evaluate_independent_test.py` to report
nearest-centroid performance on the discriminator test embeddings.

## Stage 2: fixed-partition classifier

The stage-two model predicts the 11 frozen groups. Its final six-branch
configuration contains tree-centered NAIP, tree-centered CHM, the NAIP--CHM
interaction, Sentinel-2 phenology, raw Sentinel-2 with PRISM variables, and
satellite embeddings.

```powershell
python model/stage2/run_final_model.py `
  --complete-shard-root "$DATA_ROOT/tree_centered_complete_sharded100k_prism_daily3_clean" `
  --prepared-mmap-root "$DATA_ROOT/tree_centered_prepared_float32_v1" `
  --split-manifest "$DATA_ROOT/reviewer_stage2_loader_manifest.csv.gz" `
  --output-dir "$DATA_ROOT/stage2_models" `
  --device cuda
```

`stage2/final_model_config.json` freezes the architecture, optimization,
checkpoint-selection, and pruning settings used for this command. The launcher
validates the checked-in grouping artifacts before starting training.

The exact split manifest must be reused for all final, ablation, and
single-modality models. Test records must not be used for checkpoint selection
or early stopping.

## Ensemble and six-group reporting

The evaluator averages the 11-class softmax probabilities from the five
highest-scoring validation checkpoints. It then applies the fixed mapping in
`config/final_grouping.json` by summing probabilities before selecting the
collapsed prediction.

```powershell
python model/stage2/evaluate_checkpoint_ensemble.py `
  --model-run-dir "$DATA_ROOT/stage2_models/final_pruned_k11_classifier" `
  --complete-shard-root "$DATA_ROOT/tree_centered_complete_sharded100k_prism_daily3_clean" `
  --prepared-mmap-root "$DATA_ROOT/tree_centered_prepared_float32_v1" `
  --split-manifest "$DATA_ROOT/reviewer_stage2_loader_manifest.csv.gz" `
  --output-dir "$DATA_ROOT/stage2_evaluation" `
  --top-k 5 `
  --split test `
  --device cuda
```

The evaluator loads the checksum-validated collapse from
`config/final_k11_to_k6_collapse.npz`; the mapping is not entered manually.

The ensemble is formed before the probability-sum collapse. The six reported
groups are therefore not predictions from a separately trained six-class
network.

## Manuscript analyses

The scripts in `../analysis/` reproduce the retained secondary analyses:

- leave-one-modality-out and single-modality experiments;
- paired record-bootstrap comparisons and sequential pruning summaries;
- leave-one-city-out and one-city-trained geographic transfer;
- data-quality, mixed-pixel, and gate-weight stratification;
- manuscript figures based on saved numerical outputs.

Development-only pilots, benchmarks, and smoke tests are intentionally omitted
from the public manuscript snapshot.
