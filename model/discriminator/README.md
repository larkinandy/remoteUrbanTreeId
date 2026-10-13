# Taxon discriminator and group construction

This directory contains the complete first-stage workflow used to define the
remote-sensing-derived taxon groups. Group construction uses validation data
only. Independent test embeddings are evaluated only after every mapping has
been frozen.

Download the final discriminator model bundle from
[Google Drive](https://drive.google.com/file/d/1yDKvUhOKYt_-wgEWzVxq-gO6Azwr9rrj/view?usp=sharing).
The bundle contains the frozen model weights, run configuration, training
history, and validation embeddings used for validation-only group
construction.

The corresponding model-input shard dataset is available separately from
[Google Drive](https://drive.google.com/file/d/1w_dBZZpTgRVNfBL0-zrhhU7tJkKSsjJM/view?usp=sharing).

## Script roles

| Script | Role | Status |
| --- | --- | --- |
| `train_taxon_discriminator.py` | Fit the multimodal discriminator and export validation embeddings | Final training |
| `select_group_count.py` | Compare initial candidate values of k without test data | Exploratory validation |
| `bootstrap_fixed_k_consensus.py` | Freeze core assignments at a preselected k | Final group construction |
| `select_collapsed_group_count.py` | Apply the prespecified stopping rule to nested candidates | Final validation |
| `export_rare_taxon_embeddings.py` | Embed taxa with 50–499 records using the frozen encoder | Final inference |
| `extend_partition_with_rare_taxa.py` | Assign rare taxa from development-only centroids | Final group construction |
| `evaluate_independent_test.py` | Evaluate mappings after all decisions are frozen | Independent evaluation |

`script_manifest.csv` provides the same classification in machine-readable
form. Exploratory results are not independent evidence, and independent test
results must not be used to revise group definitions.

## Output locations

Generated embeddings, checkpoints, bootstrap matrices, predictions, and result
tables are kept outside this tracked source directory. The scripts reject
output paths inside `model/discriminator`. Use `$DATA_ROOT` or an ignored
analysis-output directory. Only compact frozen mappings and their checksums
belong under `config/`.

## Reproducible order of operations

1. **Train the discriminator.** Use the frozen architecture and optimization
   settings in `final_discriminator_config.json`. Machine-specific data and
   output paths remain command-line arguments.

   ```bash
   python model/discriminator/train_taxon_discriminator.py \
     --config model/discriminator/final_discriminator_config.json \
     --complete-shard-root "$DATA_ROOT/tree_centered_complete_sharded100k_clean" \
     --output-dir "$DATA_ROOT/taxon_discrimination"
   ```

   The final run uses taxa with at least 500 records, group normalization for
   the NAIP, CHM, and NAIP–CHM interaction encoders, a 64-dimensional embedding,
   and no test evaluation during training.

2. **Select the initial group count from validation embeddings.**
   `select_group_count.py` compares validation-centroid partitions and cannot
   accept test embeddings.

3. **Freeze the `k11` core-taxon assignments.** Run
   `bootstrap_fixed_k_consensus.py` with 500 within-taxon bootstrap replicates.
   The frozen manuscript artifact is `config/core_k11_partition.npz`.

4. **Evaluate nested collapsed candidates.** Variance-aware complete-linkage
   candidates enforce maximum record and core-taxon shares of 0.5. Supply their
   validation-only diagnostics to `select_collapsed_group_count.py`, which
   applies the prespecified recoverability/separation stopping rule for
   lambda = 0.5, 1, and 2. The final `k11`-to-`collapsed_k6` mapping is frozen in
   `config/final_k11_to_k6_collapse.npz`.

5. **Embed and assign rare taxa.** `export_rare_taxon_embeddings.py` passes taxa
   with 50–499 records through the frozen discriminator and creates a
   deterministic within-taxon development/test split. Then
   `extend_partition_with_rare_taxa.py` constructs each rare-taxon centroid from
   development records only and assigns it to the nearest frozen `k11` group
   prototype. Its artifact follows the exact `config/final_k11_partition.npz`
   schema. No rare-taxon test record contributes to its assigned label.

6. **Evaluate the independent test set.** Run
   `evaluate_independent_test.py` only after all partitions and collapse mappings
   are frozen. It audits record-key overlap and records source hashes.

## Frozen artifacts

The authoritative mappings and their hashes are documented by
`config/final_grouping.json`, `config/artifact_manifest.csv`, and
`config/validate_config.py`. Group identifiers are zero-based in NPZ arrays and
one-based in manuscript displays.

## Reproducibility safeguards

- Selection scripts accept validation embeddings only.
- NPZ schemas and taxon ordering are checked before calculations.
- Rare assignments use development records only.
- Independent evaluation audits validation/test record-key overlap.
- JSON audits record hashes and whether test data were loaded.
- Seeds and final architecture settings are frozen in
  `final_discriminator_config.json`.
