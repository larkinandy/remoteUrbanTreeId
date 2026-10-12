# Taxon discriminator and group construction

This directory contains the complete first-stage workflow used to define the
remote-sensing-derived taxon groups. Group construction uses validation data
only. Independent test embeddings are evaluated only after every mapping has
been frozen.

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

3. **Freeze the k = 11 core-taxon assignments.** Run
   `bootstrap_fixed_k_consensus.py` with 500 within-taxon bootstrap replicates.
   The frozen manuscript artifact is `config/core_k11_partition.npz`.

4. **Evaluate nested collapsed candidates.** Variance-aware complete-linkage
   candidates enforce maximum record and core-taxon shares of 0.5. Supply their
   validation-only diagnostics to `select_collapsed_group_count.py`, which
   applies the prespecified recoverability/separation stopping rule for
   lambda = 0.5, 1, and 2. The final k11-to-collapsed-k6 mapping is frozen in
   `config/final_k11_to_k6_collapse.npz`.

5. **Embed and assign rare taxa.** `export_rare_taxon_embeddings.py` passes taxa
   with 50–499 records through the frozen discriminator and creates a
   deterministic within-taxon development/test split. Then
   `extend_partition_with_rare_taxa.py` constructs each rare-taxon centroid from
   development records only and assigns it to the nearest frozen k = 11 group
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
