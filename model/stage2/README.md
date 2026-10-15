# Stage-two classifier

This directory reproduces the final stage-two manuscript model. The model is
trained against the frozen `k11` taxon partition in
`../../config/final_k11_partition.npz`. It does not train directly in the
`collapsed_k6` label space.

## Final training

`run_final_model.py` is the publication entry point. It reads
`final_model_config.json`, verifies the frozen grouping artifacts and inherited
split manifest, and launches `train_fixed_partition_classifier.py` with the
final six-branch architecture:

- tree-centered NAIP;
- tree-centered CHM;
- NAIP–CHM interaction;
- Sentinel-2 phenology;
- raw Sentinel-2 with daily PRISM variables; and
- satellite embeddings.

The canopy-structure branch is excluded by the final pruning decision.

Download the complete model-input shard dataset from
[Google Drive](https://drive.google.com/file/d/1w_dBZZpTgRVNfBL0-zrhhU7tJkKSsjJM/view?usp=sharing).
The optional prepared memory-mapped sidecars are not required and are not part
of this download.

Download the final pruned model bundle from
[Google Drive](https://drive.google.com/file/d/1HoV5XiQSYIoHN0NanS3hPHoOTnQjWzU-/view?usp=sharing).
The bundle includes the five validation-selected checkpoints required to
reproduce the reported ensemble, together with the run configuration and
training history.

```powershell
python model/stage2/run_final_model.py `
  --complete-shard-root "$DATA_ROOT/tree_centered_complete_sharded100k_prism_daily3_clean" `
  --prepared-mmap-root "$DATA_ROOT/tree_centered_prepared_float32_v1" `
  --split-manifest "$DATA_ROOT/reviewer_stage2_loader_manifest.csv.gz" `
  --output-dir "$DATA_ROOT/stage2_models" `
  --device cuda
```

The split manifest is distributed separately with the study data. Its SHA-256
digest and schema are recorded in
`../../analysis/ablation/provenance/split_manifest.json`. Training stops rather
than silently generating a different final-model split when this artifact does
not match.

## Ensemble evaluation and reported groups

```powershell
python model/stage2/evaluate_checkpoint_ensemble.py `
  --model-run-dir "$DATA_ROOT/stage2_models/final_pruned_k11_classifier" `
  --complete-shard-root "$DATA_ROOT/tree_centered_complete_sharded100k_prism_daily3_clean" `
  --prepared-mmap-root "$DATA_ROOT/tree_centered_prepared_float32_v1" `
  --split-manifest "$DATA_ROOT/reviewer_stage2_loader_manifest.csv.gz" `
  --output-dir "$DATA_ROOT/stage2_evaluation" `
  --split test `
  --device cuda
```

The evaluator selects the five checkpoints ranked by validation score and
averages their `k11` softmax probabilities. It then loads
`../../config/final_k11_to_k6_collapse.npz`, sums the corresponding class
probabilities, and calculates the reported `collapsed_k6` predictions. The
ensemble therefore occurs before the probability-sum collapse.

`clean_mmap_loader.py` is an optional throughput optimization. Prepared
sidecars are identified by stable shard filenames and remain usable after the
dataset is moved to another drive or workstation.

`final_input_schema.json` stores both the legacy row-count-sensitive shard hash
used by the distributed checkpoints and a portable feature-schema hash that
ignores only record-aligned row counts. This permits evaluation of city-specific
shards after relocation while still rejecting changes to field names, dtypes,
or feature dimensions. Newly trained checkpoints record both hashes directly.

Supporting modules have deliberately narrow responsibilities:

- `artifacts.py` validates frozen artifacts, split provenance, and input schema;
- `checkpointing.py` handles validation scores and portable checkpoint manifests;
- `cli.py` defines shared portable command-line argument groups;
- `clean_mmap_loader.py` implements the optional prepared-array loader; and
- `evaluation_support.py` reconstructs the inherited evaluation records.

Fully retrained ablation experiments are in `../../analysis/ablation/`.
