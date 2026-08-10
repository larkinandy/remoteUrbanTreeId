# Current model-training pipeline

The active training sequence is:

1. Train `train_clean_tree_id_centered_taxon_discriminator.py` with
   `--export-embeddings` enabled. This writes `val_embeddings.npz` and
   `test_embeddings.npz`.
2. Run `build_oracle_centroid_taxon_partitions.py` to compare k=5 through k=12
   collapsed taxon partitions and write the `kNN_partition.npz` files.
3. Train `train_clean_tree_id_centered_k6_classifier.py` with the selected
   `k06_partition.npz`.


<img src = https://github.com/larkinandy/remoteUrbanTreeId/blob/main/images/Discriminator.png width="1024">

<img src = https://github.com/larkinandy/remoteUrbanTreeId/blob/main/images/Architecture.png width="1024">


## Rebuild the k=5--12 Oracle experiment

Install scikit-learn in the model environment, then run:

```powershell
python model/build_oracle_centroid_taxon_partitions.py `
  --run-dir H:/TreeCenteredModelInputs/taxon_discrimination_clean/clean_abq_atl_taxon_discriminator `
  --output-dir H:/TreeCenteredModelInputs/taxon_discrimination_clean/clean_abq_atl_taxon_discriminator/rebuilt_global_centroid_partitions `
  --min-groups 5 `
  --max-groups 12
```

Review `partition_summary.csv`, `k06_group_composition.csv`, and `summary.json`
before using the rebuilt `k06_partition.npz` for classifier training.
