# Geographic generalization analyses

This directory reproduces the manuscript's geographic sensitivity analyses for
the finalized stage-two classifier. All models predict the intermediate `k11`
label space. The top five validation checkpoints are ensembled before their
probabilities are summed into `collapsed_k6` using
`../../config/final_grouping.json`.

## Experiments

- `run_leave_one_city_out.py` removes the focal city from training and
  validation and evaluates the ensemble on that city's inherited test records.
- `run_seven_one_city_models.py` trains one model for each source city listed in
  `geographic_config.json`, evaluates it within its source city, and evaluates
  transfer to the other eligible cities.
- `analyze_seven_one_city_results.py` performs exact record/label audits and
  compares the one-city models with the primary national model.
- `select_median_city.py` records the earlier, primary-model-only selection of a
  representative city. Geographic sensitivity results are not used for that
  selection.
- `compare_training_regimes.py` is the earlier three-regime St. Louis diagnostic
  retained for provenance; the seven-source analysis is the final comparison.

## Reporting rules

Cities are eligible only when at least 1,000 independent test records remain
after excluding `qa_flag_low_vegetated_height_with_lidar`. City-level macro-F1
is calculated over groups with at least 10 QA-screened test records. Accuracy
and weighted-F1 use all QA-screened records. The thresholds and source-city list
are frozen in `geographic_config.json`.

The split manifest is intentionally supplied as an explicit argument because
the record-level manifest is a generated dataset artifact rather than a small
repository file. It is created by the stage-two partition workflow and must be
the same manifest used by the primary model. Both training wrappers verify it
against the frozen SHA-256 checksum; the expected record count is also
documented in `geographic_config.json`.

## Portable examples

```powershell
python analysis/geographic/run_leave_one_city_out.py `
  --complete-shard-root "$DATA_ROOT/tree_centered_complete_shards" `
  --prepared-mmap-root "$DATA_ROOT/tree_centered_prepared_float32_v1" `
  --split-manifest "$DATA_ROOT/reviewer_stage2_loader_manifest.csv.gz" `
  --reference-test-predictions "$DATA_ROOT/final_model/ensemble_predictions.npz" `
  --qa-metadata "$DATA_ROOT/final_test_sensitivity_variables.csv.gz" `
  --model-output-dir "$DATA_ROOT/models/k11_loco_pruned_v1"

python analysis/geographic/run_seven_one_city_models.py `
  --complete-shard-root "$DATA_ROOT/tree_centered_complete_shards" `
  --split-manifest "$DATA_ROOT/reviewer_stage2_loader_manifest.csv.gz" `
  --reference-test-predictions "$DATA_ROOT/final_model/ensemble_predictions.npz" `
  --qa-metadata "$DATA_ROOT/final_test_sensitivity_variables.csv.gz" `
  --output-root "$DATA_ROOT/models/k11_one_city_seven_source"

python analysis/geographic/analyze_seven_one_city_results.py `
  --one-city-root "$DATA_ROOT/models/k11_one_city_seven_source" `
  --primary-predictions "$DATA_ROOT/final_model/ensemble_predictions.npz" `
  --qa-metadata "$DATA_ROOT/final_test_sensitivity_variables.csv.gz"
```

Small publication-facing outputs and their checksums are retained in
`provenance/`. Full prediction arrays, checkpoints, and training logs remain
external study artifacts.
