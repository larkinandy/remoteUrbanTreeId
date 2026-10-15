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
  prepares comparison metrics for the one-city and primary national models.
- `report_one_city_results.py` regenerates publication summaries from the
  prepared source-model table without reopening the large prediction arrays.
- `summarize_default_loco_one_city.py` produces the final 22-target-city
  comparison and paired differences among training regimes.
- `export_loco_gis_predictions.py` exports aligned k11 and collapsed-k6 labels
  and predictions for spatial analysis.
- `analyze_loco_spatial_error.py` calculates the final Rancho Cucamonga LOCO
  Moran's I statistics used in the manuscript.

## Reporting rules

Cities are eligible only when at least 1,000 independent test records remain
after excluding `qa_flag_low_vegetated_height_with_lidar`. City-level macro-F1
is calculated over groups with at least 10 QA-screened test records. Accuracy
and weighted-F1 use all QA-screened records. The thresholds, source-city list,
seed, optimization settings, architecture, checkpoint rule, composite-score
weights, and sample-weight parameters are frozen in `geographic_config.json`.

The training wrappers derive eligible target cities from the QA-screened
primary test predictions. They write a restart-safe `run_status.csv` manifest
beside the model outputs after each training or inference stage. Re-running a
wrapper skips completed artifacts unless `--overwrite` is supplied. The
wrappers pass that option directly to the shared stage-two trainer and
inference program; the deprecated `--force` spelling is not part of
the geographic-analysis interface.

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

python analysis/geographic/report_one_city_results.py `
  --source-model-summaries "$DATA_ROOT/results/seven_one_city_analysis/source_model_summaries.csv" `
  --output-dir "$DATA_ROOT/results/seven_one_city_analysis/reported_tables"
```

Small publication-facing outputs and their checksums are retained in
`provenance/`. Full prediction arrays, checkpoints, and training logs remain
external study artifacts.
