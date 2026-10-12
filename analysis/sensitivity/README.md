# Sensitivity analyses

This directory reproduces post-hoc sensitivity analyses of the finalized
stage-two ensemble. The model predicts 11 intermediate groups (`k11`); the five
selected checkpoints are ensembled before probabilities are summed into the
six reported groups (`collapsed_k6`) using `../../config/final_grouping.json`.

## Final analyses

- `analyze_final_sensitivity_table.py` produces the complete independent-test
  sensitivity table for vegetation-height QA, crown-match distance, duplicate
  coordinates, taxon provenance, within-cell NAIP vegetation, Sentinel-2
  interpolation, and LiDAR and NAIP temporal mismatch. The vegetation-height
  contrast requires retaining all test records; all other group-performance
  and gate analyses exclude the finalized height warning when stated.
- `analyze_final_test_predictions.py` produces QA-screened performance by city
  and taxon provenance and the `collapsed_k6` confusion counts.
- `analyze_chm_mixed_pixel_strata.py` evaluates the supplementary CHM-supported
  neighborhood vegetation proxy. It is not a focal-crown coverage estimate.
- `analyze_gate_weight_strata.py` summarizes sample-dependent gate weights by
  height, city, city-by-height, and taxon provenance. Gate weights are not
  causal feature importance.

Thresholds, terminology, provenance definitions, and reporting rules are
frozen in `sensitivity_config.json`. Every script uses record-key joins and
rejects duplicate keys. The final table additionally verifies that stored
`collapsed_k6` labels and probabilities reproduce the canonical probability
sum from `k11`.

## Complete sensitivity table

```powershell
python analysis/sensitivity/analyze_final_sensitivity_table.py `
  --predictions "$DATA_ROOT/final_model/ensemble_predictions.npz" `
  --sensitivity-variables "$DATA_ROOT/final_test_sensitivity_variables.csv.gz" `
  --mixed-pixel-proxies "$DATA_ROOT/record_level_reportable_mixed_pixel_proxies.csv.gz" `
  --filter-variables "$DATA_ROOT/record_level_filter_variables.csv.gz" `
  --naip-year-variables "$DATA_ROOT/test_record_naip_inventory_years.csv.gz" `
  --complete-shard-root "$DATA_ROOT/tree_centered_complete_shards" `
  --output-dir "$DATA_ROOT/results/final_sensitivity_table"
```

The record-level Sentinel-2 interpolation fraction is read directly from the
complete shards and joined by `record_key`; positional arrays are not used for
the sensitivity analysis.

## Other analyses

```powershell
python analysis/sensitivity/analyze_final_test_predictions.py `
  --predictions "$DATA_ROOT/final_model/ensemble_predictions.npz" `
  --sensitivity-variables "$DATA_ROOT/final_test_sensitivity_variables.csv.gz" `
  --output-dir "$DATA_ROOT/results/test_performance"

python analysis/sensitivity/analyze_gate_weight_strata.py `
  --predictions "$DATA_ROOT/final_model/ensemble_predictions.npz" `
  --sensitivity-variables "$DATA_ROOT/final_test_sensitivity_variables.csv.gz" `
  --shard-root "$DATA_ROOT/tree_centered_complete_shards" `
  --output-dir "$DATA_ROOT/results/gate_weight_strata"
```

Add `--plot` when the environment has a functioning Matplotlib installation;
the statistical tables and keyed height cache do not depend on figure output.

Frozen publication-facing values and source checksums are retained in
`provenance/`. Large predictions, record-level tables, and shard files remain
external study artifacts.
