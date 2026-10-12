# Sentinel-2 data collection and phenology

This folder contains the Sentinel-2 workflow used by the study. Crown
coordinates are assigned to the containing 10 m Sentinel-2 grid cell; valid
observations are exported from Google Earth Engine (GEE), converted to regular
time series with explicit quality fields, and summarized into phenology
features. Sentinel-2 therefore supplies cell-scale phenological context rather
than an isolated measurement of an individual crown.

## Files

| File | Role |
|---|---|
| `reduce_tree_points_to_sentinel_cells.py` | Create unique occupied cells and tree-to-cell linkage tables. |
| `export_sentinel2_time_series_gee.py` | Export cloud-masked Sentinel-2 L2A observations from GEE. |
| `derive_tree_centered_sentinel_phenology.py` | Reconcile raw exports and construct regular cell time series and intermediate phenology products. |
| `derive_clean_tree_id_centered_sentinel_phenology_sidecars.py` | Create the final clean `tree_id`-keyed sidecars used by model-shard assembly. |
| `interpolate_sentinel_cell_time_series.py` | Shared interpolation implementation. |
| `sentinel_cell_features.py` | Shared reflectance-band, spectral-index, masking, and aggregation logic. |
| `sentinel_phenology_metrics.py` | Shared phenology feature definitions and calculations. |
| `sentinel_schema.py` | Canonical sequence schema shared with model-shard assembly. |
| `sentinel_common.py` | Shared city-token, date-window, and file-discovery utilities. |
| `sentinel_export_inventory.py` | Raw-export inventory and batch-completeness checks. |
| `freeze_study_sentinel2_provenance.py` | Freeze path-independent sidecar summaries. |
| `environment-sentinel2.yml` | Reproducible preprocessing environment. |
| `provenance/` | Frozen study configuration and output summaries. |

The two phenology entry points are sequentially distinct, not competing final
products. The general pipeline creates or reconciles regular per-cell time
series. The clean-sidecar builder links those time series to the final clean
crown-centered records and writes the sidecars consumed by shard assembly.

## Environment

```powershell
conda env create --file dataCollectionPreprocessing/Sentinel2/environment-sentinel2.yml
conda activate treeid-sentinel2
```

Create `dataCollectionPreprocessing/Sentinel2/.env` containing the GEE project
identifier if it will not be passed on the command line:

```text
GEE_PROJECT=your-earth-engine-project
```

The `.env` file and Earth Engine credentials are not committed. Use
`--authenticate` on the first GEE run if needed.

## Study configuration

The authoritative path-independent parameters are recorded in
`provenance/study_sentinel2_configuration.json`. The study used harmonized
Sentinel-2 surface reflectance and the Sentinel-2 cloud-probability collection
from 2021-01-01 through 2023-12-31. Cloud-probability and SCL masks were
applied before 10 m point sampling. Linear interpolation generated regular
15-day rows while retaining fields that distinguish observed from interpolated
values.

## Workflow

Define a portable local data root:

```powershell
$DATA_ROOT = "path/to/urban_tree_data"
```

### 1. Convert crown coordinates to occupied Sentinel cells

```powershell
python dataCollectionPreprocessing/Sentinel2/reduce_tree_points_to_sentinel_cells.py `
  "$DATA_ROOT/tree_records/denver_crowns.csv" `
  --tree-id-field tree_id `
  --longitude-field longitude `
  --latitude-field latitude `
  --target-crs EPSG:32613 `
  --output-dir "$DATA_ROOT/sentinel2/cells/denver"
```

The outputs are unique cell-center points, a tree-to-cell table, and an audit
table for pixels shared by multiple trees. Preserve them because `reduced_id`
is the raw-export join key.

### 2. Export valid observations from GEE

```powershell
python dataCollectionPreprocessing/Sentinel2/export_sentinel2_time_series_gee.py `
  --points-file "$DATA_ROOT/sentinel2/cells/denver/sentinel10m_unique_cells.shp" `
  --city Denver `
  --project your-earth-engine-project `
  --start-date 2021-01-01 `
  --end-date 2024-01-01 `
  --interval-days 15 `
  --batch-size 5000 `
  --drive-folder TREE_SENTINEL2_RAW `
  --completed-dir "$DATA_ROOT/sentinel2/raw/denver" `
  --dry-run
```

Remove `--dry-run` after reviewing the task plan. The GEE end date is
exclusive. Exported reflectance bands are B2, B3, B4, B5, B6, B7, B8, B8A,
B11, and B12; SCL is retained for QA.

### 3. Construct regular cell time series

```powershell
python dataCollectionPreprocessing/Sentinel2/derive_tree_centered_sentinel_phenology.py `
  --record-index-root "$DATA_ROOT/record_index" `
  --original-timeseries-root "$DATA_ROOT/sentinel2/original_timeseries" `
  --supplemental-timeseries-root "$DATA_ROOT/sentinel2/supplemental_timeseries" `
  --original-raw-sentinel-root "$DATA_ROOT/sentinel2/raw_original" `
  --additional-raw-sentinel-root "$DATA_ROOT/sentinel2/raw_supplemental" `
  --missing-raw-cell-root "$DATA_ROOT/sentinel2/missing_cells" `
  --timeseries-output-root "$DATA_ROOT/sentinel2/regular_timeseries" `
  --output-root "$DATA_ROOT/sentinel2/intermediate_phenology" `
  --stage all `
  --dry-run
```

Review the plan, then remove `--dry-run`. Use `--overwrite` only when replacing
existing products intentionally.

### 4. Create the final clean sidecars

```powershell
python dataCollectionPreprocessing/Sentinel2/derive_clean_tree_id_centered_sentinel_phenology_sidecars.py `
  --crop-root "$DATA_ROOT/tree_centered_naip_crops_clean" `
  --original-timeseries-root "$DATA_ROOT/sentinel2/original_timeseries" `
  --supplemental-timeseries-root "$DATA_ROOT/sentinel2/regular_timeseries" `
  --output-dir "$DATA_ROOT/tree_centered_sentinel_phenology_clean"
```

This stage uses the detected crown-center coordinate by default, calculates 95
phenology variables for each matched cell, and writes final sidecars keyed by
`tree_id`. Saved `config_json` values contain processing parameters and
portable filenames, not workstation paths.

### 5. Freeze output provenance

```powershell
python dataCollectionPreprocessing/Sentinel2/freeze_study_sentinel2_provenance.py `
  --sidecar-root "$DATA_ROOT/tree_centered_sentinel_phenology_clean" `
  --output-dir dataCollectionPreprocessing/Sentinel2/provenance
```

## QA checklist

1. Confirm collections, dates, masks, thresholds, and scale against the frozen configuration.
2. Confirm every export row retains its stable `row_index`/`reduced_id` linkage.
3. Review raw-export completeness before interpolation.
4. Confirm observed and interpolated rows remain distinguishable.
5. Review city-level missingness and source counts in frozen provenance.
6. Confirm all final sidecars use the same ordered 95-feature schema.
7. Preserve raw exports and cell-link tables until final shards are independently verified.
