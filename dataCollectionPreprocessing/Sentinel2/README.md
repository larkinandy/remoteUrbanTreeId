# Sentinel-2 data collection

This folder contains only the reusable steps needed to create Sentinel-2
products for a new dataset with tree-crown coordinates. Crown locations are
reduced to unique occupied 10 m cells so trees in the same Sentinel-2 pixel are
downloaded once and joined back to individual crowns later.

## Files to keep

| File | Role |
| --- | --- |
| `reduce_tree_points_to_sentinel_cells.py` | Convert crown coordinates into unique occupied 10 m cell-center points and a tree-to-cell join table. |
| `export_sentinel2_time_series_gee.py` | Download cloud-masked Sentinel-2 L2A observations from Google Earth Engine (GEE). |
| `derive_tree_centered_sentinel_phenology.py` | Interpolate the downloaded observations and calculate the current model's phenology features locally. |
| `derive_clean_tree_id_centered_sentinel_phenology_sidecars.py` | Rebuild clean `tree_id`-keyed phenology sidecars from existing per-city Sentinel time-series CSVs. |
| `sentinel_phenology_metrics.py` | Shared seasonal feature names and calculations used by both phenology sidecar builders. |
| `sentinel_cell_features.py` | Shared band definitions, spectral-index calculations, export normalization, invalid-pixel filtering, and daily cell aggregation used by local phenology preprocessing. |
| `interpolate_sentinel_cell_time_series.py` | Preserve observed daily values and insert linearly interpolated Sentinel rows at regular intervals across observation gaps. |

GEE is used only to download cloud-masked observations. Phenology is calculated
locally from those observations. `sentinel_cell_features.py` and
`interpolate_sentinel_cell_time_series.py` are helper modules used by the
phenology workflow and are not normally run directly. The metrics module is
also an imported helper rather than a standalone command.

## Setup

Install the required packages:

```powershell
python -m pip install earthengine-api geopandas numpy pandas pyogrio python-dotenv
```

Create `dataCollectionPreprocessing/Sentinel2/.env` containing:

```text
GEE_PROJECT=your-earth-engine-project
```

The `.env` file is ignored by Git. Use `--authenticate` on the first GEE run
if Earth Engine credentials are not already configured.

## 1. Convert crown coordinates to Sentinel-2 cells

The input can be a point shapefile or a CSV with longitude and latitude. Use a
projected CRS in meters appropriate for the city, normally its local UTM zone.

```powershell
python dataCollectionPreprocessing/Sentinel2/reduce_tree_points_to_sentinel_cells.py `
  D:/tree_data/denver_crowns.csv `
  --tree-id-field tree_id `
  --longitude-field longitude `
  --latitude-field latitude `
  --target-crs EPSG:32613 `
  --output-dir D:/sentinel2/denver/cells
```

The outputs are:

- `sentinel10m_unique_cells.shp`: one sampling point per occupied 10 m cell
- `tree_to_sentinel10m_cell.csv`: mapping from every crown to `reduced_id`
- `sentinel10m_multi_tree_cells.csv`: audit of cells shared by multiple crowns

Keep these outputs unchanged after downloading data. Regenerating them can
change `reduced_id`, which is the key used to join products back to crowns.

## 2. Download the raw Sentinel-2 time series

First run a short test with `--dry-run`:

```powershell
python dataCollectionPreprocessing/Sentinel2/export_sentinel2_time_series_gee.py `
  --points-file D:/sentinel2/denver/cells/sentinel10m_unique_cells.shp `
  --city Denver `
  --project your-earth-engine-project `
  --start-date 2021-06-01 `
  --end-date 2021-07-01 `
  --interval-days 15 `
  --batch-size 5000 `
  --drive-folder TREE_SENTINEL2_RAW `
  --completed-dir D:/sentinel2/denver/raw `
  --dry-run
```

Remove `--dry-run` to submit the exports. The script exports bands `B2`, `B3`,
`B4`, `B5`, `B6`, `B7`, `B8`, `B8A`, `B11`, and `B12`. It masks cloud,
shadow, cirrus, snow, no-data, and saturated pixels before sampling.

To backfill a CSV containing only missing cells, pass that CSV directly with
`--points-file`. Use `--row-index-property` if its ID column is not
`reduced_id`; no separate backfill downloader is needed.

## 3. Create the current phenology product

After downloading the GEE CSV files, interpolate the raw observations to the
regular time series and calculate phenology locally. The record-index directory
must contain the crown-centered record index generated for the dataset; its
rows link each crown record to the downloaded Sentinel cell IDs.

```powershell
python dataCollectionPreprocessing/Sentinel2/derive_tree_centered_sentinel_phenology.py `
  --record-index-root D:/sentinel2/record_index `
  --original-raw-sentinel-dir D:/sentinel2/denver/raw `
  --additional-raw-sentinel-dir D:/sentinel2/denver/raw `
  --timeseries-output-root D:/sentinel2/timeseries `
  --output-root D:/sentinel2/phenology `
  --city-token denver `
  --stage all `
  --dry-run
```

Review the discovered files and planned work, then remove `--dry-run`. Stage
`interpolate` creates regular 15-day time-series CSVs; stage `compute` converts
those time series into phenology sidecars. Stage `all` performs both steps.

The output `*_tree_centered_sentinel_phenology.npz` files contain the
`sentinel_phenology` array and its `sentinel_phenology_columns`, keyed to the
crown-centered record index for direct use by the classification pipeline.

### Rebuild clean sidecars from existing time series

If regular Sentinel time-series CSVs already exist and only the clean
`tree_id`-keyed model sidecars need to be regenerated, run:

```powershell
python dataCollectionPreprocessing/Sentinel2/derive_clean_tree_id_centered_sentinel_phenology_sidecars.py `
  --crop-root H:/TreeCenteredModelInputs/tree_centered_naip_crops_clean `
  --original-timeseries-root E:/cell/sentinel2_timeseries `
  --supplemental-timeseries-root E:/TreeCenteredModelInputs/tree_centered_sentinel_timeseries_supplemental `
  --output-dir H:/TreeCenteredModelInputs/tree_centered_sentinel_phenology_clean `
  --city-token denver
```

This clean-sidecar command uses detected-crown coordinates by default. Use it
for rebuilding the current model inputs; it does not download observations.

## 4. Join products back to crowns

The GEE exporter writes the input `reduced_id` as `row_index`. Rename it back
to `reduced_id` before joining downloaded CSV rows to crowns.

```python
import pandas as pd

measurements = pd.read_csv("D:/sentinel2/denver/raw/sentinel2_export.csv")
measurements = measurements.rename(columns={"row_index": "reduced_id"})
tree_cells = pd.read_csv(
    "D:/sentinel2/denver/cells/tree_to_sentinel10m_cell.csv"
)
tree_measurements = tree_cells.merge(
    measurements,
    on="reduced_id",
    how="inner",
    validate="many_to_many",
)
```

## Operational guidance

- Run `--dry-run` before each new city or date range.
- Point `--completed-dir` at downloaded outputs to avoid duplicate GEE tasks.
- Start with short date windows, then expand after validating the results.
- Let the exporter enforce `--max-active-tasks`; skip that check only if the
  GEE task-list service is unavailable.
