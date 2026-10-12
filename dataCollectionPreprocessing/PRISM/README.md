# PRISM Daily Climate Data Pipeline

This folder contains the workflow used for the manuscript: download daily
PRISM grids and construct deduplicated daily climate sidecars mapped to each
tree's Sentinel-2 cell. Legacy rolling-weather and climate-normal feature code
was removed because it did not generate the final model inputs.

## Files

| File | Purpose |
|---|---|
| `prism_common.py` | Shared city-token, date, range, and Sentinel-cell utilities. |
| `download_prism_sentinel_cells.py` | Plans, downloads, validates, and optionally samples PRISM grids. |
| `derive_clean_tree_id_centered_prism_daily_sidecars.py` | Creates the final daily `ppt`, `tmean`, and `vpdmax` sidecars. |
| `freeze_study_prism_provenance.py` | Freezes path-independent city, grid, and archive provenance. |
| `environment-prism.yml` | Reproducible open-source environment. |
| `provenance/` | Frozen final-study configuration and summaries. |

## Environment

```powershell
conda env create --file dataCollectionPreprocessing/PRISM/environment-prism.yml
conda activate treeid-prism
```

Downloading requires internet access. Raster sampling and sidecar creation use
Rasterio. NumPy and pandas provide array and table processing.

## Final study configuration

The authoritative parameters are in
`provenance/study_prism_configuration.json`. The final inputs contain daily:

- precipitation (`ppt`);
- mean temperature (`tmean`);
- maximum vapor-pressure deficit (`vpdmax`).

The interval is 2021-01-01 through 2023-12-31 (1,095 days). Data use the
conterminous-US, approximately 4-km, all-networks PRISM daily product. The
`25m` filename token is a PRISM grid designation; it does not mean 25 metres.

PRISM values are not independent observations for every tree. Values are
stored once for each unique PRISM grid cell, and each city file maps its trees
and Sentinel cells to those shared sequences. During shard assembly, daily
values are appended only to retained Sentinel-2 observation dates.

## Workflow dependencies

1. `dataCollectionPreprocessing/Sentinel2/reduce_tree_points_to_sentinel_cells.py`
   creates the Sentinel-cell coverage tables used by the downloader.
2. `dataCollectionPreprocessing/shard/build_clean_tree_record_metadata.py`
   creates clean tree-to-Sentinel-cell linkage tables.
3. This folder downloads PRISM and creates daily sidecars.
4. `dataCollectionPreprocessing/shard/assemble_clean_tree_id_centered_model_input_shards.py`
   joins PRISM values to Sentinel dates in the final model shards.

Large grids and arrays are not stored in Git. Every data location must be
provided explicitly; scripts contain no workstation storage defaults.

## 1. Inspect or download daily grids

Use a portable root variable for local storage:

```powershell
$DATA_ROOT = "path/to/urban_tree_data"

python dataCollectionPreprocessing/PRISM/download_prism_sentinel_cells.py `
  --all-cities `
  --cell-map-dir "$DATA_ROOT\sentinel_cells" `
  --output-dir "$DATA_ROOT\prism" `
  --product daily `
  --variables ppt,tmean,vpdmax `
  --start-date 2021-01-01 `
  --end-date 2023-12-31 `
  --dry-run
```

Review the planned URLs and request count, then remove `--dry-run`. Use
`--query-only` to write a mutable download manifest without downloading.
Existing ZIP files are checked before reuse; invalid archives are downloaded
again.

The source is the PRISM Climate Group public file server at Oregon State
University. The default URL pattern is exposed through `--url-template` so a
future server-layout change does not require modifying the downloader.

## 2. Create the final daily sidecars

```powershell
python dataCollectionPreprocessing/PRISM/derive_clean_tree_id_centered_prism_daily_sidecars.py `
  --raw-root "$DATA_ROOT\prism\raw\daily" `
  --link-root "$DATA_ROOT\tree_record_metadata_clean" `
  --output-dir "$DATA_ROOT\tree_centered_prism_daily_clean" `
  --variables ppt,tmean,vpdmax `
  --start-date 2021-01-01 `
  --end-date 2023-12-31
```

Use `--max-dates` for a short smoke test. Existing completed global output is
protected unless `--overwrite` is supplied.

The output contract is:

- `prism_daily_values.npz`
  - `values[prism_pixel, date, variable]`, `float32`;
  - `valid_mask` with the same shape;
  - `dates_yyyymmdd`;
  - `variable_names`;
  - PRISM grid rows and columns;
  - path-independent run configuration and raster-grid metadata.
- `<city>/<city>_prism_daily_index.npz`
  - `tree_id` and Sentinel cell identifiers;
  - coordinates and PRISM grid indices;
  - `prism_daily_source_row` into the global array;
  - `missing_prism_daily`.
- `prism_daily_sidecar_summary.csv`, with mapping support by city.

Nodata and non-finite raster values are stored as zero in `values` and marked
false in `valid_mask`; consumers must use the mask.

## Honolulu coverage

The retained conterminous-US PRISM grid does not cover Honolulu. All 15,428
Honolulu records have `missing_prism_daily = true`; no climate values were
invented or extrapolated. These records must be excluded from analyses that
require PRISM or handled through the existing missing-data QA pathway. A future
Hawaii analysis requires a separately documented compatible climate source.

## Frozen provenance

```powershell
python dataCollectionPreprocessing/PRISM/freeze_study_prism_provenance.py `
  --sidecar-root "$DATA_ROOT\tree_centered_prism_daily_clean" `
  --archive-root "$DATA_ROOT\prism\raw\daily" `
  --output-dir dataCollectionPreprocessing/PRISM/provenance
```

This creates a path-independent city summary, global-array/grid summary, and a
SHA-256/ZIP-integrity inventory of every retained archive. The historical
archive volume was not mounted during the current repository cleanup, so the
tracked provenance status explicitly reports that the checksum inventory is
pending. Rerun the command without `--allow-missing-archive-root` when that
volume is available.

## QA checklist

1. Confirm all 3,285 expected variable-date archives are present and valid.
2. Confirm filenames cover all 1,095 dates for each of three variables.
3. Review checksums and ZIP integrity in the frozen archive inventory.
4. Confirm the retained global array has 1,523 PRISM pixels, 1,095 dates, and
   three variables.
5. Inspect plausible ranges and units for precipitation, temperature, and VPD.
6. Review city mapping percentages, especially any partial or zero coverage.
7. Confirm masks are retained through shard assembly.
8. Preserve original archives until model inputs and frozen provenance are
   independently verified.
