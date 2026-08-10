# PRISM Climate Data Pipeline

This folder contains the minimal workflow for downloading PRISM climate grids
for the study cities and optionally deriving daily weather context aligned with
Sentinel-2 observations.

## Folder contents

| File | Purpose |
|---|---|
| `download_prism_sentinel_cells.py` | Plans and downloads PRISM normals, daily, monthly, or annual ZIP grids. It can optionally sample grids at study Sentinel-cell centers. |
| `derive_prism_daily_sentinel_context.py` | Uses downloaded daily grids, monthly normals, and Sentinel observation dates to derive rolling weather-context features. |

## Requirements

Both scripts require Python, NumPy, and pandas. Install `rasterio` when using
`--extract-values` or deriving daily context:

```powershell
python -m pip install numpy pandas rasterio
```

The scripts reuse city-coordinate helpers from:

```text
dataCollection\LiDAR\identify_tnm_lidar_city_coverage.py
```

They expect study-cell tables beneath:

```text
dataCollection\Sentinel2\mccoy_sentinel_10m_cells_utm\<city>\
```

The download step requires internet access to the public PRISM file server.

## Supported PRISM products

`download_prism_sentinel_cells.py` supports:

| Product | Date input | Typical use |
|---|---|---|
| `normal-monthly` | Months `1-12` | Thirty-year monthly climate normals |
| `normal-annual` | No date required | Thirty-year annual climate normals |
| `daily` | Start/end dates or explicit dates | Weather sequences and rolling context |
| `monthly` | Start/end dates, years/months, or explicit dates | Historical monthly climate |
| `annual` | Start/end dates, years, or explicit dates | Historical annual climate |

Default variables are:

```text
ppt,tmin,tmax,tmean,vpdmin,vpdmax
```

Common variables:

- `ppt`: precipitation;
- `tmin`: minimum temperature;
- `tmax`: maximum temperature;
- `tmean`: mean temperature;
- `vpdmin`: minimum vapor-pressure deficit;
- `vpdmax`: maximum vapor-pressure deficit.

The default source is the conterminous-US, 4 km, all-networks PRISM product.
Use `--region ak` for Alaska and override resolution or URL templates only
after confirming availability on the PRISM server.

## Output layout

The default output root is:

```text
E:\PRISM\sentinel_cells\
```

Downloaded archives are organized as:

```text
E:\PRISM\sentinel_cells\raw\<product>\<variable>\*.zip
```

For example:

```text
E:\PRISM\sentinel_cells\raw\daily\ppt\prism_ppt_us_25m_20200101.zip
E:\PRISM\sentinel_cells\raw\normal-monthly\tmean\prism_tmean_us_25m_202001_avg_30y.zip
```

Each invocation also writes:

- `prism_download_manifest.csv`;
- `prism_download_manifest.json`.

The manifest records product, variable, date, URL, destination, status,
downloaded bytes, and the last error.

## 1. Inspect a download plan

The downloader requires either one or more `--city-token` values or
`--all-cities`. Begin with `--dry-run`, which prints planned requests without
writing a manifest or downloading files.

Example monthly-normal plan for one city:

```powershell
python dataCollection\PRISM\download_prism_sentinel_cells.py `
  --city-token atlanta `
  --product normal-monthly `
  --variables ppt,tmin,tmax,tmean,vpdmin,vpdmax `
  --months 1-12 `
  --output-dir E:\PRISM\sentinel_cells `
  --dry-run
```

Confirm the printed URLs, variables, and number of requests before downloading.

## 2. Download monthly climate normals

Download all 12 monthly normals for the current variables:

```powershell
python dataCollection\PRISM\download_prism_sentinel_cells.py `
  --all-cities `
  --product normal-monthly `
  --variables ppt,tmin,tmax,tmean,vpdmin,vpdmax `
  --months 1-12 `
  --output-dir E:\PRISM\sentinel_cells
```

PRISM grids are national products, so selecting additional cities does not
duplicate grid downloads. City selection controls which study cells are loaded
and later sampled.

To create only the expected manifest without downloading:

```powershell
python dataCollection\PRISM\download_prism_sentinel_cells.py `
  --all-cities `
  --product normal-monthly `
  --variables ppt,tmean,vpdmax `
  --months 1-12 `
  --output-dir E:\PRISM\sentinel_cells `
  --query-only
```

`--query-only` writes the CSV and JSON manifests; `--dry-run` only prints the
plan.

## 3. Download daily PRISM grids

The current daily tree-centered sidecar uses `ppt`, `tmean`, and `vpdmax` by
default. Download the full date range required by the model dataset:

```powershell
python dataCollection\PRISM\download_prism_sentinel_cells.py `
  --all-cities `
  --product daily `
  --variables ppt,tmean,vpdmax `
  --start-date 2018-01-01 `
  --end-date 2023-12-31 `
  --output-dir E:\PRISM\sentinel_cells
```

For a short test:

```powershell
python dataCollection\PRISM\download_prism_sentinel_cells.py `
  --city-token atlanta `
  --product daily `
  --variables ppt,tmean,vpdmax `
  --start-date 2022-06-01 `
  --end-date 2022-06-07 `
  --output-dir E:\PRISM\sentinel_cells `
  --dry-run
```

Alternative date selectors include:

- `--dates 20220601,20220602,20220603`;
- `--years 2020,2022-2023` for annual products;
- `--years 2020-2022 --months 4-10` for monthly products.

For daily products, `--years` alone generates annual-style date tokens rather
than every day in those years. Use `--start-date` and `--end-date` for a
continuous daily sequence.

## 4. Resume or repair downloads

The downloader validates existing files as ZIP archives. Valid files are
marked `skipped_existing`; missing, empty, or invalid files are downloaded
again.

Useful options:

- `--retries 3`: number of attempts per file;
- `--timeout 120`: request timeout in seconds;
- `--sleep 0.2`: delay between attempts and requests;
- `--chunk-mb 8`: streaming download chunk size;
- `--overwrite`: redownload valid existing files;
- `--no-verify-tls`: troubleshooting option for certificate problems;
- `--exclude-city-token honolulu`: exclude an unsupported or unwanted city;
- `--manifest-name NAME`: use a distinct manifest name for separate runs.

The manifest is rewritten periodically and at completion, so rerunning the same
command safely skips valid archives.

## 5. Optionally sample PRISM values at study cells

Add `--extract-values` to sample downloaded grids at each selected Sentinel-cell
center:

```powershell
python dataCollection\PRISM\download_prism_sentinel_cells.py `
  --city-token atlanta `
  --product normal-monthly `
  --variables ppt,tmean,vpdmax `
  --months 1-12 `
  --output-dir E:\PRISM\sentinel_cells `
  --extract-values
```

The default sampled output is:

```text
E:\PRISM\sentinel_cells\prism_sentinel_cell_values.csv
```

It contains city, cell identifiers, coordinates, product, variable, date,
value, and source archive. Raster files extracted from ZIPs are deleted after
sampling unless `--keep-extracted-rasters` is supplied.

Use a different `--cell-output-name` when sampling multiple product families
to avoid replacing a previous CSV.

## 6. Create daily weather products

After downloading daily grids, `derive_prism_daily_sentinel_context.py` can
calculate weather features for dates represented in the Sentinel-2 time-series
tables. Its default features include:

- cumulative growing-degree days above 5 °C;
- 30-day growing-degree days;
- 30- and 90-day precipitation;
- 30-day dry-day count;
- 30-day mean and maximum `vpdmax`;
- precipitation and `vpdmax` anomalies relative to monthly normals.

The current model uses daily precipitation (`ppt`), mean temperature (`tmean`),
and maximum vapor-pressure deficit (`vpdmax`). It does not use the older PRISM
climate-normal sidecars or normal-anomaly features.

Run a small dry test first:

```powershell
python dataCollectionPreprocessing\PRISM\derive_prism_daily_sentinel_context.py `
  --city-token atlanta `
  --sentinel2-dir E:\cell\sentinel2_timeseries `
  --prism-daily-root E:\PRISM\sentinel_cells\raw\daily `
  --output-dir E:\PRISM\sentinel_cells\daily_context `
  --start-date 2022-06-01 `
  --end-date 2022-06-30 `
  --max-cells-per-city 100 `
  --max-target-dates 5 `
  --dry-run
```

Then remove the debug limits and `--dry-run`:

```powershell
python dataCollectionPreprocessing\PRISM\derive_prism_daily_sentinel_context.py `
  --all-cities `
  --sentinel2-dir E:\cell\sentinel2_timeseries `
  --prism-daily-root E:\PRISM\sentinel_cells\raw\daily `
  --output-dir E:\PRISM\sentinel_cells\daily_context `
  --variables ppt,tmean,vpdmax
```

By default, missing daily archives are errors. Use
`--missing-daily-policy nan` to retain dates with missing values or
`--missing-daily-policy skip-date` to omit incomplete dates.

## Current tree-centered model inputs

Run the current clean daily sidecar workflow with:

```powershell
dataCollectionPreprocessing\PRISM\run_clean_prism_daily_sidecars.cmd
```

This invokes
`dataCollectionPreprocessing/PRISM/derive_clean_tree_id_centered_prism_daily_sidecars.py`
for 2021-01-01 through 2023-12-31.

The daily sidecar script reads raw daily archives directly from:

```text
E:\PRISM\sentinel_cells\raw\daily\<variable>\
```

Do not delete the raw ZIP archives after optional cell-value extraction.

The resulting products are written under
`H:\TreeCenteredModelInputs\tree_centered_prism_daily_clean`. Daily values are
stored once per unique PRISM pixel, and city sidecars map each `tree_id` and
Sentinel cell to the shared daily sequence.

## QA checklist

Before deriving sidecars or training models:

1. Confirm every requested variable/date has status `downloaded` or
   `skipped_existing` in the manifest.
2. Investigate HTTP errors and invalid ZIP responses.
3. Confirm daily archive filenames contain the expected `YYYYMMDD` date.
4. Confirm the daily series covers the full model interval without unintended gaps.
5. Sample a few grids and verify temperatures, precipitation, and VPD are in
   plausible ranges.
6. Confirm the selected region and resolution cover all study cities.
7. Preserve raw ZIP archives until all sidecars and model shards are validated.

If PRISM changes its public URL layout, use `--url-template` to supply a new
template. Available placeholders include `{variable}`, `{date}`, `{year}`,
`{month}`, `{product}`, `{region}`, `{resolution}`, `{grid_token}`,
`{time_series_kind}`, and `{stability}`.
