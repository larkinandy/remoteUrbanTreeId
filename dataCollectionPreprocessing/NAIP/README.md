# Tree-Centered NAIP Data Pipeline

This folder contains the reproducible workflow used to select and download
National Agriculture Imagery Program (NAIP) imagery and create RGBNIR image
chips centered on identified tree crowns. These are tree-centered crops, not
Sentinel-cell-centered crops.

<img src="https://github.com/larkinandy/remoteUrbanTreeId/blob/main/images/NAIP.png" width="800">

## Folder contents

| File | Purpose |
|---|---|
| `build_county_manifest.py` | Intersects inventory coordinates with county boundaries and creates the city-to-county manifest. |
| `county_boundaries.geojson` | Exact county boundary file used in the workflow. |
| `download_naip_paired_box.py` | Selects and downloads same-year natural-color and companion/NIR county archives. |
| `extract_tree_id_centered_naip_crops.py` | Creates RGBNIR crops from paired natural-color and companion/NIR imagery. |
| `extract_tree_id_centered_5band_naip_crops.py` | Creates RGBNIR crops from single multiband imagery. |
| `tree_centered_naip_utils.py` | Shared raster, archive, reprojection, block-reading, resizing, and QA functions. |
| `naip_cli.py` | Shared CLI definitions and input-validation helpers used by the large workflow entry points. |
| `inspect_sid_metadata.py` | Prints ArcPy raster, spatial-reference, and band metadata for SID files. |
| `freeze_study_naip_provenance.py` | Reconstructs path-independent frozen source tables from retained outputs. |
| `provenance/` | Frozen study configuration, boundary metadata, city source formats, and raster inventory. |
| `environment-naip-download.yml` | Open-source environment for manifest construction and downloads. |

Generated manifests, catalog caches, selections, summaries, and crop metadata
are ignored outside `provenance/`. Frozen files in `provenance/` are tracked.

## Environments

Two environments were used.

### Archive discovery and download

```powershell
conda env create --file dataCollectionPreprocessing/NAIP/environment-naip-download.yml
conda activate treeid-naip-download
python -m playwright install msedge
```

This environment runs the manifest builder, Box downloader, and provenance
freezer.

### MrSID crop extraction

The two crop extractors and `inspect_sid_metadata.py` were run with the Python
environment bundled with **ArcGIS Pro 3.6**, including ArcPy. ArcPy is licensed
software and cannot be installed by this repository. NumPy and pandas must
remain compatible with ArcGIS Pro; add `pyproj` to an ArcGIS environment clone
if projected crown coordinates are not already available.

## Configuration and paths

Scientific settings and storage locations are separate:

- `provenance/study_naip_configuration.json` freezes the manuscript settings.
- Inputs, outputs, caches, and mutable manifests are explicit runtime paths.
  The scripts contain no workstation drive-letter defaults.
- Example commands use a `$DATA_ROOT` variable rather than a workstation path.

The study used 64 × 64-pixel, 38 × 38-metre RGBNIR crops and best-overlap
source selection. Baltimore and Washington, DC used the single-multiband
workflow; the other retained cities used paired natural-color and companion
imagery. The frozen configuration and city table are authoritative.

## Workflow dependencies

1. Municipal inventory files provide coordinates for county assignment.
2. `dataCollectionPreprocessing/treeCrown/spatial_join_clean_tree_records_to_detected_crowns.py`
   creates the clean tree-to-crown joins used as crop centers.
3. This folder selects/downloads NAIP and builds row-aligned crops and metadata.
4. `dataCollectionPreprocessing/LiDAR/create_tree_centered_lidar_crop_products_laspy_pipeline.py`
   uses the NAIP footprint and crown coordinates for aligned LiDAR products.
5. `dataCollectionPreprocessing/shard/assemble_clean_tree_id_centered_model_input_shards.py`
   consumes the NAIP and LiDAR products.

All referenced scripts are in this repository. Large inventories, imagery,
joins, and output arrays are not stored in Git.

## Crown-coordinate input

Clean tree-to-crown joins require `tree_id`, `crown_id`, `crown_epsg`, and
`match_distance_m`, plus either `crown_x_utm`/`crown_y_utm` or
`crown_lon`/`crown_lat`. `tree_id` must be unique within a city.

## 1. Build the county manifest

The exact boundary file is checksum-frozen in
`provenance/county_boundaries_provenance.json`. Its original source and vintage
were not retained and are explicitly recorded as a legacy limitation.

```powershell
$DATA_ROOT = "path/to/urban_tree_data"
python dataCollectionPreprocessing/NAIP/build_county_manifest.py `
  --source "$DATA_ROOT\inventories" `
  --boundaries dataCollectionPreprocessing/NAIP/county_boundaries.geojson `
  --output "$DATA_ROOT\manifests\naip_county_manifest.json"
```

Inventory files must match `*_Final_*.csv` and contain longitude and latitude
columns in the singular or plural naming supported by the script. Review all
unmatched points and city/county assignments.

## 2. Select and download paired archives

```powershell
python dataCollectionPreprocessing/NAIP/download_naip_paired_box.py `
  --manifest "$DATA_ROOT\manifests\naip_county_manifest.json" `
  --output "$DATA_ROOT\naip_paired" `
  --index-cache "$DATA_ROOT\manifests\naip_paired_box_index.json" `
  --selection-output "$DATA_ROOT\manifests\naip_paired_selection.json" `
  --target-year 2022 `
  --year-window 6 `
  --skip-city-code BAL `
  --skip-city-code DCA `
  --test-city Atlanta `
  --dry-run
```

Remove `--test-city` and `--dry-run` for the complete paired download. The two
explicit skip arguments correspond to the frozen single-multiband exceptions;
they are not hidden in the code. Preserve each run's selection manifest.

## 3. Create paired RGBNIR crops

```powershell
python dataCollectionPreprocessing/NAIP/extract_tree_id_centered_naip_crops.py `
  --city-token atlanta `
  --join-root "$DATA_ROOT\tree_to_crowns" `
  --naip-dir "$DATA_ROOT\naip_paired" `
  --county-manifest "$DATA_ROOT\manifests\naip_county_manifest.json" `
  --output-dir "$DATA_ROOT\tree_centered_naip" `
  --crop-size 64 `
  --crop-metres 38 `
  --sid-selection-mode best-overlap `
  --max-records 100
```

Inspect this smoke test and remove `--max-records`. Repeat `--city-token` for
multiple cities. Alternatively, use one `--join-csv` with exactly one city or
provide a `--job-csv`. City discovery requires `--join-root`.

## 4. Create single-multiband crops

```powershell
python dataCollectionPreprocessing/NAIP/extract_tree_id_centered_5band_naip_crops.py `
  --city-token baltimore `
  --join-csv "$DATA_ROOT\tree_to_crowns\baltimore\baltimore_tree_to_nearest_detected_crown_5m.csv" `
  --naip-dir "$DATA_ROOT\naip_multiband\BAL" `
  --output-dir "$DATA_ROOT\tree_centered_naip" `
  --sid-pattern *.sid `
  --bands 1,2,3,4 `
  --crop-size 64 `
  --crop-metres 38 `
  --sid-selection-mode best-overlap `
  --max-records 100
```

Bands are one-based and ordered red, green, blue, NIR. Remove `--max-records`
after validating the test output.

## Outputs and restart behavior

Each city directory contains a row-aligned `uint8` RGBNIR `.npy` array, a
metadata CSV with tree/crown identifiers and raster/band/QA provenance, a run
configuration JSON, and temporary progress files during incomplete runs.
Completed outputs are protected unless `--force` is supplied.

## Frozen study provenance

The retained outputs were summarized with:

```powershell
python dataCollectionPreprocessing/NAIP/freeze_study_naip_provenance.py `
  --crop-root "$DATA_ROOT\tree_centered_naip" `
  --boundaries dataCollectionPreprocessing/NAIP/county_boundaries.geojson `
  --output-dir dataCollectionPreprocessing/NAIP/provenance
```

The source table removes workstation directory prefixes while preserving
source filenames, roles, bands, cities, and selected-record counts. The study
has 40 retained configurations: 38 paired and two single-multiband.
The original mutable Box selection file was not retained, so the frozen tables
identify the exact rasters actually read rather than attaching unverified
current catalog URLs.

## QA checklist

1. Confirm arrays and metadata have identical row counts.
2. Confirm unique `tree_id` values remain aligned with `crop_index`.
3. Investigate or exclude `crop_failed` records.
4. Review blackout, whiteout, saturation, and valid-pixel fractions.
5. Inspect RGB and false-color NIR samples.
6. Verify source rasters and bands for every city.
7. Preserve archives, manifests, and configurations until outputs are backed up.
