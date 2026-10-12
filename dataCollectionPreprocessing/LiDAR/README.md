# Tree-Centered LiDAR Data Pipeline

This folder contains the workflow for discovering and downloading USGS 3D
Elevation Program (3DEP) LiDAR tiles and creating DTM, DSM, CHM, and point-count
products centered on identified tree crowns.

<img src = https://github.com/larkinandy/remoteUrbanTreeId/blob/main/images/CHM.png width="800">


## Folder contents

| File | Purpose |
|---|---|
| `identify_tnm_lidar_city_coverage.py` | Queries TNM and provides the city-footprint, coordinate, sampling, and catalog helpers used during manifest construction. |
| `build_tnm_lidar_download_manifest.py` | Queries The National Map (TNM), evaluates city coverage, selects suitable LiDAR projects, and writes download manifests. |
| `download_tnm_lidar_manifest.py` | Downloads the LAZ files in the manifest with restartable, bounded parallelism. |
| `create_tree_centered_lidar_crop_products_laspy_pipeline.py` | Creates crown-centered DTM, DSM, CHM, and point-count products from downloaded LAZ/LAS files. |
| `derive_clean_tree_id_centered_chm_structure_sidecars.py` | Aligns NAIP crops with CHM rasters and calculates the 114 structure features consumed during shard assembly. |
| `naip_chm_structure_metrics.py` | Shared definitions and calculations for the 114 NAIP/CHM structure features. |
| `tree_centered_lidar_utils.py` | Shared manifest, CRS, vertical-unit, memmap, and output utilities. This module is imported by the product generator and is not run directly. |
| `provenance/study_lidar_configuration.json` | Frozen parameters used for the study workflow. |
| `provenance/` | Frozen configuration, tile, city, and project records describing the LiDAR inputs actually used. |
| `build_frozen_study_provenance.py` | Rebuilds the frozen provenance tables from retained bin-state and final LiDAR-index files. |

Transient query manifests, download status files, run summaries, `__pycache__`,
and `.pyc` files are generated and ignored. Files under `provenance/` are
frozen study records and are intentionally version controlled.

## Requirements

Two environments were used:

- Upstream GIS operations were run with the Python environment bundled with
  **ArcGIS Pro 3.6**, including ArcPy. ArcPy is licensed software and cannot be
  installed by this repository's Conda files.
- TNM discovery, download management, laspy-first binning, CHM derivation, and
  structure-sidecar construction can be run with `environment-lidar.yml`.

Create the open-source LiDAR environment with:

```powershell
conda env create --file dataCollectionPreprocessing/LiDAR/environment-lidar.yml
conda activate treeid-lidar
```

The TNM discovery and download steps require internet access. Product creation
is local once the LAZ files have been downloaded.

## Configuration and path handling

Scientific parameters and filesystem locations are intentionally handled
separately:

- `provenance/study_lidar_configuration.json` freezes the scientific and
  processing choices used in the manuscript, including project-year and
  coverage criteria, raster dimensions, point classifications, vertical-unit
  handling, and structure-feature thresholds.
- Input and output locations are runtime concerns. The scripts do not assume
  workstation drive letters or storage layouts; all data roots, manifests,
  summaries, and output destinations must be supplied as command-line
  arguments.
- Paths shown below under `D:\urban_tree_data` are illustrative. They are not
  defaults and may be replaced with any local or mounted storage location.

This separation allows the published study parameters to remain fixed while
the workflow is run on systems with different storage layouts.

## Workflow overview

```mermaid
flowchart TD
    A["Study cities and coverage footprints"] --> B["Query TNM and select LiDAR projects"]
    B --> C["TNM download manifest and city summary"]
    C --> D["Download LAZ tiles"]
    E["Tree-centered NAIP metadata with crown coordinates"] --> F["Bin LiDAR returns around crowns"]
    D --> F
    F --> G["Candidate ground and surface memmaps"]
    G --> H["DTM, DSM, CHM, and point-count arrays"]
    H --> I["LiDAR index and run summary"]
```

## Frozen study provenance

The parameters used for the manuscript are recorded in
`provenance/study_lidar_configuration.json`. The `provenance/` directory contains:

- `frozen_study_lidar_tile_manifest.csv`: the 8,023 tile filenames actually
  read by the completed study binning runs, together with city, project,
  preference rank, relative path, and retained file size;
- `frozen_study_lidar_city_summary.csv`: the 36 city-level project selections
  represented in the final CHM products;
- `frozen_study_lidar_project_table.csv`: a compact publication-facing table
  of acquisition year, record support, tile count, and vertical scaling.

The original manifest was a mutable download-management file: its status,
attempt, timestamp, and local-path columns changed while downloads ran. That
mutable file was not retained. The frozen tables were reconstructed from the
completed bin-state files and final per-record LiDAR indexes, so they preserve
the exact tile filenames and project assignments used to create the study
products. Their URL column is intentionally blank rather than populated with
unverified current catalog URLs. A fresh TNM query can create a downloadable
manifest, but live catalog results may change over time.

## Inputs

### Dependencies on other repository workflows

All scripts required to create the LiDAR pipeline inputs are included:

1. `Sentinel2/reduce_tree_points_to_sentinel_cells.py` creates the city-level
   Sentinel-cell coverage tables used for TNM discovery.
2. `NAIP/build_county_manifest.py` creates `naip_county_manifest.json`, which
   supplies canonical city names and codes.
3. `NAIP/extract_tree_id_centered_naip_crops.py` creates the tree-centered NAIP
   crops and coordinate metadata used as LiDAR raster footprints.
4. This folder discovers/downloads LiDAR, creates DTM/DSM/CHM products, and
   derives structure sidecars.
5. `shard/assemble_clean_tree_id_centered_model_input_shards.py` consumes the
   resulting sidecars during model-input assembly.

Large generated coverage tables, imagery, LAZ files, and model arrays are not
stored in Git. Their locations must be passed explicitly to each script.

### Coverage inputs for TNM discovery

The current manifest builder expects:

- city coverage tables beneath
  `dataCollectionPreprocessing/Sentinel2/mccoy_sentinel_10m_cells_utm/<city>/`;
- `dataCollectionPreprocessing/NAIP/naip_county_manifest.json` for city names and
  three-letter codes.

These coverage tables are used only to select TNM projects and LAZ tiles. They
are not used as model records or LiDAR chip centers.

### Crown metadata for product creation

The product generator reads one metadata CSV per city beneath the directory
passed with `--crop-root`:

```text
<crop-root>\<city>\
  <city>_tree_id_centered_nearest_64px_metadata.csv
```

The filename is controlled by `--metadata-pattern`. Required fields are:

- `crop_index`;
- `crop_failed`;
- either `peak_x_utm` or `crown_x_utm`;
- either `peak_y_utm` or `crown_y_utm`;
- either `crown_epsg` or a `cell_id` containing an EPSG code.

Recommended fields are:

- `tree_id`;
- `tree_centered_index`;
- `crop_metres` and `crop_size`;
- `source_file` and `source_row`;
- `taxon_label`.

The clean NAIP crown extractor produces metadata matching this contract.

## 1. Build a TNM download manifest

Begin with one city and a dry run:

```powershell
python dataCollectionPreprocessing\LiDAR\build_tnm_lidar_download_manifest.py `
  --cell-map-dir dataCollectionPreprocessing\Sentinel2\mccoy_sentinel_10m_cells_utm `
  --city-manifest dataCollectionPreprocessing\NAIP\naip_county_manifest.json `
  --lidar-root D:\urban_tree_data\lidar `
  --output-csv D:\urban_tree_data\manifests\tnm_lidar_download_manifest.csv `
  --output-json D:\urban_tree_data\manifests\tnm_lidar_download_manifest.json `
  --city-summary-csv D:\urban_tree_data\manifests\tnm_lidar_download_city_summary.csv `
  --city-token atlanta `
  --dry-run
```

Generate the manifest after reviewing project age and coverage:

```powershell
python dataCollectionPreprocessing\LiDAR\build_tnm_lidar_download_manifest.py `
  --cell-map-dir dataCollectionPreprocessing\Sentinel2\mccoy_sentinel_10m_cells_utm `
  --city-manifest dataCollectionPreprocessing\NAIP\naip_county_manifest.json `
  --lidar-root D:\urban_tree_data\lidar `
  --output-csv D:\urban_tree_data\manifests\tnm_lidar_download_manifest.csv `
  --output-json D:\urban_tree_data\manifests\tnm_lidar_download_manifest.json `
  --city-summary-csv D:\urban_tree_data\manifests\tnm_lidar_download_city_summary.csv
```

Important selection options:

- `--city-token`: restrict to a city name, folder token, or code; repeatable;
- `--project-selection best|all`: select one preferred project or retain all;
- `--min-project-year`: reject older projects by default;
- `--preferred-project-year`: prefer sufficiently complete newer projects;
- `--allow-stale-projects`: permit projects older than the minimum;
- `--allow-event-projects`: allow flood, wildfire, hurricane, or other
  event-specific collections to compete normally;
- `--max-coverage-points`: subsample coverage footprints for faster discovery;
- `--max-tiles-per-city`: cap tiles during a smoke test;
- `--no-verify-tls`: troubleshooting option for environments with TLS issues.

Review the generated city summary before downloading. Confirm:

1. the selected project year is acceptable;
2. coverage is sufficient for the intended crowns;
3. event-specific or stale projects were not selected unintentionally;
4. the expected tile count and download size are reasonable.

## 2. Download the LAZ tiles

Inspect the queue without downloading:

```powershell
python dataCollectionPreprocessing\LiDAR\download_tnm_lidar_manifest.py `
  --manifest D:\urban_tree_data\manifests\tnm_lidar_download_manifest.csv `
  --status-json D:\urban_tree_data\manifests\tnm_lidar_download_status.json `
  --lidar-root D:\urban_tree_data\lidar `
  --city-token atlanta `
  --max-downloads 5 `
  --dry-run
```

Download one city:

```powershell
python dataCollectionPreprocessing\LiDAR\download_tnm_lidar_manifest.py `
  --manifest D:\urban_tree_data\manifests\tnm_lidar_download_manifest.csv `
  --status-json D:\urban_tree_data\manifests\tnm_lidar_download_status.json `
  --lidar-root D:\urban_tree_data\lidar `
  --city-token atlanta `
  --workers 4 `
  --per-city 2 `
  --resume-partial
```

Remove `--city-token` to download the entire manifest. The default `city`
schedule finishes one city before launching the next; use `--schedule global`
to prioritize the full manifest instead.

The downloader:

- skips files whose existing size is complete;
- writes `.partial` files during active transfers;
- updates the manifest with attempts, timestamps, downloaded bytes, and errors;
- periodically writes `tnm_lidar_download_status.json`;
- can resume partial HTTP downloads with `--resume-partial`.

Downloaded files are written beneath the explicitly supplied LiDAR root:

```text
<lidar-root>\<relative_path from manifest>
```

Avoid `--overwrite` unless existing LAZ files are known to be corrupt.

## 3. Test crown-centered LiDAR product creation

The product generator has two stages:

- `bin`: read LAZ points and create restartable ground/surface/count memmaps;
- `derive`: select the preferred project and derive DTM, DSM, CHM, count
  products, and the row index;
- `all`: run both stages, which is the default.

Start with one city, a small record limit, and a tile limit:

```powershell
python dataCollectionPreprocessing\LiDAR\create_tree_centered_lidar_crop_products_laspy_pipeline.py `
  --manifest D:\urban_tree_data\manifests\tnm_lidar_download_manifest.csv `
  --city-summary D:\urban_tree_data\manifests\tnm_lidar_download_city_summary.csv `
  --crop-root D:\urban_tree_data\tree_centered_naip_crops `
  --lidar-root D:\urban_tree_data\lidar `
  --output-root D:\urban_tree_data\tree_centered_lidar_products `
  --run-summary D:\urban_tree_data\manifests\tree_centered_lidar_run_summary.csv `
  --city-token atlanta `
  --max-records 100 `
  --max-tiles 2 `
  --dry-run
```

Remove `--dry-run` for the smoke test. After verifying its outputs, rerun the
city without `--max-records` and `--max-tiles` in a clean output directory.

## 4. Create full crown-centered products

```powershell
python dataCollectionPreprocessing\LiDAR\create_tree_centered_lidar_crop_products_laspy_pipeline.py `
  --stage all `
  --manifest D:\urban_tree_data\manifests\tnm_lidar_download_manifest.csv `
  --city-summary D:\urban_tree_data\manifests\tnm_lidar_download_city_summary.csv `
  --crop-root D:\urban_tree_data\tree_centered_naip_crops `
  --lidar-root D:\urban_tree_data\lidar `
  --output-root D:\urban_tree_data\tree_centered_lidar_products `
  --run-summary D:\urban_tree_data\manifests\tree_centered_lidar_run_summary.csv `
  --pixel-size 1 `
  --bin-workers 1 `
  --derive-workers 1
```

Useful options:

- `--city-token` and `--exclude-city-token`: select cities;
- `--crop-metres`: override the metadata footprint width;
- `--pixels`: override output pixels per side;
- `--target-epsg`: override the city output CRS;
- `--laspy-source-epsg`: fallback when a LAS/LAZ header has no CRS;
- `--tile-point-chunk-size`: stream unusually large tiles in chunks;
- `--checkpoint-every-tiles`: control bin-state checkpoint frequency;
- `--auto-accept-coverage`: minimum preferred-project coverage;
- `--preference-penalty`: penalty applied to lower-ranked candidate projects;
- `--allow-empty-products`: permit cities without usable returns;
- `--overwrite`: discard existing candidate state and rebuild;
- `--retry-active-tile`: retry an interrupted active tile only when it is known
  not to have flushed partial updates.

Keep `--bin-workers` modest. Each worker reads LAZ tiles and writes several
large memmaps.

## Vertical units and classifications

By default, the generator inspects sample LAS/LAZ CRS metadata and attempts to
convert vertical values to metres. Review the saved scaling reason and
confidence for every project.

Relevant options:

- `--auto-z-scale` / `--no-auto-z-scale`;
- `--z-scale`, such as `0.3048` for feet to metres;
- `--z-units`;
- `--z-scale-table` for per-city or per-project overrides;
- `--z-scale-audit-tiles` to control the number of inspected headers.

An override CSV may contain:

```csv
city_token,city_code,project,z_scale,z_units,confidence,reason
atlanta,ATL,example_project,1.0,meters,reviewed,manual_header_review
```

Default point classifications are:

- DTM: class `2` (ground);
- DSM: classes `1,2,3,4,5,6,9,17`.

Override these using `--dtm-class-codes` and `--dsm-class-codes` only after
reviewing the source project classification scheme.

## Outputs

The directory supplied through `--output-root` contains:

```text
Candidates\<city>\<project>\Binned\  restartable intermediate memmaps
DTM\<city>\                         ground elevation arrays
DSM\<city>\                         surface elevation arrays
CHM\<city>\                         canopy-height arrays and LiDAR index
Point_Counts\<city>\                return-count arrays
```

The CHM is derived from DSM minus DTM after applying the selected vertical
scale. The LiDAR index preserves tree/crop identifiers, source project,
coverage, and summary statistics required for later alignment.

The run summary is written to the explicit `--run-summary` path.

## Restart and recovery

- Completed bin markers cause the bin stage to skip finished candidates.
- Bin-state JSON records processed and active tiles.
- Interrupted runs can normally be rerun without `--overwrite`.
- If a run stopped during an active tile, inspect its state before using
  `--retry-active-tile`; double-counting is possible if partial updates were
  flushed.
- Use `--overwrite` only when intentionally rebuilding candidate memmaps.

## QA checklist

Before adding LiDAR products to model shards:

1. Confirm the LiDAR index and crown metadata have matching tree identifiers.
2. Confirm DTM, DSM, CHM, and count arrays have identical record and raster
   dimensions.
3. Verify `CHM >= 0` for valid pixels and inspect extreme canopy heights.
4. Review ground and surface return coverage by city and project.
5. Inspect vertical-unit inference and manually verify ambiguous projects.
6. Review several CHM chips alongside their corresponding NAIP crown crops.
7. Investigate cities with low coverage, empty products, or extensive missing
   ground returns.
8. Preserve the original LAZ files and manifests until products have been
   validated and backed up.

After QA, the clean CHM products can be converted to structure sidecars using
`dataCollectionPreprocessing/LiDAR/derive_clean_tree_id_centered_chm_structure_sidecars.py`:

```powershell
python dataCollectionPreprocessing/LiDAR/derive_clean_tree_id_centered_chm_structure_sidecars.py `
  --crop-root D:/urban_tree_data/tree_centered_naip_crops `
  --lidar-product-root D:/urban_tree_data/tree_centered_lidar_products `
  --output-dir D:/urban_tree_data/tree_centered_chm_structure `
  --dry-run
```

Review the discovered city inputs, then remove `--dry-run`. The resulting
`*_tree_id_centered_chm_structure_metrics.npz` sidecars are consumed by the QA
and shard-assembly stages.

`naip_chm_structure_metrics.py` contains the shared NAIP/CHM feature definitions
used by the sidecar builder and is not normally run directly.
