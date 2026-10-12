# Google Satellite Embeddings

This directory reproduces the annual Google Satellite Embedding inputs used by
the multimodal tree classifier. The public workflow starts from the final
crown-centered NAIP metadata; it does not require the transitional
"missing-cell" inventories created during development of the study.

## Workflow

1. Run `export_tree_centered_satellite_embeddings.py` on the final NAIP crop
   metadata. The script converts each matched crown coordinate to its unique
   10 m grid cell and exports bands `A00`-`A63` for 2021-2023 from
   `GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL` through Google Earth Engine.
2. Download the completed CSV exports from Google Drive into one directory.
3. Run `derive_clean_tree_id_centered_gee_sidecars.py` to join the annual
   embeddings to crop rows by `crown_cell_id` and create one NPZ sidecar per
   city.
4. Supply that sidecar root to
   `../Shard/assemble_clean_tree_id_centered_model_input_shards.py`.

Shared city normalization, crown-cell construction, constants, and quality
calculations are implemented in `satellite_embedding_utils.py`; both workflow
scripts import the same definitions.

The final-study files were produced while the project was transitioning from
inventory-coordinate to crown-centered extraction. Some study rows therefore
used an original export whose coordinate resolved to the same crown-centered
10 m cell, while the remainder used supplemental crown-cell exports. This
historical recovery procedure is reported in `provenance/`, but it is not
needed when rebuilding all embeddings from scratch.

## Requirements

Use the repository `environment.yml`. Earth Engine must be authenticated, and
`GEE_PROJECT` may be set to the billing/project identifier. Input paths are
always supplied explicitly; examples below use `$DATA_ROOT` as a placeholder.

## 1. Submit exports

```powershell
python dataCollectionPreprocessing/satelliteEmbeddings/export_tree_centered_satellite_embeddings.py `
  --crop-root "$DATA_ROOT/tree_centered_naip_crops_clean" `
  --completed-dir "$DATA_ROOT/tree_centered_satellite_embedding_exports" `
  --manifest "$DATA_ROOT/runtime/satellite_embedding_export_manifest.jsonl" `
  --drive-folder GEE_TREE_CENTERED_SATELLITE_EMBEDDING `
  --years 2021 2022 2023
```

The JSONL manifest is a resumable operational ledger containing Earth Engine
task identifiers. It is generated data and should not be committed.

Use `--points-file` with a CSV containing `longitude` and `latitude` only when
resubmitting an already prepared single-city crown-cell table. Use
`--city-token CITY` to identify that city.

## 2. Build sidecars

```powershell
python dataCollectionPreprocessing/satelliteEmbeddings/derive_clean_tree_id_centered_gee_sidecars.py `
  --crop-root "$DATA_ROOT/tree_centered_naip_crops_clean" `
  --embedding-root "$DATA_ROOT/tree_centered_satellite_embedding_exports" `
  --output-dir "$DATA_ROOT/tree_centered_satellite_embedding_sidecars" `
  --embedding-years 2021 2022 2023
```

The default export filename pattern is
`tree_centered_satellite_embedding_{city}_{year}_batch_*.csv`.

## Output and QA

Each city sidecar contains the arrays documented in
`provenance/satellite_embedding_sidecar_schema.json`. Missing embeddings are
retained as zero-filled vectors with a false year mask and are subsequently
screened during shard assembly. The saved `config_json` contains portable
processing settings and filenames, not workstation paths.

Use `--overwrite` to replace existing city sidecars. All year options accept a
space-separated list, for example `--embedding-years 2021 2022 2023` and
`--expected-years 2021 2022 2023`.

### Embedding source codes

The stable `embedding_source_code` values are:

- `0`: no annual satellite embedding was available for the row.
- `1`: historical original-coordinate export that resolved to the final
  crown-centered cell. This occurs only in frozen study sidecars.
- `2`: direct crown-centered cell export. All matched rows in a clean
  from-scratch rebuild receive this code.

The compatibility flags `used_original_satellite_embedding` and
`used_additional_satellite_embedding` correspond to codes `1` and `2`,
respectively.

Validate the sidecars before shard assembly:

```powershell
python dataCollectionPreprocessing/satelliteEmbeddings/validate_satellite_embedding_sidecars.py `
  --sidecar-root "$DATA_ROOT/tree_centered_satellite_embedding_sidecars" `
  --summary "$DATA_ROOT/runtime/satellite_embedding_validation_summary.csv"
```

The validator checks schema completeness, row alignment, dimensions, finite
values, year masks, quality counts, source flags, unique identifiers, expected
years, and portable saved configuration. It exits with a nonzero status when
any sidecar fails.

## Frozen provenance

`provenance/` records the immutable study configuration and a city-level
summary of the sidecars used for the manuscript. Runtime manifests and newly
generated summaries are deliberately excluded from version control.
