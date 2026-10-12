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
  --embedding-years 2021,2022,2023
```

The default export filename pattern is
`tree_centered_satellite_embedding_{city}_{year}_batch_*.csv`.

## Output and QA

Each city sidecar contains the arrays documented in
`provenance/satellite_embedding_sidecar_schema.json`. Missing embeddings are
retained as zero-filled vectors with a false year mask and are subsequently
screened during shard assembly. The saved `config_json` contains portable
processing settings and filenames, not workstation paths.

## Frozen provenance

`provenance/` records the immutable study configuration and a city-level
summary of the sidecars used for the manuscript. Runtime manifests and newly
generated summaries are deliberately excluded from version control.
