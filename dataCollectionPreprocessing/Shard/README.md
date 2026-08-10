# Model-input shard assembly

This folder is the final preprocessing boundary before model training. Its
scripts align independently created modality products, validate row identity
and completeness, and write the physical NPZ shards and manifests consumed by
the models.

Training scripts belong in `model`; scripts that create or validate the
training-ready shard files belong here.

## Current tree-ID-centered pipeline

First, normalize the raw city inventory files into stable, clean `tree_id`
records. This removes invalid coordinates and records sharing an exact
coordinate, assigns taxon labels, and writes the metadata consumed by the
tree-crown matching stage:

```powershell
python dataCollectionPreprocessing/Shard/build_clean_tree_record_metadata.py `
  --inventory-root D:/tree_inventories `
  --inventory-pattern "*_Final_*.csv" `
  --output-root H:/TreeCenteredModelInputs/tree_record_metadata_clean `
  --min-unique-coordinate-percent 50
```

Review `tree_record_metadata_clean_summary.csv` before continuing. Cities below
the configured unique-coordinate threshold are marked for exclusion and are
not selected by the crown spatial-join script.

After all crown-centered modality products have been created, calculate the
record-level QA table used to screen shard rows:

```powershell
python dataCollectionPreprocessing/Shard/apply_tree_centered_crop_qa.py `
  --crop-root H:/TreeCenteredModelInputs/tree_centered_naip_crops_clean `
  --structure-dir H:/TreeCenteredModelInputs/tree_centered_chm_structure_clean `
  --lidar-chm-root H:/TreeCenteredModelInputs/tree_centered_lidar_products_clean/CHM `
  --sentinel-phenology-dir H:/TreeCenteredModelInputs/tree_centered_sentinel_phenology_clean `
  --gee-dir H:/TreeCenteredModelInputs/tree_centered_gee_inputs_clean `
  --output-dir H:/TreeCenteredModelInputs/tree_centered_crop_qa_clean
```

This stage recomputes NAIP crop quality, central vegetation and LiDAR evidence,
checks required sidecars, and writes `<city>_tree_centered_qa_metadata.csv`.
The shard assembler uses `qa_exclude_from_model` from these files to omit
records that do not meet the current model-input requirements.

Use `assemble_clean_tree_id_centered_model_input_shards.py` for the current
clean crown-centered dataset. It aligns products by stable `tree_id` and writes
complete shards containing the required NAIP, LiDAR/structure, Sentinel-2,
satellite-embedding, and PRISM inputs.

```powershell
python dataCollectionPreprocessing/Shard/assemble_clean_tree_id_centered_model_input_shards.py `
  --qa-root H:/TreeCenteredModelInputs/tree_centered_crop_qa_clean `
  --crop-root H:/TreeCenteredModelInputs/tree_centered_naip_crops_clean `
  --structure-root H:/TreeCenteredModelInputs/tree_centered_chm_structure_clean `
  --sentinel-phenology-root H:/TreeCenteredModelInputs/tree_centered_sentinel_phenology_clean `
  --gee-root H:/TreeCenteredModelInputs/tree_centered_gee_inputs_clean `
  --prism-daily-root H:/TreeCenteredModelInputs/tree_centered_prism_daily_clean `
  --use-prism-daily-temperature `
  --prism-daily-sentinel-end-date 20231231 `
  --no-require-prism-normals `
  --output-root H:/TreeCenteredModelInputs/tree_centered_complete_sharded100k_prism_daily3_clean `
  --shard-size 100000 `
  --dry-run
```

Run with `--dry-run` first, inspect the discovered products and exclusion
counts, then remove the flag to write shards. The assembler writes a manifest
and summary alongside each city's NPZ shard files.

`run_assemble_clean_prism_daily_temperature_shards.cmd` is the saved Windows
runner for the current daily-temperature PRISM configuration. Extra command-line
arguments are forwarded to the assembler.

The current shards contain daily precipitation, mean temperature, and maximum
VPD aligned to the Sentinel sequence. They do not require or include the older
PRISM climate-normal sidecars.

Use the clean assembler's dry run and generated per-city summaries to verify
product alignment before training.

## Generated artifacts

Manifests are generated outputs rather than repository configuration files.
Keep them beside the shards they describe. Outputs include:

- `*_tree_id_centered_complete_shards_manifest.csv`
- per-city assembly summary JSON files

Do not move a generated manifest away from its shards; training and audit tools
use the manifest paths and shard-relative filenames to preserve row identity.

## Create compact shards for sharing

To build a smaller, directly trainable distribution without modifying the
canonical shards:

```powershell
python dataCollectionPreprocessing/Shard/create_shareable_compact_shards.py `
  --input-root H:/TreeCenteredModelInputs/tree_centered_complete_sharded100k_prism_daily3_clean `
  --output-root H:/ShareShards
```

The compact format stores 30x30 center NAIP crops, centimetre-scaled `uint16`
CHM arrays, `float16` continuous features, and boolean masks. It omits the
unused vegetation-CHM weight and duplicate standalone daily-PRISM arrays. The
active model loader reads the CHM scale metadata automatically.

Run a city first with `--city-token denver` and inspect
`compact_shard_conversion_report.json`. For one upload archive per city, add
`--archive-cities`; this requires the `zstandard` package.
