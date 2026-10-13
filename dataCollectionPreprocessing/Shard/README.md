# Model-input shard assembly

This folder is the final preprocessing boundary before model training. The
scripts normalize inventory records, calculate record-level QA, align all
modality products by stable `tree_id`, and write the NPZ shards consumed by the
discriminator and stage-two classifier.

The completed model-input shard dataset used in the study is available from
[Google Drive](https://drive.google.com/file/d/1w_dBZZpTgRVNfBL0-zrhhU7tJkKSsjJM/view?usp=sharing).
It contains the source NPZ shards rather than the optional, larger
memory-mapped performance sidecars.

All commands use an explicit data root. In PowerShell, define it for the local
workstation before running the examples:

```powershell
$DATA_ROOT = "<path-to-data-root>"
```

The Python environment is the repository-level `treeid-gpu` environment in
[`environment.yml`](../../environment.yml).

## Pipeline order and dependencies

1. `build_clean_tree_record_metadata.py` normalizes municipal inventories,
   removes invalid and duplicated coordinates, and assigns stable `tree_id`
   values.
2. [`../treeCrown/spatial_join_clean_tree_records_to_detected_crowns.py`](../treeCrown/spatial_join_clean_tree_records_to_detected_crowns.py)
   reads the metadata summary, excludes cities below the configured unique-
   coordinate threshold, and matches retained records to detected crowns.
3. The LiDAR, NAIP, Sentinel-2, satellite-embedding, and PRISM workflows create
   their tree-centered products. See the READMEs in the sibling folders.
4. `apply_tree_centered_crop_qa.py` calculates hard-exclusion and retained-
   warning fields.
5. `assemble_clean_tree_id_centered_model_input_shards.py` aligns the complete
   products and writes training-ready shards.

The historical metadata column `taxon_label` is a coarse preprocessing label
used to identify catch-all source records. It is **not** a discriminator taxon,
a remote-sensing-derived group, or a final model target. Discriminator labels
are derived from normalized scientific names during model training.

## 1. Normalize inventory records

```powershell
python -m dataCollectionPreprocessing.Shard.build_clean_tree_record_metadata `
  --inventory-root "$DATA_ROOT/inventories" `
  --inventory-pattern "*_Final_*.csv" `
  --output-root "$DATA_ROOT/tree_record_metadata_clean" `
  --min-unique-coordinate-percent 50
```

The 50% rule is evaluated at the city level and recorded in
`tree_record_metadata_clean_summary.csv`. This script does not silently remove
an entire city; the downstream crown spatial join reads the flag and applies
the city exclusion.

## 2. Calculate record-level QA

```powershell
python -m dataCollectionPreprocessing.Shard.apply_tree_centered_crop_qa `
  --crop-root "$DATA_ROOT/tree_centered_naip_crops_clean" `
  --structure-dir "$DATA_ROOT/tree_centered_chm_structure_clean" `
  --lidar-chm-root "$DATA_ROOT/tree_centered_lidar_products_clean/CHM" `
  --sentinel-phenology-dir "$DATA_ROOT/tree_centered_sentinel_phenology_clean" `
  --gee-dir "$DATA_ROOT/tree_centered_gee_inputs_clean" `
  --output-dir "$DATA_ROOT/tree_centered_crop_qa_clean"
```

Hard exclusions are stored in `qa_exclude_from_model`. Other conditions remain
available as warning fields for stratified evaluation. The exact study
thresholds are frozen in
[`provenance/study_shard_configuration.json`](provenance/study_shard_configuration.json).

## 3. Assemble the final shards

Run a dry run first:

```powershell
python -m dataCollectionPreprocessing.Shard.assemble_clean_tree_id_centered_model_input_shards `
  --qa-root "$DATA_ROOT/tree_centered_crop_qa_clean" `
  --crop-root "$DATA_ROOT/tree_centered_naip_crops_clean" `
  --structure-root "$DATA_ROOT/tree_centered_chm_structure_clean" `
  --sentinel-phenology-root "$DATA_ROOT/tree_centered_sentinel_phenology_clean" `
  --original-raw-sentinel-timeseries-root "$DATA_ROOT/sentinel2_timeseries_original" `
  --supplemental-raw-sentinel-timeseries-root "$DATA_ROOT/sentinel2_timeseries_supplemental" `
  --gee-root "$DATA_ROOT/tree_centered_gee_inputs_clean" `
  --prism-root "$DATA_ROOT/tree_centered_prism_normals_clean" `
  --prism-daily-root "$DATA_ROOT/tree_centered_prism_daily_clean" `
  --use-prism-daily `
  --prism-daily-sentinel-end-date 20231231 `
  --no-require-prism-normals `
  --output-root "$DATA_ROOT/tree_centered_complete_sharded100k_prism_daily3_clean" `
  --shard-size 100000 `
  --max-sentinel-sequence-length 80 `
  --dry-run
```

Inspect the discovered products and exclusion counts, then remove `--dry-run`
to write the shards. `--use-prism-daily-temperature` remains accepted as a
backward-compatible alias for `--use-prism-daily`; the branch contains daily
precipitation, mean temperature, and maximum vapor-pressure deficit.

The assembler writes uncompressed NPZ files by default, paired row metadata,
and generated summaries beside the data. Manifest paths are relative to the
shard root. Do not separate a manifest from the shards it describes.

Validate the completed shards against the canonical member, dtype, row-count,
and `tree_id` ordering contract:

```powershell
python -m dataCollectionPreprocessing.Shard.validate_model_input_shards `
  --input-root "$DATA_ROOT/tree_centered_complete_sharded100k_prism_daily3_clean"
```

Reusable discovery, screening, output, and schema logic is separated into
`shard_discovery.py`, `shard_screening.py`, `shard_output.py`, and
`shard_schema.py`; shared array and path utilities are in `shard_common.py`.

## Frozen study provenance

Tracked, path-neutral study records are under [`provenance/`](provenance/):

- final configuration and thresholds;
- inventory and duplicate-coordinate summary;
- QA exclusion summary;
- final city/shard manifest;
- NPZ member schema and expected dtypes.

Generated CSV and JSON files outside `provenance/` are intentionally ignored.

## Optional compact distribution

`create_shareable_compact_shards.py` creates a smaller distribution without
modifying the canonical shards:

```powershell
python -m dataCollectionPreprocessing.Shard.create_shareable_compact_shards `
  --input-root "$DATA_ROOT/tree_centered_complete_sharded100k_prism_daily3_clean" `
  --output-root "$DATA_ROOT/shareable_compact_shards"
```

This optional step crops NAIP to 30×30 pixels, encodes CHM at centimetre
resolution in `uint16`, converts selected continuous arrays to `float16`, and
adds format-version metadata. It is a distribution transformation, not the
canonical study shard-assembly step.
