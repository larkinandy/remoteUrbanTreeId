# Tree-crown detection and inventory matching

This folder reproduces the boundary between high-resolution crown detection
and the clean municipal inventory records used by the crown-centered model.

<img src="https://github.com/larkinandy/remoteUrbanTreeId/blob/main/images/treeCrown.png" width="800">

## Reproducibility boundary

The study used the external
[UrbanTreeDetector](https://github.com/jonathanventura/urban-tree-detection)
repository at commit `9d6916af97da62906004c63bf675558e3fb0368b` with its
`pretrained_rgbn` model (60 cm, four-band NAIP). The exact model identity,
download URL, checksums, inference settings, and matching thresholds are frozen
under [`provenance`](provenance/README.md). The weights are not redistributed.

Create the detector environment with:

```powershell
conda env create --file dataCollectionPreprocessing/treeCrown/environment-tree-crown.yml
conda activate urban-tree-detection
```

Clone and check out the pinned external code, download the model archive, and
verify its weights before use:

```powershell
git clone https://github.com/jonathanventura/urban-tree-detection.git UrbanTreeDetector
git -C UrbanTreeDetector checkout 9d6916af97da62906004c63bf675558e3fb0368b
Get-FileHash $DATA_ROOT/models/pretrained_rgbn/weights.best.h5 -Algorithm SHA256
```

The expected hash is recorded in
`provenance/study_tree_crown_configuration.json`.

## 1. Detect crown centers

The final workflow operated on indexed, city-level NAIP chip stacks rather than
on GeoJSON converted after inference. Each city directory must contain exactly
one `*_chips.npy` array and one `*_index.csv` table. The index must contain
`chip_index`; projected crown coordinates are produced when it also contains
`cell_center_x`, `cell_center_y`, and `cell_epsg`. Optional cell and vegetation
metadata are retained in the output.

```powershell
python dataCollectionPreprocessing/treeCrown/detect_tree_crowns_with_urban_tree_detector.py `
  --detector-repo $DATA_ROOT/software/UrbanTreeDetector `
  --model-dir $DATA_ROOT/models/pretrained_rgbn `
  --input-root $DATA_ROOT/NAIP_Paired_Chips `
  --output-root $DATA_ROOT/detected_tree_crowns `
  --bands RGBN `
  --batch-size 128 `
  --pixel-size 0.6
```

The output is one `<city>_tree_centers.csv` per city. Required downstream
columns are `approx_x`, `approx_y`, `cell_epsg`, `confidence`, `chip_index`,
`x`, and `y`. The maintained workflow retains the actual detector confidence;
it does not substitute a constant proxy.

## 2. Match clean inventory records to crown centers

This step depends on the clean per-city inventory products created upstream and
uses their `tree_record_metadata_clean_summary.csv` to retain eligible cities.
It transforms inventory longitude/latitude into each crown table's projected
CRS and chooses the nearest crown within 5 m with confidence at least 0.10.

```powershell
python dataCollectionPreprocessing/treeCrown/spatial_join_clean_tree_records_to_detected_crowns.py `
  --tree-metadata-root $DATA_ROOT/tree_record_metadata_clean `
  --crown-root $DATA_ROOT/detected_tree_crowns `
  --output-root $DATA_ROOT/tree_to_detected_crowns `
  --match-radius-m 5 `
  --min-crown-confidence 0.10
```

Each inventory record receives at most one crown. A detected crown may be the
nearest qualifying crown for more than one inventory record; the frozen city
summary reports how often this occurred. A city with no matches receives a
header-only, schema-valid output rather than a missing or malformed file.

## 3. Downstream dependency

The resulting join tables are consumed by
`dataCollectionPreprocessing/NAIP/extract_tree_id_centered_naip_crops.py`, which
centers final NAIP samples on detected crown coordinates. Do not substitute the
original inventory coordinates when reproducing the study inputs.

## Frozen study summary

`provenance/tree_crown_city_summary.csv` contains city-level detection counts,
confidence ranges, matched-record counts, match-distance ranges, and crown reuse
counts from the final products. Regenerate it only when intentionally freezing
a new dataset version:

```powershell
python dataCollectionPreprocessing/treeCrown/freeze_tree_crown_provenance.py `
  --crown-root $DATA_ROOT/detected_tree_crowns `
  --match-root $DATA_ROOT/tree_to_detected_crowns `
  --output dataCollectionPreprocessing/treeCrown/provenance/tree_crown_city_summary.csv
```
