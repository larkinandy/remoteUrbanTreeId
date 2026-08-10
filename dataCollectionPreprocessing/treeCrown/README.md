# Tree-crown detection and inventory matching

This folder covers the preprocessing boundary between detected tree crowns and
the clean inventory records used by the crown-centered model.

<img src = https://github.com/larkinandy/remoteUrbanTreeId/blob/main/images/treeCrown.png width="800">

## Step 1: detect tree crowns

Clone the UrbanTreeDetector repository and create its TensorFlow 2.4.1 conda
environment. Download one of its pretrained model archives; the extracted log
directory must contain `weights.best.h5` and may contain `params.yaml`.

Run the maintained wrapper in this folder against a NAIP TIFF or a directory of
NAIP TIFFs:

```powershell
conda activate urban-tree-detection
python dataCollectionPreprocessing/treeCrown/detect_tree_crowns_with_urban_tree_detector.py `
  --detector-repo C:/path/to/urban-tree-detection `
  --input H:/NAIP/city_tiles `
  --city-token losangeles `
  --log-dir C:/path/to/downloaded/model_log `
  --output-root H:/TreeCenteredModelInputs/detected_tree_crowns_clean `
  --bands RGBN `
  --cell-epsg 26911
```

The official detector produces georeferenced point GeoJSON files. This wrapper
consolidates them into the following pipeline input:

The current matching script expects one crown-center CSV per city:

```text
H:/TreeCenteredModelInputs/detected_tree_crowns_clean/
  <city>_tree_centers.csv
```

Each file must contain:

- `approx_x` and `approx_y`: projected crown-center coordinates
- `cell_epsg`: EPSG code for those coordinates
- `confidence`: tree-detector confidence score

Tree detection is performed by the external
[UrbanTreeDetector](https://github.com/jonathanventura/urban-tree-detection),
while the local wrapper owns orchestration and conversion into this repository's
stable CSV interface. The detector's current GeoJSON does not include peak
confidence, so the wrapper writes `confidence=1.0` plus
`confidence_is_proxy=true`. Do not interpret that compatibility value as a
calibrated model score.

The spatial-join script in this folder has been reconstructed around the
detector CSV format and the input schema required by the current NAIP crop
pipeline. It writes a header-only, schema-valid CSV when a city has no matches,
so downstream processing can distinguish an empty result from a damaged file.

## Step 2: match inventory records to crowns

After producing the city crown-center CSVs, match each clean inventory point to
the nearest sufficiently confident detected crown:

```powershell
python dataCollectionPreprocessing/treeCrown/spatial_join_clean_tree_records_to_detected_crowns.py `
  --tree-metadata-root H:/TreeCenteredModelInputs/tree_record_metadata_clean `
  --crown-root H:/TreeCenteredModelInputs/detected_tree_crowns_clean `
  --output-root H:/TreeCenteredModelInputs/tree_to_detected_crowns_clean `
  --match-radius-m 5 `
  --min-crown-confidence 0.10
```

The script transforms inventory coordinates into each crown dataset's projected
CRS, finds the nearest crown within the configured radius, and writes the crown
coordinates, confidence, distance, and identifiers keyed by `tree_id`.

These join CSVs are consumed by the NAIP crop scripts so imagery is centered
on detected crowns rather than potentially offset inventory coordinates.
