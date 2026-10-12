# Frozen Sentinel-2 provenance

These files record the path-independent configuration and output summaries for
the Sentinel-2 products used in the study. Generated operational manifests and
local data paths are intentionally excluded.

- `study_sentinel2_configuration.json`: source collections, masks, dates,
  spatial scale, interpolation, and output contract.
- `frozen_study_sentinel2_city_summary.csv`: record and missingness counts by
  city.
- `frozen_study_sentinel2_global_summary.json`: aggregate counts and frozen
  phenology feature names.
- `frozen_study_sentinel2_provenance_status.json`: inventory-completeness status
  for final sidecars and retained raw exports.
