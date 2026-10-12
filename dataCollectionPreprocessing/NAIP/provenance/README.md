# Frozen NAIP provenance

This directory contains study-specific scientific settings and source
provenance. Files here are tracked even though generated CSV and JSON files in
the parent NAIP workflow are ignored.

- `study_naip_configuration.json` freezes the settings used to build the
  manuscript RGBNIR crops and identifies the multiband city exceptions.
- `county_boundaries_provenance.json` identifies and checksums the exact county
  boundary file used by the workflow. Its original download source and vintage
  were not retained, which is explicitly recorded rather than reconstructed by
  assumption.
- `study_naip_city_sources.csv` summarizes the retained per-city extraction
  configurations.
- `study_naip_source_rasters.csv` inventories the source raster and band
  combinations recorded in the retained crop metadata.
- `study_naip_county_manifest.json` reconstructs the counties represented by
  the exact retained source rasters. The original mutable downloader selection
  manifest and Box URLs were not retained; this limitation is not filled with
  current, potentially changed catalog URLs.
- `study_naip_provenance_summary.json` records the provenance-freezing inputs
  and totals.

The last three files are regenerated with `freeze_study_naip_provenance.py`
from the retained extraction outputs. They contain no workstation paths.
