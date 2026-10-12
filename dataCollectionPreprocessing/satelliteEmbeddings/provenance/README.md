# Frozen satellite-embedding provenance

- `study_satellite_embedding_configuration.json` records the immutable data
  product, years, bands, spatial sampling, and sidecar rules.
- `satellite_embedding_sidecar_schema.json` defines every array written to a
  city sidecar.
- `frozen_study_satellite_embedding_city_summary.csv` reports the final-study
  coverage by city. The original and supplemental columns describe the
  historical transition to crown-centered extraction; a clean rebuild uses
  direct crown-cell exports for every row.
- `frozen_study_satellite_embedding_global_summary.json` provides corresponding
  study-wide totals.

Operational Earth Engine task manifests are not provenance artifacts. They
contain transient task identifiers and are intentionally ignored by Git.
