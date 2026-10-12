# Frozen PRISM provenance

- `study_prism_configuration.json` freezes the final model input settings.
- `frozen_study_prism_city_summary.csv` records mapping support and coverage by city.
- `frozen_study_prism_global_summary.json` records the retained array dimensions, dates, variables, mask coverage, and grid definition.
- `frozen_study_prism_archive_inventory.csv` is the checksum inventory of retained daily ZIP archives.
- `frozen_study_prism_provenance_status.json` states whether the source archive volume was available when provenance was frozen.

Honolulu lies outside the retained conterminous-US grid. Its 15,428 records
have zero valid PRISM mappings and remain explicitly marked missing.
