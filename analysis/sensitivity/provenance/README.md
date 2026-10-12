# Frozen sensitivity provenance

`final_sensitivity_table_collapsed_k6.csv` contains the publication-facing
results regenerated from the finalized independent-test ensemble.
`input_checksums.csv` identifies the external source artifacts. Paths are
descriptive only; reproducibility depends on the checksums and portable command
documented in the parent README.

The reproducibility pass corrected two earlier ad-hoc joins: Sentinel-2
interpolation is now read directly from each record's shard rather than joined
through a positional array, and the NAIP vegetation strata follow the actual
`naip_vegetation_fraction` inequality. The frozen table is therefore the
authoritative result.
