# Frozen tree-crown provenance

This directory records the detector, weights, peak-extraction parameters, and
matching thresholds used to create the manuscript products. The city summary is
descriptive provenance, not a runtime manifest. Its file columns are relative
names so the table remains portable.

- `study_tree_crown_configuration.json`: immutable study settings and checksums.
- `urban_tree_detector_params.yaml`: local-maximum parameters supplied with the pretrained model.
- `tree_crown_city_summary.csv`: frozen counts and quality summaries from the final products.

The external model weights are not redistributed. Download them from the URL in
the configuration and verify the SHA-256 checksum before inference.
