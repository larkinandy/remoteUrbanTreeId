# Frozen shard provenance

These files document the inventory filtering, QA, and final shard assembly
used for the study. They are frozen repository artifacts, not live outputs.
Paths in the tables are relative product identifiers so the records remain
portable. Generated summaries elsewhere in `Shard/` remain ignored.

The CSV tables were frozen from the completed study outputs. The JSON files
record the analytical settings and expected NPZ interface. Large NPZ files are
distributed separately; their per-shard relative paths and record counts are
listed in the frozen manifest, with an additional city-level count table.
`frozen_study_shard_checksums.csv` records file sizes and SHA-256 hashes for
all 44 NPZ shards and all 44 paired metadata files.
