# Frozen manuscript configuration

`final_k11_partition.npz` contains the taxon labels and zero-based assignments
for the 11 intermediate groups used by the stage-two classifier. It includes
579 taxa: 280 core discriminator taxa and 299 development-only extensions for
taxa with 50–499 records.

`final_grouping.json` records the partition checksum and the one-based mapping
from the 11 intermediate classes to the six groups reported in the manuscript.
Checkpoint probabilities are averaged before this mapping is applied.

Group assignments in every NPZ array are zero-based. Group identifiers in the
manuscript, figures, the frozen CSV table, and `collapse_map_one_based` are
one-based. The stable display labels for both label spaces are stored in
`final_grouping.json`; group numbers are identifiers rather than ranks.

The reported six-group mapping is the frozen balanced validation candidate:
`1,1,2,2,3,3,4,5,3,6,2`. It was selected to limit the largest record and
taxon shares while retaining classification performance. Higher-ranked but
more imbalanced exploratory collapse candidates are not manuscript groupings.

`core_k11_partition.npz`, `core_k11_group_prototypes.npz`, and
`final_k11_to_k6_collapse.npz` retain the core-taxon artifacts needed to
reproduce the independent discriminator evaluation. Their checksums are stored
in `final_grouping.json`.

`frozen_taxon_group_mapping.csv` is the human-readable equivalent of the
binary mapping. It reports each taxon's provenance, intermediate `k11` group,
reported `collapsed_k6` group, and the applicable development or held-out
support counts.

The 50-record rare-taxon threshold was applied before the deterministic 80%
development and 20% held-out split. Therefore, the smallest eligible taxon has
40 development records and 10 held-out records even though its total support
is 50.

## NPZ schemas

All arrays currently stored in these NPZ files have non-object NumPy dtypes.
They can and should be loaded with `numpy.load(path, allow_pickle=False)`.

`core_k11_partition.npz`

- `assignments`: zero-based `k11` assignment for each of the 280 core taxa.
- `label_columns`: ordered scientific-label tokens corresponding to rows.
- `validation_counts`: validation records used to estimate each core centroid.
- `full_validation_centroids`: 280 × 64 taxon centroids from validation data.

`core_k11_group_prototypes.npz`

- `group_centroids`: 11 × 64 frozen intermediate-group prototypes.
- `label_columns`, `assignments`, and `validation_counts`: redundant core-taxon
  identifiers used to audit prototype alignment.

`final_k11_partition.npz`

- `assignments`: zero-based `k11` assignments for all 579 taxa.
- `group_centroids`: the 11 × 64 core-derived group prototypes.
- `label_columns`: ordered core taxa followed by rare-extension taxa.
- `core_validation_counts`: validation support for the first 280 taxa.
- `rare_development_counts` and `rare_test_counts`: deterministic split support
  for the following 299 rare taxa.
- `core_taxon_count` and `rare_extension_taxon_count`: scalar boundary values.

`final_k11_to_k6_collapse.npz`

- `initial_group_assignment`: zero-based mapping from 11 intermediate groups
  to six reported groups.
- `initial_group_names`: stable source-group identifiers.
- `label_columns`: the same ordered 579 taxa as the final partition.
- `label_group`: zero-based reported group for every taxon.

`artifact_manifest.csv` lists each frozen artifact's role, byte size, and
SHA-256 digest. `validate_config.py` verifies the manifest as well as the
relationships among the artifacts.

Run the configuration validator from the repository root after cloning or
whenever one of these artifacts changes:

```powershell
python config/validate_config.py
```

The validator checks file digests, required NPZ schemas, dimensions and group
ranges, exact taxon ordering, core-prefix identity, prototype consistency,
the JSON/NPZ collapse mapping, and every row in the frozen CSV table. Use
`--write-table --overwrite` only when deliberately regenerating the table from
already validated binary artifacts.

These files are versioned inputs to the reported analyses. Do not regenerate
or reorder them during independent test evaluation.
