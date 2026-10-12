# Frozen manuscript configuration

`final_k11_partition.npz` contains the taxon labels and zero-based assignments
for the 11 intermediate groups used by the stage-two classifier. It includes
579 taxa: the core discriminator taxa and the development-only extensions for
taxa with 50--499 records.

`final_grouping.json` records the partition checksum and the one-based mapping
from the 11 intermediate classes to the six groups reported in the manuscript.
Checkpoint probabilities are averaged before this mapping is applied.

The reported six-group mapping is the frozen balanced validation candidate:
`1,1,2,2,3,3,4,5,3,6,2`. It was selected to limit the largest record and
taxon shares while retaining classification performance. Higher-ranked but
more imbalanced exploratory collapse candidates are not manuscript groupings.

`core_k11_partition.npz`, `core_k11_group_prototypes.npz`, and
`final_k11_to_k6_collapse.npz` retain the core-taxon artifacts needed to
reproduce the independent discriminator evaluation. Their checksums are stored
in `final_grouping.json`.

These files are versioned inputs to the reported analyses. Do not regenerate
or reorder them during independent test evaluation.
