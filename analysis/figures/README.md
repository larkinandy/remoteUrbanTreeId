# Publication figure workflows

This directory contains the tracked scripts and compact frozen inputs used for
the manuscript and supplementary figures. Generated SVG, PDF, and PNG files
remain under `outputs/` and are not version controlled.

`figure_manifest.csv` identifies each retained figure, its status, generating
script, inputs, and default output. `figure_grouping.json` contains display-only
group ordering and colors. The scientific k=11-to-k=6 collapse is always read
from `../../config/final_grouping.json`.

## Frozen inputs

Small publication-facing inputs are stored in `provenance/`; their sizes and
SHA-256 checksums are recorded in `provenance/manifest.csv`. Large model
predictions, embeddings, and complete shards remain external study artifacts.
Scripts that require them expose explicit command-line paths.

The prototype-margin tables were regenerated using rare-taxon centroids from
development records only. Core-taxon count and order are read from the frozen
partition rather than assumed in code.

## Examples

From the repository root:

```powershell
python analysis/figures/plot_combined_discrimination_space.py

python analysis/figures/plot_gate_weight_multipanel.py

python analysis/figures/plot_probability_error_vs_match_distance.py `
  --group-space collapsed_k6 `
  --shard-root "$DATA_ROOT/tree_centered_complete_shards"

python analysis/figures/plot_taxon_error_vs_margin.py `
  --group-space collapsed_k6 `
  --shard-root "$DATA_ROOT/tree_centered_complete_shards"
```

The discrimination-space renderer uses `--group-space collapsed_k6` for the
reported six-group view and `--hybrid-legend` for the canonical hierarchy
legend. The latter can be combined with the documented manuscript group order.

## Terminology

- `k11`: intermediate model label space.
- `collapsed_k6`: final reporting space obtained by summing ensembled k=11
  probabilities according to the frozen collapse.
- Core taxa: taxa used to train the discriminator.
- Rare-extension taxa: taxa assigned after discriminator training using
  development-only centroids.
