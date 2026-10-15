# Publication figure workflows

This directory contains the tracked scripts and compact frozen inputs used for
the manuscript and supplementary figures. Generated SVG, PDF, and PNG files
remain under `outputs/` and are not version controlled.

`figure_manifest.csv` identifies each retained figure, its status, generating
script, inputs, and default output. `figure_grouping.json` contains display-only
group ordering and colors. The scientific k11-to-collapsed_k6 mapping is always read
from `../../config/final_grouping.json`.

## Frozen inputs

Small publication-facing inputs are stored in `provenance/`; their sizes and
SHA-256 checksums are recorded in `provenance/manifest.csv`. Large model
predictions, embeddings, and complete shards remain external study artifacts.
Scripts that require them expose explicit command-line paths.

Shared loading, boolean parsing, and export behavior is implemented in
`figure_data.py`. Matplotlib figures are exported consistently as SVG, PDF,
and 600-dpi PNG files. The custom discrimination-space SVG renderers retain
SVG as their canonical editable output.

The prototype-margin tables were regenerated using rare-taxon centroids from
development records only. Core-taxon count and order are read from the frozen
partition rather than assumed in code.

## Examples

From the repository root:

```powershell
python analysis/figures/plot_combined_discrimination_space.py

python analysis/figures/plot_gate_weight_multipanel.py

python analysis/figures/plot_single_modality_performance.py

python analysis/figures/plot_ablation_performance.py

python analysis/figures/plot_geographic_training_regime_boxplots.py

python analysis/figures/plot_probability_error_vs_match_distance.py `
  --group-space collapsed_k6 `
  --predictions "$DATA_ROOT/final_model/ensemble_predictions.npz" `
  --shard-root "$DATA_ROOT/tree_centered_complete_shards"

python analysis/figures/plot_taxon_error_vs_margin.py `
  --group-space collapsed_k6 `
  --predictions "$DATA_ROOT/final_model/ensemble_predictions.npz" `
  --rare-embeddings "$DATA_ROOT/discriminator/rare_taxon_embeddings.npz" `
  --shard-root "$DATA_ROOT/tree_centered_complete_shards"
```

The discrimination-space renderer uses `--group-space collapsed_k6` for the
reported `collapsed_k6` view and `--hybrid-legend` for the canonical hierarchy
legend. The latter can be combined with the documented manuscript group order.

The submitted four-panel modality figure combines the two-panel gate-weight
output with the single-modality and ablation panels. Each component is generated
from a tracked, checksummed source table. The final layout was assembled without
altering the plotted values.

The Sentinel-2 interpolation script requires explicit paths to the final split
manifest, interpolation summaries, and complete shards; no workstation path or
superseded partition is assumed. Its companion Node renderer accepts the
prepared output directory as its first positional argument.

## Terminology

- `k11`: intermediate model label space.
- `collapsed_k6`: final reporting space obtained by summing ensembled k11
  probabilities according to the frozen collapse.
- Core taxa: taxa used to train the discriminator.
- Rare-extension taxa: taxa assigned after discriminator training using
  development-only centroids.
