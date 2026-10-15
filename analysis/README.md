# Manuscript analyses

This directory contains the analyses retained for the submitted-manuscript snapshot.
They consume frozen partitions, saved checkpoints, or exported predictions;
they do not redefine the study labels from independent test results.

- `ablation/` contains leave-one-modality-out, single-modality, pruning, and
  paired-bootstrap analyses.
- `geographic/` contains leave-one-city-out and one-city-trained transfer
  analyses.
- `sensitivity/` contains post hoc test-set strata and gate-weight
  summaries.
- `figures/` contains scripts for manuscript figures derived from saved
  numerical outputs.

Machine-specific data and output locations shown by `--help` can be overridden
with command-line arguments. Generated outputs are intentionally excluded from
version control.

Each analysis subdirectory contains a README describing its reporting rules and
frozen provenance. [`figures/figure_manifest.csv`](figures/figure_manifest.csv)
maps publication figures and components to their generating scripts and source
tables. Compact final numerical products are versioned under the corresponding
`provenance/` directory; large predictions, checkpoints, and record-level tables
remain external artifacts.
