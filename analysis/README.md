# Manuscript analyses

This directory contains the analyses retained for the manuscript snapshot.
They consume frozen partitions, saved checkpoints, or exported predictions;
they do not redefine the study labels from independent test results.

- `ablation/` contains leave-one-modality-out, single-modality, pruning, and
  paired-bootstrap analyses.
- `geographic/` contains leave-one-city-out and one-city-trained transfer
  analyses.
- `sensitivity/` contains prespecified test-set strata and gate-weight
  summaries.
- `figures/` contains scripts for manuscript figures derived from saved
  numerical outputs.

Machine-specific data and output locations shown by `--help` can be overridden
with command-line arguments. Generated outputs are intentionally excluded from
version control.
