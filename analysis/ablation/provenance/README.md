# Frozen ablation results

These compact tables are the publication-facing summaries generated from the
saved validation and independent-test predictions after applying the canonical
balanced collapse `1,1,2,2,3,3,4,5,3,6,2`.

- `single_modality_validation_summary.csv`: validation performance used to
  describe modality sufficiency during development.
- `single_modality_test_summary.csv`: independent-test single-modality results.
- `independent_discriminator_test_summary.csv`: independent discriminator
  evaluation for the frozen k=11 and collapsed k=6 groups.
- `final_pruned_leave_one_out_paired_bootstrap.csv`: final-pruned leave-one-out
  effects and paired-bootstrap confidence intervals for the six retained
  branches.

The full record-level predictions and trained checkpoints are external study
artifacts and are not stored in Git.

`split_manifest.json` records the file name, schema, record count, and SHA-256
checksum of the exact inherited split manifest. The manifest contains record
keys and split labels derived from the restricted modeling data, so it remains
an external study artifact; every reproduction must verify the checksum before
running a comparison.
