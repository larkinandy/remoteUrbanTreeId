# Results figure sources

`confusion-matrix.json` transcribes all 36 counts from the manuscript figure
supplied by the user on 2026-10-17. Rows are labeled groups and columns are
predicted groups. The counts sum to 153,529, identifying the height-screened
test subset, rather than the 161,508-record full test dataset. Displayed row
percentages are recalculated from integer counts. The largest cross-group
flows are Group 2 to Group 3 (8,213) and Group 3 to Group 2 (8,366).
The matrix is static; counts and percentages remain visible without interaction.

`geographic-results.json` preserves the 66 rows from
`analysis/geographic/provenance/final_training_regime_by_target_city.csv`:
22 target cities under each of three regimes. Accuracy, macro-F1, and
weighted-F1 are reported for collapsed six-group predictions. The source table
averages eligible source-model results within target city for the one-city
regime; these are not 147 independent city observations. See
`analysis/geographic/provenance/final_geographic_analysis_summary.json` and
`analysis/geographic/README.md` for eligibility and QA-screening rules.

Box plots use linearly interpolated quartiles, median center lines, and whiskers
ending at the most extreme observation within 1.5 IQR. Every city is plotted,
including outliers. Jitter is deterministic and has no analytical meaning.
Hover or keyboard focus compares one target city across the three regimes;
selection highlights that city across all metrics.
Pointer tooltips show the hovered city, training regime, and the corresponding
metric value (accuracy as a percentage, F1 scores to three decimal places).

The two tables are embedded in `storyboard/story.html` under `sb-results-data`.
`storyboard/results.js` is also embedded so the app and browser copies operate
without external requests. Keep these embedded copies synchronized when editing
the source tables or interaction script.
