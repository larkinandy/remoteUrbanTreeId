import os
import pandas as pd
import numpy as np

# ============================================================
# USER SETTINGS
# ============================================================
OUT_DIR = r"C:/users/larki/Desktop/"

ROW_PRED_PATH = os.path.join(
    OUT_DIR,
    "hybrid_lstm_broad2_treefused_era5_test_row_predictions.csv"
)

TREE_PRED_PATH = os.path.join(
    OUT_DIR,
    "hybrid_lstm_broad2_treefused_era5_test_tree_predictions.csv"
)

ERROR_OUT_DIR = os.path.join(OUT_DIR, "model_error_analysis")
os.makedirs(ERROR_OUT_DIR, exist_ok=True)


# ============================================================
# LOAD
# ============================================================
row_df = pd.read_csv(ROW_PRED_PATH)
tree_df = pd.read_csv(TREE_PRED_PATH)

print("Row-level records:", len(row_df))
print("Tree-level records:", len(tree_df))


# ============================================================
# ROW-LEVEL ERROR FLAGS
# ============================================================
row_df["correct"] = row_df["y_true_name"] == row_df["y_pred_name"]
row_df["error_type"] = np.where(
    row_df["correct"],
    "correct",
    row_df["y_true_name"] + "_as_" + row_df["y_pred_name"]
)

prob_cols = [c for c in row_df.columns if c.startswith("prob_")]
row_df["max_prob"] = row_df[prob_cols].max(axis=1)

if "prob_conifer" in row_df.columns:
    row_df["prob_true_class"] = np.where(
        row_df["y_true_name"] == "conifer",
        row_df["prob_conifer"],
        row_df["prob_broadleaf"],
    )


# ============================================================
# TREE-LEVEL ERROR FLAGS
# ============================================================
tree_df["correct"] = tree_df["y_true_name"] == tree_df["y_pred_fused_name"]
tree_df["error_type"] = np.where(
    tree_df["correct"],
    "correct",
    tree_df["y_true_name"] + "_as_" + tree_df["y_pred_fused_name"]
)


# ============================================================
# 1. OVERALL ERROR SUMMARY
# ============================================================
print("\nRow-level error types:")
print(row_df["error_type"].value_counts())

print("\nTree-fused error types:")
print(tree_df["error_type"].value_counts())

row_df["error_type"].value_counts().to_csv(
    os.path.join(ERROR_OUT_DIR, "row_error_type_counts.csv")
)

tree_df["error_type"].value_counts().to_csv(
    os.path.join(ERROR_OUT_DIR, "tree_error_type_counts.csv")
)


# ============================================================
# 2. ERRORS BY GENUS
# ============================================================
genus_summary = (
    row_df
    .groupby("genus_name")
    .agg(
        n_rows=("uniqueID", "size"),
        row_accuracy=("correct", "mean"),
        mean_confidence=("max_prob", "mean"),
        mean_valid_measurements=("valid_measurements", "mean"),
    )
    .reset_index()
    .sort_values("row_accuracy")
)

print("\nLowest row-level accuracy by genus:")
print(genus_summary.head(20))

genus_summary.to_csv(
    os.path.join(ERROR_OUT_DIR, "row_accuracy_by_genus.csv"),
    index=False
)

genus_error_counts = (
    row_df[row_df["correct"] == False]
    .groupby(["genus_name", "error_type"])
    .size()
    .reset_index(name="n_errors")
    .sort_values("n_errors", ascending=False)
)

print("\nMost common genus-specific errors:")
print(genus_error_counts.head(30))

genus_error_counts.to_csv(
    os.path.join(ERROR_OUT_DIR, "genus_error_counts.csv"),
    index=False
)


# ============================================================
# 3. ERRORS BY YEAR
# ============================================================
year_summary = (
    row_df
    .groupby("year")
    .agg(
        n_rows=("uniqueID", "size"),
        row_accuracy=("correct", "mean"),
        mean_confidence=("max_prob", "mean"),
        mean_valid_measurements=("valid_measurements", "mean"),
    )
    .reset_index()
    .sort_values("year")
)

print("\nAccuracy by year:")
print(year_summary)

year_summary.to_csv(
    os.path.join(ERROR_OUT_DIR, "row_accuracy_by_year.csv"),
    index=False
)


# ============================================================
# 4. COVERAGE / VALID MEASUREMENT PATTERNS
# ============================================================
coverage_summary = (
    row_df
    .assign(
        coverage_bin=pd.qcut(
            row_df["valid_measurements"],
            q=5,
            duplicates="drop"
        )
    )
    .groupby("coverage_bin", observed=True)
    .agg(
        n_rows=("uniqueID", "size"),
        row_accuracy=("correct", "mean"),
        mean_confidence=("max_prob", "mean"),
        mean_valid_measurements=("valid_measurements", "mean"),
    )
    .reset_index()
)

print("\nAccuracy by valid-measurement bin:")
print(coverage_summary)

coverage_summary.to_csv(
    os.path.join(ERROR_OUT_DIR, "row_accuracy_by_valid_measurement_bin.csv"),
    index=False
)


# ============================================================
# 5. HIGH-CONFIDENCE ERRORS
# ============================================================
high_conf_errors = (
    row_df[(row_df["correct"] == False) & (row_df["max_prob"] >= 0.90)]
    .sort_values("max_prob", ascending=False)
)

print("\nHigh-confidence row-level errors:")
print(high_conf_errors.head(30))

high_conf_errors.to_csv(
    os.path.join(ERROR_OUT_DIR, "high_confidence_row_errors.csv"),
    index=False
)


# ============================================================
# 6. TREE-FUSED ERRORS WITH ROW-LEVEL CONTEXT
# ============================================================
tree_errors = tree_df[tree_df["correct"] == False].copy()

tree_error_rows = row_df.merge(
    tree_errors[["uniqueID", "y_true_name", "y_pred_fused_name", "error_type"]],
    on="uniqueID",
    how="inner",
    suffixes=("_row", "_tree")
)

tree_errors.to_csv(
    os.path.join(ERROR_OUT_DIR, "tree_fused_errors.csv"),
    index=False
)

tree_error_rows.to_csv(
    os.path.join(ERROR_OUT_DIR, "tree_fused_error_row_context.csv"),
    index=False
)

print("\nTree-fused errors:")
print(tree_errors.head(30))


# ============================================================
# 7. PER-GENUS TREE-LEVEL ERROR SUMMARY
# ============================================================
# Need genus name from row_df because tree_df may not include genus.
tree_genus = (
    row_df[["uniqueID", "genus_name"]]
    .drop_duplicates(subset=["uniqueID"])
)

tree_df2 = tree_df.merge(tree_genus, on="uniqueID", how="left")

tree_genus_summary = (
    tree_df2
    .groupby("genus_name")
    .agg(
        n_trees=("uniqueID", "size"),
        tree_accuracy=("correct", "mean"),
        n_errors=("correct", lambda x: int((~x).sum())),
    )
    .reset_index()
    .sort_values(["tree_accuracy", "n_trees"])
)

print("\nTree-level accuracy by genus:")
print(tree_genus_summary)

tree_genus_summary.to_csv(
    os.path.join(ERROR_OUT_DIR, "tree_accuracy_by_genus.csv"),
    index=False
)


# ============================================================
# 8. SAVE FULL ANNOTATED TABLES
# ============================================================
row_df.to_csv(
    os.path.join(ERROR_OUT_DIR, "annotated_row_predictions.csv"),
    index=False
)

tree_df2.to_csv(
    os.path.join(ERROR_OUT_DIR, "annotated_tree_predictions.csv"),
    index=False
)

print("\nSaved error analysis outputs to:")
print(ERROR_OUT_DIR)