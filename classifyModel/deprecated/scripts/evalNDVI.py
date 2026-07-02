import os
import numpy as np
import pandas as pd


# ============================================================
# USER SETTINGS
# ============================================================
X_PATH = r"C:/users/larki/Desktop/PollenSense/xDataNormalized.csv"

TREE_PRED_PATH = r"C:/users/larki/Desktop/hybrid_lstm_broad2_treefused_era5_test_tree_predictions.csv"
ROW_PRED_PATH = r"C:/users/larki/Desktop/hybrid_lstm_broad2_treefused_era5_test_row_predictions.csv"

OUT_DIR = r"C:/users/larki/Desktop/PollenSense/model_error_screening_broad2_era5"
os.makedirs(OUT_DIR, exist_ok=True)

ID_COL = "uniqueID"
DATE_COL = "date"

BLUE_COL = "B2"
GREEN_COL = "B3"
RED_COL = "B4"
NIR_COL = "B8"

NDVI_SUMMER_THRESHOLDS = [0.30, 0.35, 0.40, 0.45, 0.50]
BRIGHTNESS_SUMMER_THRESHOLDS = [0.06, 0.08, 0.10, 0.12, 0.14, 0.16]


# ============================================================
# LOAD SENTINEL-2 DATA
# ============================================================
x = pd.read_csv(X_PATH)

x[ID_COL] = x[ID_COL].astype(str)
x[DATE_COL] = pd.to_datetime(x[DATE_COL])
x["year"] = x[DATE_COL].dt.year
x["month"] = x[DATE_COL].dt.month

for c in [BLUE_COL, GREEN_COL, RED_COL, NIR_COL]:
    x[c] = pd.to_numeric(x[c], errors="coerce")

eps = 1e-10

x["raw_NDVI"] = (
    (x[NIR_COL] - x[RED_COL]) /
    (x[NIR_COL] + x[RED_COL] + eps)
)

x["raw_visible_brightness"] = x[[BLUE_COL, GREEN_COL, RED_COL]].mean(axis=1)
x["raw_red_brightness"] = x[RED_COL]
x["raw_green_brightness"] = x[GREEN_COL]
x["raw_blue_brightness"] = x[BLUE_COL]


# ============================================================
# SUMMARIZE SENTINEL-2 BY TREE-YEAR
# ============================================================
summer = x[x["month"].isin([6, 7, 8])].copy()

summer_year = (
    summer
    .groupby([ID_COL, "year"], as_index=False)
    .agg(
        raw_NDVI_summer_mean=("raw_NDVI", "mean"),
        raw_NDVI_summer_min=("raw_NDVI", "min"),
        raw_NDVI_summer_max=("raw_NDVI", "max"),
        raw_NDVI_summer_std=("raw_NDVI", "std"),
        raw_visible_brightness_summer_mean=("raw_visible_brightness", "mean"),
        raw_visible_brightness_summer_max=("raw_visible_brightness", "max"),
        raw_red_brightness_summer_mean=("raw_red_brightness", "mean"),
        raw_green_brightness_summer_mean=("raw_green_brightness", "mean"),
        raw_blue_brightness_summer_mean=("raw_blue_brightness", "mean"),
        n_summer_obs=("raw_NDVI", "count"),
    )
)

all_year = (
    x
    .groupby([ID_COL, "year"], as_index=False)
    .agg(
        raw_NDVI_annual_mean=("raw_NDVI", "mean"),
        raw_NDVI_annual_min=("raw_NDVI", "min"),
        raw_NDVI_annual_max=("raw_NDVI", "max"),
        raw_visible_brightness_annual_mean=("raw_visible_brightness", "mean"),
        n_annual_obs=("raw_NDVI", "count"),
    )
)

tree_year_s2 = all_year.merge(
    summer_year,
    on=[ID_COL, "year"],
    how="left",
)


# ============================================================
# LOAD CURRENT MODEL PREDICTIONS
# ============================================================
tree_pred = pd.read_csv(TREE_PRED_PATH)
row_pred = pd.read_csv(ROW_PRED_PATH)

tree_pred[ID_COL] = tree_pred[ID_COL].astype(str)
row_pred[ID_COL] = row_pred[ID_COL].astype(str)

required_tree_cols = [
    ID_COL,
    "y_true_name",
    "y_pred_fused_name",
]

missing = [c for c in required_tree_cols if c not in tree_pred.columns]
if missing:
    raise ValueError(f"Tree prediction file missing columns: {missing}")

required_row_cols = [
    ID_COL,
    "year",
    "y_true_name",
    "y_pred_name",
    "prob_conifer",
    "prob_broadleaf",
]

missing = [c for c in required_row_cols if c not in row_pred.columns]
if missing:
    raise ValueError(f"Row prediction file missing columns: {missing}")

tree_pred["tree_correct"] = (
    tree_pred["y_true_name"] == tree_pred["y_pred_fused_name"]
)

tree_pred["tree_error_type"] = "correct"
tree_pred.loc[
    (tree_pred["y_true_name"] == "conifer") &
    (tree_pred["y_pred_fused_name"] == "broadleaf"),
    "tree_error_type"
] = "conifer_false_negative"

tree_pred.loc[
    (tree_pred["y_true_name"] == "broadleaf") &
    (tree_pred["y_pred_fused_name"] == "conifer"),
    "tree_error_type"
] = "broadleaf_false_positive"

row_pred["row_correct"] = (
    row_pred["y_true_name"] == row_pred["y_pred_name"]
)

row_pred["row_error_type"] = "correct"
row_pred.loc[
    (row_pred["y_true_name"] == "conifer") &
    (row_pred["y_pred_name"] == "broadleaf"),
    "row_error_type"
] = "conifer_false_negative"

row_pred.loc[
    (row_pred["y_true_name"] == "broadleaf") &
    (row_pred["y_pred_name"] == "conifer"),
    "row_error_type"
] = "broadleaf_false_positive"


# ============================================================
# MERGE ROW-LEVEL MODEL OUTPUTS WITH S2 SUMMARIES
# ============================================================
row_pred_small = row_pred[
    [
        ID_COL,
        "year",
        "genus_name",
        "valid_measurements",
        "y_true_name",
        "y_pred_name",
        "prob_conifer",
        "prob_broadleaf",
        "row_correct",
        "row_error_type",
    ]
].copy()

tree_year = tree_year_s2.merge(
    row_pred_small,
    on=[ID_COL, "year"],
    how="inner",
)

tree_pred_small = tree_pred[
    [
        ID_COL,
        "y_pred_fused_name",
        "tree_correct",
        "tree_error_type",
    ]
].drop_duplicates(ID_COL)

tree_year = tree_year.merge(
    tree_pred_small,
    on=ID_COL,
    how="left",
)

print("Merged tree-year records:", len(tree_year))
print("Unique trees:", tree_year[ID_COL].nunique())


# ============================================================
# TREE-LEVEL SUMMARY ACROSS YEARS
# ============================================================
tree_summary = (
    tree_year
    .groupby(
        [
            ID_COL,
            "genus_name",
            "y_true_name",
            "y_pred_fused_name",
            "tree_correct",
            "tree_error_type",
        ],
        as_index=False,
    )
    .agg(
        mean_prob_conifer=("prob_conifer", "mean"),
        max_prob_conifer=("prob_conifer", "max"),
        min_prob_conifer=("prob_conifer", "min"),
        mean_prob_broadleaf=("prob_broadleaf", "mean"),

        mean_raw_NDVI_summer_mean=("raw_NDVI_summer_mean", "mean"),
        min_raw_NDVI_summer_mean=("raw_NDVI_summer_mean", "min"),
        max_raw_NDVI_summer_mean=("raw_NDVI_summer_mean", "max"),
        mean_raw_NDVI_summer_std=("raw_NDVI_summer_std", "mean"),

        mean_raw_visible_brightness_summer_mean=("raw_visible_brightness_summer_mean", "mean"),
        max_raw_visible_brightness_summer_mean=("raw_visible_brightness_summer_mean", "max"),
        mean_raw_visible_brightness_summer_max=("raw_visible_brightness_summer_max", "mean"),

        mean_raw_red_brightness_summer_mean=("raw_red_brightness_summer_mean", "mean"),
        mean_raw_green_brightness_summer_mean=("raw_green_brightness_summer_mean", "mean"),
        mean_raw_blue_brightness_summer_mean=("raw_blue_brightness_summer_mean", "mean"),

        mean_raw_NDVI_annual_mean=("raw_NDVI_annual_mean", "mean"),
        mean_raw_visible_brightness_annual_mean=("raw_visible_brightness_annual_mean", "mean"),

        n_years=("year", "nunique"),
        n_years_with_summer_obs=("raw_NDVI_summer_mean", "count"),
        total_valid_measurements=("valid_measurements", "sum"),
    )
)


# ============================================================
# THRESHOLD SCREENING
# ============================================================
def summarize_combined_thresholds(
    df,
    ndvi_col,
    brightness_col,
    ndvi_thresholds,
    brightness_thresholds,
    level_name,
    error_col,
):
    rows = []
    base = df.dropna(subset=[ndvi_col, brightness_col]).copy()

    for ndvi_thr in ndvi_thresholds:
        for bright_thr in brightness_thresholds:
            tmp = base.copy()

            tmp["removed"] = (
                (tmp[ndvi_col] < ndvi_thr) &
                (tmp[brightness_col] > bright_thr)
            )

            grouped = (
                tmp
                .groupby(["y_true_name", error_col], as_index=False)
                .agg(
                    n_total=(ID_COL, "size"),
                    n_removed=("removed", "sum"),
                )
            )

            grouped["pct_removed"] = 100 * grouped["n_removed"] / grouped["n_total"]
            grouped["ndvi_metric"] = ndvi_col
            grouped["brightness_metric"] = brightness_col
            grouped["ndvi_threshold"] = ndvi_thr
            grouped["brightness_threshold"] = bright_thr
            grouped["level"] = level_name
            grouped["error_col"] = error_col
            grouped["rule"] = (
                f"{ndvi_col} < {ndvi_thr} AND "
                f"{brightness_col} > {bright_thr}"
            )

            rows.append(grouped)

    return pd.concat(rows, ignore_index=True)


tree_mean_screen = summarize_combined_thresholds(
    df=tree_summary,
    ndvi_col="mean_raw_NDVI_summer_mean",
    brightness_col="mean_raw_visible_brightness_summer_mean",
    ndvi_thresholds=NDVI_SUMMER_THRESHOLDS,
    brightness_thresholds=BRIGHTNESS_SUMMER_THRESHOLDS,
    level_name="tree_mean_across_years",
    error_col="tree_error_type",
)

tree_min_screen = summarize_combined_thresholds(
    df=tree_summary,
    ndvi_col="min_raw_NDVI_summer_mean",
    brightness_col="max_raw_visible_brightness_summer_mean",
    ndvi_thresholds=NDVI_SUMMER_THRESHOLDS,
    brightness_thresholds=BRIGHTNESS_SUMMER_THRESHOLDS,
    level_name="tree_min_ndvi_max_brightness_across_years",
    error_col="tree_error_type",
)

tree_year_screen = summarize_combined_thresholds(
    df=tree_year,
    ndvi_col="raw_NDVI_summer_mean",
    brightness_col="raw_visible_brightness_summer_mean",
    ndvi_thresholds=NDVI_SUMMER_THRESHOLDS,
    brightness_thresholds=BRIGHTNESS_SUMMER_THRESHOLDS,
    level_name="tree_year",
    error_col="row_error_type",
)

screen_summary = pd.concat(
    [tree_mean_screen, tree_min_screen, tree_year_screen],
    ignore_index=True,
)


# ============================================================
# SELECTIVITY SCORE FOR CONIFER FALSE NEGATIVES
# ============================================================
def add_selectivity_scores(summary):
    rows = []

    group_cols = [
        "level",
        "ndvi_metric",
        "brightness_metric",
        "ndvi_threshold",
        "brightness_threshold",
        "rule",
        "error_col",
    ]

    for keys, g in summary.groupby(group_cols, dropna=False):
        row = dict(zip(group_cols, keys))
        error_col = row["error_col"]

        def get_pct(label, error_type):
            m = (g["y_true_name"] == label) & (g[error_col] == error_type)
            if m.any():
                return float(g.loc[m, "pct_removed"].iloc[0])
            return np.nan

        def get_n(label, error_type, field):
            m = (g["y_true_name"] == label) & (g[error_col] == error_type)
            if m.any():
                return int(g.loc[m, field].iloc[0])
            return 0

        row["pct_removed_conifer_false_negative"] = get_pct(
            "conifer", "conifer_false_negative"
        )
        row["pct_removed_correct_conifer"] = get_pct(
            "conifer", "correct"
        )
        row["pct_removed_correct_broadleaf"] = get_pct(
            "broadleaf", "correct"
        )
        row["pct_removed_broadleaf_false_positive"] = get_pct(
            "broadleaf", "broadleaf_false_positive"
        )

        row["n_removed_conifer_false_negative"] = get_n(
            "conifer", "conifer_false_negative", "n_removed"
        )
        row["n_removed_correct_conifer"] = get_n(
            "conifer", "correct", "n_removed"
        )
        row["n_removed_correct_broadleaf"] = get_n(
            "broadleaf", "correct", "n_removed"
        )

        row["selectivity_vs_correct_broadleaf"] = (
            row["pct_removed_conifer_false_negative"] -
            row["pct_removed_correct_broadleaf"]
        )

        row["selectivity_vs_correct_conifer"] = (
            row["pct_removed_conifer_false_negative"] -
            row["pct_removed_correct_conifer"]
        )

        rows.append(row)

    return pd.DataFrame(rows)


selectivity_summary = add_selectivity_scores(screen_summary)

selectivity_summary = selectivity_summary.sort_values(
    [
        "selectivity_vs_correct_broadleaf",
        "selectivity_vs_correct_conifer",
        "pct_removed_conifer_false_negative",
    ],
    ascending=False,
)


# ============================================================
# ERROR-SPECIFIC TABLES
# ============================================================
conifer_false_negatives = tree_summary[
    tree_summary["tree_error_type"] == "conifer_false_negative"
].sort_values("mean_prob_conifer")

correct_conifers = tree_summary[
    (tree_summary["y_true_name"] == "conifer") &
    (tree_summary["tree_error_type"] == "correct")
].sort_values("mean_prob_conifer")

broadleaf_false_positives = tree_summary[
    tree_summary["tree_error_type"] == "broadleaf_false_positive"
].sort_values("mean_prob_conifer", ascending=False)


# ============================================================
# SAVE OUTPUTS
# ============================================================
screen_summary_path = os.path.join(
    OUT_DIR,
    "ndvi_brightness_threshold_screening_summary.csv"
)

selectivity_path = os.path.join(
    OUT_DIR,
    "ndvi_brightness_threshold_selectivity_summary.csv"
)

tree_year_path = os.path.join(
    OUT_DIR,
    "tree_year_s2_metrics_with_model_predictions.csv"
)

tree_summary_path = os.path.join(
    OUT_DIR,
    "tree_level_s2_metrics_with_model_predictions.csv"
)

conifer_fn_path = os.path.join(
    OUT_DIR,
    "conifer_false_negative_tree_summary.csv"
)

correct_conifer_path = os.path.join(
    OUT_DIR,
    "correct_conifer_tree_summary.csv"
)

broadleaf_fp_path = os.path.join(
    OUT_DIR,
    "broadleaf_false_positive_tree_summary.csv"
)

screen_summary.to_csv(screen_summary_path, index=False)
selectivity_summary.to_csv(selectivity_path, index=False)
tree_year.to_csv(tree_year_path, index=False)
tree_summary.to_csv(tree_summary_path, index=False)
conifer_false_negatives.to_csv(conifer_fn_path, index=False)
correct_conifers.to_csv(correct_conifer_path, index=False)
broadleaf_false_positives.to_csv(broadleaf_fp_path, index=False)


# ============================================================
# PRINT USEFUL VIEWS
# ============================================================
print("\nTree-level error counts:")
print(tree_summary["tree_error_type"].value_counts())

print("\nTree-level error counts by true class:")
print(pd.crosstab(tree_summary["y_true_name"], tree_summary["tree_error_type"]))

print("\nTop threshold rules for isolating conifer false negatives:")
print(selectivity_summary.head(30))

print("\nConifer false negatives with lowest mean conifer probability:")
print(conifer_false_negatives.head(30))

print("\nSaved:")
print(screen_summary_path)
print(selectivity_path)
print(tree_year_path)
print(tree_summary_path)
print(conifer_fn_path)
print(correct_conifer_path)
print(broadleaf_fp_path)