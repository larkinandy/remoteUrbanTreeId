import os
import numpy as np
import pandas as pd
from sklearn.neighbors import KDTree
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# ============================================================
# Paths
# ============================================================
MIX_TREE_PATH = r"C:/users/larki/Desktop/mixTreeTest.csv"

TRAIN_PRED_PATH = r"C:/users/larki/Desktop/hybrid_lstm_broad2_treefused_era5_train_tree_predictionsv5.csv"
VAL_PRED_PATH   = r"C:/users/larki/Desktop/hybrid_lstm_broad2_treefused_era5_val_tree_predictionsv5_seq1_ann1_height0.csv"
TEST_PRED_PATH  = r"C:/users/larki/Desktop/hybrid_lstm_broad2_treefused_era5_test_tree_predictionsv5_seq1_ann1_height0.csv"

OUT_DIR = r"C:/users/larki/Desktop/tree_binary_QA_mixed_removed_threshold_sweep"
os.makedirs(OUT_DIR, exist_ok=True)

OUT_MERGED_PATH = os.path.join(OUT_DIR, "all_tree_predictions_with_coords.csv")
OUT_NEIGHBOR_PATH = os.path.join(OUT_DIR, "all_tree_neighbor_QA_base_table.csv")
OUT_SWEEP_PATH = os.path.join(OUT_DIR, "QA_threshold_sweep_results.csv")
OUT_BEST_QA_PATH = os.path.join(OUT_DIR, "best_QA_threshold_predictions_with_flags.csv")
OUT_RETAINED_PATH = os.path.join(OUT_DIR, "best_QA_binary_eval_records_retained.csv")
OUT_REMOVED_PATH = os.path.join(OUT_DIR, "best_QA_records_removed.csv")
OUT_SUMMARY_PATH = os.path.join(OUT_DIR, "best_QA_binary_performance_summary.csv")
OUT_REPORT_PATH = os.path.join(OUT_DIR, "best_QA_binary_performance_report.csv")
OUT_CONFUSION_PATH = os.path.join(OUT_DIR, "best_QA_binary_performance_confusion.csv")
OUT_QA_SUMMARY_PATH = os.path.join(OUT_DIR, "best_QA_filter_summary.csv")
OUT_LOADED_FILES_PATH = os.path.join(OUT_DIR, "prediction_files_loaded_status.csv")

# ============================================================
# Settings
# ============================================================
ID_COL = "uniqueID"
X_COL = "X_COORD"
Y_COL = "Y_COORD"
GENUS_COL = "BOTANICALG"
PRED_COL = "y_pred_fused"

RADIUS_M = 20.0
COORD_UNITS = "feet"
RADIUS = RADIUS_M * 3.280833333 if COORD_UNITS == "feet" else RADIUS_M

# ============================================================
# QA threshold sweep settings
# ============================================================
# True-neighborhood QA flag:
# remove if:
#   n_neighbors >= min_total_neighbors_for_QA
#   AND n_true_opposite_neighbors >= min_true_opposite_neighbors_for_QA
#   AND true_opposite_neighbor_rate >= min_true_opposite_rate_for_QA
MIN_TOTAL_NEIGHBORS_FOR_QA_GRID = [1, 2, 3]
MIN_TRUE_OPPOSITE_NEIGHBORS_FOR_QA_GRID = [1, 2, 3]
MIN_TRUE_OPPOSITE_RATE_FOR_QA_GRID = [0.0, 0.10, 0.20, 0.30, 0.40]

# Optional prediction-disagreement QA flag:
# remove additionally if:
#   n_neighbors >= min_pred_neighbors_for_QA
#   AND (
#       n_pred_disagree_neighbors >= min_pred_disagree_neighbors_for_QA
#       OR neighbor_prediction_disagreement_rate >= min_pred_disagree_rate_for_QA
#   )
USE_PREDICTED_QA_GRID = [False, True]
MIN_PRED_NEIGHBORS_FOR_QA_GRID = [1, 2, 3]
MIN_PRED_DISAGREE_NEIGHBORS_FOR_QA_GRID = [1, 2, 3]
MIN_PRED_DISAGREE_RATE_FOR_QA_GRID = [0.10, 0.20, 0.30, 0.40, 0.50]

# Optimization target:
# - "non_training" is recommended for choosing thresholds.
# - If no non-training data are available, script falls back to training.
OPTIMIZE_ON_TRAIN_STATUS = "non_training"

# Metric to optimize.
# Good options: "macro_f1", "conifer_f1", "accuracy", "weighted_f1"
OPTIMIZE_METRIC = "macro_f1"

# Avoid choosing thresholds that remove too much data.
# These constraints are applied to the optimization group only.
MAX_REMOVED_FRACTION = 0.50
MIN_EVALUATED_RECORDS = 100

# ============================================================
# Label mappings
# ============================================================
CONIFER_GENERA = {"Picea", "Pinus"}

BROADLEAF_GENERA = {
    "Acer", "Tilia", "Ulmus", "Zelkova", "Gleditsia", "Carya", "Juglans",
    "Carpinus", "Celtis", "Cercis", "Cladrastis", "Gymnocladus",
    "Liriodendron", "Prunus", "Quercus", "Ginkgo", "Pyrus",
}

IDX_TO_LABEL = {0: "conifer", 1: "broadleaf"}
BINARY_CLASS_NAMES = ["conifer", "broadleaf"]


# ============================================================
# Helpers
# ============================================================
def genus_to_label(genus):
    genus = str(genus).strip()
    if genus in CONIFER_GENERA:
        return 0
    if genus in BROADLEAF_GENERA:
        return 1
    return np.nan


def standardize_unique_id_column(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "unique_id" in df.columns and "uniqueID" not in df.columns:
        df = df.rename(columns={"unique_id": "uniqueID"})
    if "uniqueId" in df.columns and "uniqueID" not in df.columns:
        df = df.rename(columns={"uniqueId": "uniqueID"})
    return df


def load_pred_file(path, split_name):
    df = pd.read_csv(path).copy()
    df = standardize_unique_id_column(df)

    if "uniqueID" not in df.columns:
        raise ValueError(f"{path} is missing uniqueID, unique_id, or uniqueId")
    if PRED_COL not in df.columns:
        raise ValueError(f"{path} is missing {PRED_COL}")

    df["split"] = split_name
    df["train_status"] = "training" if split_name == "train" else "non_training"

    if df["uniqueID"].duplicated().any():
        print(f"Warning: duplicate uniqueID values in {split_name}; keeping first row per tree.")
        df = df.sort_values("uniqueID").drop_duplicates(subset=["uniqueID"], keep="first")

    return df


def maybe_load_pred_file(path, split_name, required=False):
    status = {
        "split": split_name,
        "train_status": "training" if split_name == "train" else "non_training",
        "path": path,
        "required": required,
        "loaded": False,
        "n_rows": 0,
        "message": "",
    }

    if not os.path.exists(path):
        status["message"] = "missing"
        if required:
            raise FileNotFoundError(f"Required prediction file is missing: {path}")
        print(f"Skipping missing optional prediction file for {split_name}:\n  {path}")
        return None, status

    try:
        df = load_pred_file(path, split_name)
    except Exception as e:
        status["message"] = f"error: {e}"
        if required:
            raise
        print(f"Skipping optional prediction file for {split_name} because it could not be loaded:")
        print(f"  {path}")
        print(f"  Error: {e}")
        return None, status

    status["loaded"] = True
    status["n_rows"] = int(len(df))
    status["message"] = "loaded"
    print(f"Loaded {split_name}: {len(df)} rows")
    return df, status


def report_to_df(report_dict, group_name, task_name, extra=None):
    rows = []
    extra = extra or {}
    for label, metrics in report_dict.items():
        if isinstance(metrics, dict):
            rows.append({
                **extra,
                "task": task_name,
                "train_status": group_name,
                "class_or_average": label,
                "precision": metrics.get("precision", np.nan),
                "recall": metrics.get("recall", np.nan),
                "f1_score": metrics.get("f1-score", np.nan),
                "support": metrics.get("support", np.nan),
            })
        else:
            rows.append({
                **extra,
                "task": task_name,
                "train_status": group_name,
                "class_or_average": label,
                "precision": np.nan,
                "recall": np.nan,
                "f1_score": metrics,
                "support": np.nan,
            })
    return pd.DataFrame(rows)


def evaluate_binary_by_train_status(df, true_col, pred_col, task_name, extra=None, verbose=True):
    summary_rows = []
    report_dfs = []
    confusion_dfs = []
    extra = extra or {}

    for group_name, g in df.groupby("train_status", sort=False):
        if len(g) == 0:
            continue

        y_true = g[true_col].to_numpy(dtype=int)
        y_pred = g[pred_col].to_numpy(dtype=int)

        report_dict = classification_report(
            y_true,
            y_pred,
            labels=[0, 1],
            target_names=BINARY_CLASS_NAMES,
            output_dict=True,
            zero_division=0,
        )

        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        cm_df = pd.DataFrame(
            cm,
            index=[f"true_{c}" for c in BINARY_CLASS_NAMES],
            columns=[f"pred_{c}" for c in BINARY_CLASS_NAMES],
        )

        cm_long = cm_df.reset_index().rename(columns={"index": "true_class"})
        cm_long = cm_long.melt(
            id_vars="true_class",
            var_name="predicted_class",
            value_name="n",
        )
        cm_long.insert(0, "train_status", group_name)
        cm_long.insert(0, "task", task_name)
        for k, v in reversed(list(extra.items())):
            cm_long.insert(0, k, v)
        confusion_dfs.append(cm_long)

        report_dfs.append(report_to_df(report_dict, group_name, task_name, extra=extra))

        summary_row = {
            **extra,
            "task": task_name,
            "train_status": group_name,
            "n_evaluated": int(len(g)),
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "macro_precision": report_dict["macro avg"]["precision"],
            "macro_recall": report_dict["macro avg"]["recall"],
            "macro_f1": report_dict["macro avg"]["f1-score"],
            "weighted_precision": report_dict["weighted avg"]["precision"],
            "weighted_recall": report_dict["weighted avg"]["recall"],
            "weighted_f1": report_dict["weighted avg"]["f1-score"],
            "conifer_precision": report_dict["conifer"]["precision"],
            "conifer_recall": report_dict["conifer"]["recall"],
            "conifer_f1": report_dict["conifer"]["f1-score"],
            "conifer_support": report_dict["conifer"]["support"],
            "broadleaf_precision": report_dict["broadleaf"]["precision"],
            "broadleaf_recall": report_dict["broadleaf"]["recall"],
            "broadleaf_f1": report_dict["broadleaf"]["f1-score"],
            "broadleaf_support": report_dict["broadleaf"]["support"],
        }
        summary_rows.append(summary_row)

        if verbose:
            print("\n" + "=" * 72)
            print(f"{task_name} | {group_name}")
            print("=" * 72)
            print(classification_report(
                y_true,
                y_pred,
                labels=[0, 1],
                target_names=BINARY_CLASS_NAMES,
                digits=4,
                zero_division=0,
            ))
            print("Confusion matrix:")
            print(cm_df)

    summary_df = pd.DataFrame(summary_rows)
    report_df = pd.concat(report_dfs, ignore_index=True) if report_dfs else pd.DataFrame()
    confusion_df = pd.concat(confusion_dfs, ignore_index=True) if confusion_dfs else pd.DataFrame()
    return summary_df, report_df, confusion_df


def apply_QA_filter(
    base_df,
    min_total_neighbors_for_QA,
    min_true_opposite_neighbors_for_QA,
    min_true_opposite_rate_for_QA,
    use_predicted_QA,
    min_pred_neighbors_for_QA,
    min_pred_disagree_neighbors_for_QA,
    min_pred_disagree_rate_for_QA,
):
    df = base_df.copy()

    true_rate = df["true_opposite_neighbor_rate"].fillna(0.0)
    pred_rate = df["neighbor_prediction_disagreement_rate"].fillna(0.0)

    df["true_mixed_QA_flag"] = (
        (df["n_neighbors_within_radius"] >= min_total_neighbors_for_QA)
        & (df["n_true_opposite_neighbors_within_radius"] >= min_true_opposite_neighbors_for_QA)
        & (true_rate >= min_true_opposite_rate_for_QA)
    )

    df["predicted_mixed_QA_flag"] = False
    if use_predicted_QA:
        df["predicted_mixed_QA_flag"] = (
            (df["n_neighbors_within_radius"] >= min_pred_neighbors_for_QA)
            & (
                (df["n_pred_disagree_neighbors_within_radius"] >= min_pred_disagree_neighbors_for_QA)
                | (pred_rate >= min_pred_disagree_rate_for_QA)
            )
        )

    df["remove_by_mixed_QA"] = df["true_mixed_QA_flag"] | df["predicted_mixed_QA_flag"]
    return df


def compute_sweep_metrics(filtered_df, full_df):
    rows = []

    for group_name, g_full in full_df.groupby("train_status", sort=False):
        g_eval = filtered_df[filtered_df["train_status"] == group_name].copy()

        n_total = len(g_full)
        n_evaluated = len(g_eval)
        n_removed = n_total - n_evaluated
        removed_fraction = n_removed / n_total if n_total > 0 else np.nan

        row = {
            "train_status": group_name,
            "n_total_before_QA": int(n_total),
            "n_evaluated_after_QA": int(n_evaluated),
            "n_removed_by_QA": int(n_removed),
            "removed_fraction": float(removed_fraction),
        }

        if n_evaluated == 0:
            row.update({
                "accuracy": np.nan,
                "macro_f1": np.nan,
                "weighted_f1": np.nan,
                "conifer_f1": np.nan,
                "broadleaf_f1": np.nan,
                "conifer_recall": np.nan,
                "broadleaf_recall": np.nan,
                "conifer_precision": np.nan,
                "broadleaf_precision": np.nan,
            })
            rows.append(row)
            continue

        y_true = g_eval["true_label"].to_numpy(dtype=int)
        y_pred = g_eval["pred_label"].to_numpy(dtype=int)

        report = classification_report(
            y_true,
            y_pred,
            labels=[0, 1],
            target_names=BINARY_CLASS_NAMES,
            output_dict=True,
            zero_division=0,
        )

        row.update({
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "macro_f1": report["macro avg"]["f1-score"],
            "weighted_f1": report["weighted avg"]["f1-score"],
            "conifer_f1": report["conifer"]["f1-score"],
            "broadleaf_f1": report["broadleaf"]["f1-score"],
            "conifer_recall": report["conifer"]["recall"],
            "broadleaf_recall": report["broadleaf"]["recall"],
            "conifer_precision": report["conifer"]["precision"],
            "broadleaf_precision": report["broadleaf"]["precision"],
        })
        rows.append(row)

    return rows


# ============================================================
# 1. Load train and any available non-training predictions
# ============================================================
prediction_specs = [
    (TRAIN_PRED_PATH, "train", True),
    (VAL_PRED_PATH, "val", False),
    (TEST_PRED_PATH, "test", False),
]

pred_parts = []
load_status_rows = []

for path, split_name, required in prediction_specs:
    df_part, status = maybe_load_pred_file(path, split_name, required=required)
    load_status_rows.append(status)
    if df_part is not None and len(df_part) > 0:
        pred_parts.append(df_part)

load_status_df = pd.DataFrame(load_status_rows)
load_status_df.to_csv(OUT_LOADED_FILES_PATH, index=False)

if not pred_parts:
    raise RuntimeError("No prediction files were loaded. Check paths.")

pred_df = pd.concat(pred_parts, ignore_index=True)

if pred_df["uniqueID"].duplicated().any():
    dupes = pred_df.loc[pred_df["uniqueID"].duplicated(), "uniqueID"].unique()
    raise ValueError(
        f"Some uniqueID values appear in multiple loaded prediction files. "
        f"Example duplicates: {dupes[:10]}"
    )

print(f"\nPredicted trees loaded: {len(pred_df)}")
print("\nLoaded/skipped file status:")
print(load_status_df)
print("\nSplit counts:")
print(pred_df["split"].value_counts())
print("\nTraining-status counts:")
print(pred_df["train_status"].value_counts())

if "non_training" not in set(pred_df["train_status"]):
    print("\nWarning: no non-training prediction files were loaded. Optimizing on training.")
    optimize_group = "training"
else:
    optimize_group = OPTIMIZE_ON_TRAIN_STATUS

# ============================================================
# 2. Load coords/genus from mixTreeTest and join to predictions
# ============================================================
coord_cols = [ID_COL, X_COL, Y_COL, GENUS_COL]

mix_df = pd.read_csv(MIX_TREE_PATH, usecols=coord_cols).copy()
mix_df = standardize_unique_id_column(mix_df)

mix_df = mix_df.dropna(subset=[ID_COL, X_COL, Y_COL, GENUS_COL]).copy()
mix_df = mix_df.drop_duplicates(subset=[ID_COL], keep="first")

mix_df["true_label_from_genus"] = mix_df[GENUS_COL].apply(genus_to_label)

unknown = mix_df.loc[mix_df["true_label_from_genus"].isna(), GENUS_COL].dropna().unique()
if len(unknown) > 0:
    print("Warning: dropping unrecognized genera:")
    print(sorted(unknown))

mix_df = mix_df[mix_df["true_label_from_genus"].notna()].copy()
mix_df["true_label_from_genus"] = mix_df["true_label_from_genus"].astype(int)
mix_df["true_label_name_from_genus"] = mix_df["true_label_from_genus"].map(IDX_TO_LABEL)

merged = pred_df.merge(mix_df, on=ID_COL, how="left", validate="one_to_one")

missing_coords = merged[X_COL].isna().sum()
if missing_coords > 0:
    print(f"Warning: {missing_coords} predicted trees missing coordinates/genus after merge.")
    merged = merged.dropna(subset=[X_COL, Y_COL, "true_label_from_genus"]).copy()

if len(merged) == 0:
    raise RuntimeError("No prediction rows remain after joining coordinates/genus.")

merged[PRED_COL] = merged[PRED_COL].astype(int)
merged["pred_label_name"] = merged[PRED_COL].map(IDX_TO_LABEL)

merged.to_csv(OUT_MERGED_PATH, index=False)
print(f"\nSaved merged dataset:\n{OUT_MERGED_PATH}")
print(f"Merged rows: {len(merged)}")

# ============================================================
# 3. Calculate neighborhood quantities
# ============================================================
coords = merged[[X_COL, Y_COL]].to_numpy(dtype=float)
true_labels = merged["true_label_from_genus"].to_numpy(dtype=int)
pred_labels = merged[PRED_COL].to_numpy(dtype=int)

tree = KDTree(coords, metric="euclidean")
neighbors = tree.query_radius(coords, r=RADIUS)

rows = []

for i, neigh_idx in enumerate(neighbors):
    neigh_idx = neigh_idx[neigh_idx != i]

    own_true = true_labels[i]
    own_pred = pred_labels[i]

    if len(neigh_idx) == 0:
        n_neighbors = 0
        n_true_opposite = 0
        n_pred_disagree = 0
        true_opposite_rate = np.nan
        pred_disagreement_rate = np.nan
    else:
        neighbor_true = true_labels[neigh_idx]
        neighbor_pred = pred_labels[neigh_idx]

        n_neighbors = len(neigh_idx)
        n_true_opposite = int(np.sum(neighbor_true != own_true))
        n_pred_disagree = int(np.sum(neighbor_pred != own_pred))

        true_opposite_rate = n_true_opposite / n_neighbors
        pred_disagreement_rate = n_pred_disagree / n_neighbors

    rows.append({
        ID_COL: merged.iloc[i][ID_COL],
        "split": merged.iloc[i]["split"],
        "train_status": merged.iloc[i]["train_status"],
        "genus": merged.iloc[i][GENUS_COL],
        "true_label": int(own_true),
        "true_label_name": IDX_TO_LABEL[int(own_true)],
        "pred_label": int(own_pred),
        "pred_label_name": IDX_TO_LABEL[int(own_pred)],
        "n_neighbors_within_radius": n_neighbors,
        "n_true_opposite_neighbors_within_radius": n_true_opposite,
        "true_opposite_neighbor_rate": true_opposite_rate,
        "n_pred_disagree_neighbors_within_radius": n_pred_disagree,
        "neighbor_prediction_disagreement_rate": pred_disagreement_rate,
    })

base_df = pd.DataFrame(rows)
base_df.to_csv(OUT_NEIGHBOR_PATH, index=False)

# ============================================================
# 5. Sweep QA thresholds
# ============================================================
sweep_rows = []

print("\nStarting QA threshold sweep...")

for min_total_neighbors_for_QA in MIN_TOTAL_NEIGHBORS_FOR_QA_GRID:
    for min_true_opposite_neighbors_for_QA in MIN_TRUE_OPPOSITE_NEIGHBORS_FOR_QA_GRID:
        for min_true_opposite_rate_for_QA in MIN_TRUE_OPPOSITE_RATE_FOR_QA_GRID:
            for use_predicted_QA in USE_PREDICTED_QA_GRID:

                pred_neighbor_grid = MIN_PRED_NEIGHBORS_FOR_QA_GRID if use_predicted_QA else [np.nan]
                pred_count_grid = MIN_PRED_DISAGREE_NEIGHBORS_FOR_QA_GRID if use_predicted_QA else [np.nan]
                pred_rate_grid = MIN_PRED_DISAGREE_RATE_FOR_QA_GRID if use_predicted_QA else [np.nan]

                for min_pred_neighbors_for_QA in pred_neighbor_grid:
                    for min_pred_disagree_neighbors_for_QA in pred_count_grid:
                        for min_pred_disagree_rate_for_QA in pred_rate_grid:

                            qa_df = apply_QA_filter(
                                base_df,
                                min_total_neighbors_for_QA=min_total_neighbors_for_QA,
                                min_true_opposite_neighbors_for_QA=min_true_opposite_neighbors_for_QA,
                                min_true_opposite_rate_for_QA=min_true_opposite_rate_for_QA,
                                use_predicted_QA=use_predicted_QA,
                                min_pred_neighbors_for_QA=(
                                    999999 if pd.isna(min_pred_neighbors_for_QA) else int(min_pred_neighbors_for_QA)
                                ),
                                min_pred_disagree_neighbors_for_QA=(
                                    999999 if pd.isna(min_pred_disagree_neighbors_for_QA) else int(min_pred_disagree_neighbors_for_QA)
                                ),
                                min_pred_disagree_rate_for_QA=(
                                    999999.0 if pd.isna(min_pred_disagree_rate_for_QA) else float(min_pred_disagree_rate_for_QA)
                                ),
                            )

                            retained_df = qa_df[~qa_df["remove_by_mixed_QA"]].copy()
                            metric_rows = compute_sweep_metrics(retained_df, base_df)

                            for metric_row in metric_rows:
                                sweep_rows.append({
                                    "min_total_neighbors_for_QA": min_total_neighbors_for_QA,
                                    "min_true_opposite_neighbors_for_QA": min_true_opposite_neighbors_for_QA,
                                    "min_true_opposite_rate_for_QA": min_true_opposite_rate_for_QA,
                                    "use_predicted_QA": use_predicted_QA,
                                    "min_pred_neighbors_for_QA": min_pred_neighbors_for_QA,
                                    "min_pred_disagree_neighbors_for_QA": min_pred_disagree_neighbors_for_QA,
                                    "min_pred_disagree_rate_for_QA": min_pred_disagree_rate_for_QA,
                                    **metric_row,
                                })

sweep_df = pd.DataFrame(sweep_rows)
sweep_df.to_csv(OUT_SWEEP_PATH, index=False)

print(f"\nSaved sweep results:\n{OUT_SWEEP_PATH}")

# ============================================================
# 6. Select best QA thresholds
# ============================================================
candidate_df = sweep_df[sweep_df["train_status"] == optimize_group].copy()

candidate_df = candidate_df[
    (candidate_df["removed_fraction"] <= MAX_REMOVED_FRACTION)
    & (candidate_df["n_evaluated_after_QA"] >= MIN_EVALUATED_RECORDS)
].copy()

if len(candidate_df) == 0:
    raise RuntimeError(
        "No candidate QA settings satisfy MAX_REMOVED_FRACTION and MIN_EVALUATED_RECORDS. "
        "Relax those constraints."
    )

best_row = (
    candidate_df
    .sort_values(
        [OPTIMIZE_METRIC, "conifer_f1", "macro_f1", "accuracy", "removed_fraction"],
        ascending=[False, False, False, False, True],
    )
    .iloc[0]
)

print("\nBest QA thresholds:")
print(best_row)

# ============================================================
# 7. Apply best QA threshold and save final outputs
# ============================================================
best_qa_df = apply_QA_filter(
    base_df,
    min_total_neighbors_for_QA=int(best_row["min_total_neighbors_for_QA"]),
    min_true_opposite_neighbors_for_QA=int(best_row["min_true_opposite_neighbors_for_QA"]),
    min_true_opposite_rate_for_QA=float(best_row["min_true_opposite_rate_for_QA"]),
    use_predicted_QA=bool(best_row["use_predicted_QA"]),
    min_pred_neighbors_for_QA=(
        999999 if pd.isna(best_row["min_pred_neighbors_for_QA"]) else int(best_row["min_pred_neighbors_for_QA"])
    ),
    min_pred_disagree_neighbors_for_QA=(
        999999 if pd.isna(best_row["min_pred_disagree_neighbors_for_QA"]) else int(best_row["min_pred_disagree_neighbors_for_QA"])
    ),
    min_pred_disagree_rate_for_QA=(
        999999.0 if pd.isna(best_row["min_pred_disagree_rate_for_QA"]) else float(best_row["min_pred_disagree_rate_for_QA"])
    ),
)

retained_df = best_qa_df[~best_qa_df["remove_by_mixed_QA"]].copy()
removed_df = best_qa_df[best_qa_df["remove_by_mixed_QA"]].copy()

best_extra = {
    "selected_on_train_status": optimize_group,
    "optimize_metric": OPTIMIZE_METRIC,
    "min_total_neighbors_for_QA": int(best_row["min_total_neighbors_for_QA"]),
    "min_true_opposite_neighbors_for_QA": int(best_row["min_true_opposite_neighbors_for_QA"]),
    "min_true_opposite_rate_for_QA": float(best_row["min_true_opposite_rate_for_QA"]),
    "use_predicted_QA": bool(best_row["use_predicted_QA"]),
    "min_pred_neighbors_for_QA": best_row["min_pred_neighbors_for_QA"],
    "min_pred_disagree_neighbors_for_QA": best_row["min_pred_disagree_neighbors_for_QA"],
    "min_pred_disagree_rate_for_QA": best_row["min_pred_disagree_rate_for_QA"],
    "radius_m": RADIUS_M,
    "max_removed_fraction_constraint": MAX_REMOVED_FRACTION,
}

summary_df, report_df, confusion_df = evaluate_binary_by_train_status(
    retained_df,
    true_col="true_label",
    pred_col="pred_label",
    task_name="binary_after_optimized_mixed_QA_filter",
    extra=best_extra,
    verbose=True,
)

qa_summary = (
    best_qa_df
    .groupby(["train_status", "split"])
    .agg(
        n_total=(ID_COL, "count"),
        n_removed_by_mixed_QA=("remove_by_mixed_QA", "sum"),
        n_true_mixed_QA=("true_mixed_QA_flag", "sum"),
        n_predicted_mixed_QA=("predicted_mixed_QA_flag", "sum"),
        mean_neighbors=("n_neighbors_within_radius", "mean"),
        mean_true_opposite_neighbors=("n_true_opposite_neighbors_within_radius", "mean"),
        mean_true_opposite_rate=("true_opposite_neighbor_rate", "mean"),
        mean_pred_disagree_neighbors=("n_pred_disagree_neighbors_within_radius", "mean"),
        mean_pred_disagreement_rate=("neighbor_prediction_disagreement_rate", "mean"),
    )
    .reset_index()
)

qa_summary["pct_removed_by_mixed_QA"] = qa_summary["n_removed_by_mixed_QA"] / qa_summary["n_total"]

best_qa_df.to_csv(OUT_BEST_QA_PATH, index=False)
retained_df.to_csv(OUT_RETAINED_PATH, index=False)
removed_df.to_csv(OUT_REMOVED_PATH, index=False)
summary_df.to_csv(OUT_SUMMARY_PATH, index=False)
report_df.to_csv(OUT_REPORT_PATH, index=False)
confusion_df.to_csv(OUT_CONFUSION_PATH, index=False)
qa_summary.to_csv(OUT_QA_SUMMARY_PATH, index=False)

print("\nBest QA filter summary:")
print(qa_summary)

print("\nSaved:")
print(OUT_SWEEP_PATH)
print(OUT_BEST_QA_PATH)
print(OUT_RETAINED_PATH)
print(OUT_REMOVED_PATH)
print(OUT_QA_SUMMARY_PATH)
print(OUT_SUMMARY_PATH)
print(OUT_REPORT_PATH)
print(OUT_CONFUSION_PATH)
