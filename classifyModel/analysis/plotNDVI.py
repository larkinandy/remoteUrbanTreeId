import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# USER SETTINGS
# ============================================================
OUT_DIR = r"C:/users/larki/Desktop/"

CACHE_PATH = os.path.join(OUT_DIR, "hybrid_lstm_broad2_treefused_era5_cache.pkl")

ROW_PRED_PATH = os.path.join(
    OUT_DIR,
    "hybrid_lstm_broad2_treefused_era5_test_row_predictions.csv"
)

PLOT_OUT = os.path.join(
    OUT_DIR,
    "NDVI_mean_vs_prob_conifer.png"
)

CSV_OUT = os.path.join(
    OUT_DIR,
    "NDVI_mean_vs_prob_conifer_data.csv"
)


# ============================================================
# LOAD DATA
# ============================================================
with open(CACHE_PATH, "rb") as f:
    cache = pickle.load(f)

row_df = pd.read_csv(ROW_PRED_PATH)

annual_metrics = np.asarray(cache["annual_metrics"])
annual_metric_names = list(cache["annual_metric_names"])

# Fix older cache where these 5 duplicate NDVI names were appended
# to annual_metric_names but not added to annual_metrics.
duplicate_tail = [
    "NDVI_mean",
    "NDVI_min",
    "NDVI_max",
    "NDVI_range",
    "NDVI_std",
]

if (
    len(annual_metric_names) == annual_metrics.shape[1] + len(duplicate_tail)
    and annual_metric_names[-len(duplicate_tail):] == duplicate_tail
):
    print("Patching annual_metric_names: removing duplicate NDVI summary names from tail.")
    annual_metric_names = annual_metric_names[:-len(duplicate_tail)]

print("annual_metrics shape:", annual_metrics.shape)
print("annual_metric_names length:", len(annual_metric_names))

if annual_metrics.shape[1] != len(annual_metric_names):
    raise ValueError(
        f"Still mismatched: annual_metrics has {annual_metrics.shape[1]} columns, "
        f"but annual_metric_names has {len(annual_metric_names)} names."
    )

feature_df = pd.DataFrame({
    "uniqueID": cache["unique_ids"],
    "year": cache["years"],
})

annual_df = pd.DataFrame(
    annual_metrics,
    columns=annual_metric_names
)

feature_df = pd.concat(
    [feature_df.reset_index(drop=True), annual_df.reset_index(drop=True)],
    axis=1
)

row_df["uniqueID"] = row_df["uniqueID"].astype(str)
feature_df["uniqueID"] = feature_df["uniqueID"].astype(str)

row_df["year"] = row_df["year"].astype(int)
feature_df["year"] = feature_df["year"].astype(int)

df = row_df.merge(
    feature_df,
    on=["uniqueID", "year"],
    how="inner"
)

if "NDVI_mean" not in df.columns:
    raise ValueError("NDVI_mean not found. Check annual_metric_names/cache.")

if "prob_conifer" not in df.columns:
    raise ValueError("prob_conifer not found in prediction CSV.")


# ============================================================
# CLEAN / PREP
# ============================================================
df = df.replace([np.inf, -np.inf], np.nan)
df = df.dropna(subset=["NDVI_mean", "prob_conifer", "y_true_name"])

df["correct"] = df["y_true_name"] == df["y_pred_name"]

df.to_csv(CSV_OUT, index=False)


# ============================================================
# PLOT 1: ALL POINTS
# ============================================================
plt.figure(figsize=(8, 6))

colors = {
    "conifer": "tab:green",
    "broadleaf": "tab:blue",
}

for label, g in df.groupby("y_true_name"):
    plt.scatter(
        g["NDVI_mean"],
        g["prob_conifer"],
        s=12,
        alpha=0.35,
        label=f"True {label}",
        color=colors.get(label, None),
    )

plt.axhline(0.5, linestyle="--", linewidth=1, color="black")
plt.xlabel("Annual NDVI mean")
plt.ylabel("Predicted P(conifer)")
plt.title("NDVI_mean vs Predicted Probability of Conifer")
plt.legend()
plt.tight_layout()
plt.savefig(PLOT_OUT, dpi=300)
plt.close()


# ============================================================
# PLOT 2: HIGHLIGHT MISSED CONIFERS
# ============================================================
PLOT_OUT_2 = os.path.join(
    OUT_DIR,
    "NDVI_mean_vs_prob_conifer_highlight_missed_conifers.png"
)

missed_conifers = df[
    (df["y_true_name"] == "conifer") &
    (df["y_pred_name"] != "conifer")
].copy()

correct_conifers = df[
    (df["y_true_name"] == "conifer") &
    (df["y_pred_name"] == "conifer")
].copy()

broadleaf = df[df["y_true_name"] == "broadleaf"].copy()

plt.figure(figsize=(8, 6))

plt.scatter(
    broadleaf["NDVI_mean"],
    broadleaf["prob_conifer"],
    s=8,
    alpha=0.18,
    label="True broadleaf",
)

plt.scatter(
    correct_conifers["NDVI_mean"],
    correct_conifers["prob_conifer"],
    s=18,
    alpha=0.55,
    label="Correct conifer",
)

plt.scatter(
    missed_conifers["NDVI_mean"],
    missed_conifers["prob_conifer"],
    s=24,
    alpha=0.85,
    label="Missed conifer",
    marker="x",
)

plt.axhline(0.5, linestyle="--", linewidth=1, color="black")
plt.xlabel("Annual NDVI mean")
plt.ylabel("Predicted P(conifer)")
plt.title("NDVI_mean vs P(conifer): Missed Conifers Highlighted")
plt.legend()
plt.tight_layout()
plt.savefig(PLOT_OUT_2, dpi=300)
plt.close()


# ============================================================
# PLOT 3: BINNED SUMMARY
# ============================================================
PLOT_OUT_3 = os.path.join(
    OUT_DIR,
    "NDVI_mean_binned_prob_conifer_summary.png"
)

df["ndvi_bin"] = pd.qcut(df["NDVI_mean"], q=10, duplicates="drop")

bin_summary = (
    df
    .groupby(["ndvi_bin", "y_true_name"], observed=True)
    .agg(
        ndvi_mean=("NDVI_mean", "mean"),
        prob_conifer_mean=("prob_conifer", "mean"),
        prob_conifer_median=("prob_conifer", "median"),
        n=("uniqueID", "size"),
    )
    .reset_index()
)

bin_summary.to_csv(
    os.path.join(OUT_DIR, "NDVI_mean_binned_prob_conifer_summary.csv"),
    index=False
)

plt.figure(figsize=(8, 6))

for label, g in bin_summary.groupby("y_true_name"):
    plt.plot(
        g["ndvi_mean"],
        g["prob_conifer_mean"],
        marker="o",
        label=f"True {label}",
    )

plt.axhline(0.5, linestyle="--", linewidth=1, color="black")
plt.xlabel("Annual NDVI mean bin center")
plt.ylabel("Mean predicted P(conifer)")
plt.title("Binned NDVI_mean vs Mean P(conifer)")
plt.legend()
plt.tight_layout()
plt.savefig(PLOT_OUT_3, dpi=300)
plt.close()


# ============================================================
# PRINT SUMMARY
# ============================================================
print("Saved:")
print(PLOT_OUT)
print(PLOT_OUT_2)
print(PLOT_OUT_3)
print(CSV_OUT)

print("\nMissed conifer NDVI/probability summary:")
print(
    missed_conifers[["NDVI_mean", "prob_conifer"]]
    .describe()
)

print("\nCorrect conifer NDVI/probability summary:")
print(
    correct_conifers[["NDVI_mean", "prob_conifer"]]
    .describe()
)

print("\nBroadleaf NDVI/probability summary:")
print(
    broadleaf[["NDVI_mean", "prob_conifer"]]
    .describe()
)