import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, top_k_accuracy_score


# ============================================================
# Sentinel-2 indices
# ============================================================
def add_sentinel2_indices(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    eps = 1e-10

    green = df["B3"].astype(float)
    red = df["B4"].astype(float)
    re1 = df["B5"].astype(float)
    re2 = df["B6"].astype(float)
    re3 = df["B7"].astype(float)
    nir = df["B8"].astype(float)
    nirn = df["B8A"].astype(float)
    swir1 = df["B11"].astype(float)

    df["NDVI"] = (nir - red) / (nir + red + eps)
    df["GNDVI"] = (nir - green) / (nir + green + eps)
    df["CIg"] = (nir / (green + eps)) - 1.0
    df["CIre"] = (nirn / (re1 + eps)) - 1.0
    df["MTCI"] = (re2 - re1) / (re1 - red + eps)
    df["MCARI"] = ((re1 - red) - 0.2 * (re1 - green)) * (re1 / (red + eps))
    df["NDVIre1"] = (nir - re1) / (nir + re1 + eps)
    df["NDVIre2"] = (nir - re2) / (nir + re2 + eps)
    df["REPI"] = 700.0 + 40.0 * (((red + re3) / 2.0 - re1) / (re2 - re1 + eps))
    df["NDII"] = (nir - swir1) / (nir + swir1 + eps)
    df["MSAVI"] = 0.5 * (2.0 * nir + 1.0 - np.sqrt((2.0 * nir + 1.0) ** 2 - 8.0 * (nir - red)))

    ndvi = df["NDVI"]
    lai_ndvi = -np.log((0.69 - ndvi) / 0.59 + eps)
    df["LAI_ndvi"] = np.clip(lai_ndvi, 0, 6) / 6.0

    cire = df["CIre"]
    lai_re = 3.618 * cire - 0.118
    df["LAI_re"] = np.clip(lai_re, 0, 6) / 6.0

    df = df.replace([np.inf, -np.inf], np.nan)
    return df


# ============================================================
# Broad label mapping
# ============================================================
def genus_to_broad_label(genus: str, use_two_broad_classes: bool = False) -> str:
    genus = str(genus).strip()

    conifer = {
        "Abies", "Calocedrus", "Chamaecyparis", "Cryptomeria", "Cupressocyparis",
        "Ginkgo", "Juniperus", "Larix", "Metasequoia", "Picea", "Pinus",
        "Platycladus", "Pseudotsuga", "Taxodium", "Taxus", "Thuja", "Tsuga",
        "Cedrus", "Cupressus", "Glyptostrobus", "Sequoia", "Sequoiadendron"
    }

    if use_two_broad_classes:
        if genus in conifer:
            return "conifer"
        return "broadleaf"

    urban_ornamental = {
        "Aesculus", "Amelanchier", "Catalpa", "Cercidiphyllum", "Cercis",
        "Chionanthus", "Cladrastis", "Cornus", "Cotinus", "Crataegus",
        "Cydonia", "Eucommia", "Ficus", "Gymnocladus", "Halesia", "Ilex",
        "Koelreuteria", "Laburnum", "Liriodendron", "Maclura", "Maackia",
        "Magnolia", "Malus", "Parrotia", "Paulownia", "Phellodendron",
        "Platanus", "Prunus", "Pyrus", "Sassafras", "Sorbus", "Sophora",
        "Syringa", "Xanthoceras", "Zelkova"
    }

    if genus in conifer:
        return "conifer"
    if genus in urban_ornamental:
        return "urban_ornamental"
    return "deciduous"


# ============================================================
# Genus labels with structured "other" groups
# ============================================================
def make_genus_labels_with_structured_other(
    y_df: pd.DataFrame,
    unique_id_col: str = "uniqueID",
    genus_col: str = "BOTANICALG",
    min_genus_samples: int = 500,
    use_two_broad_classes: bool = False,
):
    df = y_df[[unique_id_col, genus_col]].copy()
    df = df.drop_duplicates(subset=[unique_id_col])

    df[genus_col] = df[genus_col].astype(str).str.strip()
    df["broad_label"] = df[genus_col].apply(
        lambda g: genus_to_broad_label(g, use_two_broad_classes=use_two_broad_classes)
    )

    genus_counts = df[genus_col].value_counts()
    kept_genera = set(genus_counts[genus_counts >= min_genus_samples].index)

    def map_genus(genus: str, broad_label: str) -> str:
        if genus in kept_genera:
            return genus
        if broad_label == "deciduous":
            return "other_deciduous"
        elif broad_label == "conifer":
            return "other_conifer"
        elif broad_label == "urban_ornamental":
            return "other_ornamental"
        else:
            return "other"

    df["genus_label"] = [
        map_genus(g, b) for g, b in zip(df[genus_col], df["broad_label"])
    ]

    genus_classes = sorted(df["genus_label"].unique())
    genus_to_idx = {g: i for i, g in enumerate(genus_classes)}
    idx_to_genus = {i: g for g, i in genus_to_idx.items()}

    df["genus_idx"] = df["genus_label"].map(genus_to_idx).astype(int)

    meta = {
        "genus_classes": genus_classes,
        "genus_to_idx": genus_to_idx,
        "idx_to_genus": idx_to_genus,
        "genus_counts_raw": genus_counts.to_dict(),
        "kept_genera": sorted(kept_genera),
        "min_genus_samples": min_genus_samples,
    }
    return df, meta


# ============================================================
# Utilities
# ============================================================
def save_confusion_matrix(cm, labels, out_csv, out_png, title):
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)
    cm_df.to_csv(out_csv)

    cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-10)
    cm_norm_df = pd.DataFrame(cm_norm, index=labels, columns=labels)
    cm_norm_df.to_csv(out_csv.replace(".csv", "_normalized.csv"))

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_norm_df, cmap="Blues", square=False, cbar=True)
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()


# ============================================================
# Season helpers
# ============================================================
def month_to_season(month: int) -> str:
    if month in [12, 1, 2]:
        return "winter"
    elif month in [3, 4, 5]:
        return "spring"
    elif month in [6, 7, 8]:
        return "summer"
    else:
        return "fall"


def encode_doy(doy):
    angle = 2 * np.pi * doy / 365.0
    return np.sin(angle), np.cos(angle)


# ============================================================
# Phenology feature extraction
# ============================================================
def compute_series_features(g: pd.DataFrame, value_col: str) -> dict:
    out = {}

    s = g[["date", "doy", "season", value_col]].copy()
    s[value_col] = pd.to_numeric(s[value_col], errors="coerce")
    s = s.dropna(subset=[value_col]).sort_values("date")

    prefix = value_col

    feature_names = [
        "mean", "min", "max", "range", "std",
        "doy_max_sin", "doy_max_cos",
        "doy_min_sin", "doy_min_cos",
        "center_green_sin", "center_green_cos",
        "max_positive_slope", "max_negative_slope",
        "frac_above_yearly_mean",
        "frac_time_above_yearly_mean",
        "frac_time_above_80pct_max",
        "summer_mean", "winter_mean",
        "spring_minus_winter", "fall_minus_summer",
    ]

    if len(s) == 0:
        for name in feature_names:
            out[f"{prefix}_{name}"] = np.nan
        return out

    vals = s[value_col].to_numpy(dtype=float)
    doys = s["doy"].to_numpy(dtype=float)

    yearly_mean = np.mean(vals)
    yearly_min = np.min(vals)
    yearly_max = np.max(vals)
    yearly_range = yearly_max - yearly_min
    yearly_std = np.std(vals)

    doy_of_max = float(doys[np.argmax(vals)])
    doy_of_min = float(doys[np.argmin(vals)])

    sin_max, cos_max = encode_doy(doy_of_max)
    sin_min, cos_min = encode_doy(doy_of_min)

    frac_above_yearly_mean = float(np.mean(vals > yearly_mean))

    weights = np.clip(vals, a_min=0, a_max=None)
    if weights.sum() > 0:
        weighted_doy = np.sum(doys * weights) / np.sum(weights)
        center_green_sin, center_green_cos = encode_doy(weighted_doy)
    else:
        center_green_sin, center_green_cos = np.nan, np.nan

    if len(s) >= 2:
        delta_val = np.diff(vals)
        delta_day = np.diff(doys)

        valid = delta_day > 0
        if valid.any():
            slopes = delta_val[valid] / delta_day[valid]
            pos_slopes = slopes[slopes > 0]
            neg_slopes = slopes[slopes < 0]

            max_positive_slope = float(pos_slopes.max()) if len(pos_slopes) > 0 else 0.0
            max_negative_slope = float(neg_slopes.min()) if len(neg_slopes) > 0 else 0.0
        else:
            max_positive_slope = 0.0
            max_negative_slope = 0.0
    else:
        max_positive_slope = 0.0
        max_negative_slope = 0.0

    if len(s) >= 2:
        v0 = vals[:-1]
        v1 = vals[1:]
        d0 = doys[:-1]
        d1 = doys[1:]
        interval_days = d1 - d0

        valid = interval_days > 0
        if valid.any():
            interval_days = interval_days[valid]
            mid_vals = (v0[valid] + v1[valid]) / 2.0

            total_days = interval_days.sum()
            frac_time_above_yearly_mean = float(
                interval_days[mid_vals > yearly_mean].sum() / total_days
            )

            high_thresh = 0.8 * yearly_max
            frac_time_above_80pct_max = float(
                interval_days[mid_vals > high_thresh].sum() / total_days
            )
        else:
            frac_time_above_yearly_mean = np.nan
            frac_time_above_80pct_max = np.nan
    else:
        frac_time_above_yearly_mean = np.nan
        frac_time_above_80pct_max = np.nan

    seasonal_means = s.groupby("season")[value_col].mean()

    summer_mean = float(seasonal_means.get("summer", np.nan))
    winter_mean = float(seasonal_means.get("winter", np.nan))
    spring_mean = float(seasonal_means.get("spring", np.nan))
    fall_mean = float(seasonal_means.get("fall", np.nan))

    spring_minus_winter = (
        float(spring_mean - winter_mean)
        if pd.notna(spring_mean) and pd.notna(winter_mean)
        else np.nan
    )
    fall_minus_summer = (
        float(fall_mean - summer_mean)
        if pd.notna(fall_mean) and pd.notna(summer_mean)
        else np.nan
    )

    out[f"{prefix}_mean"] = float(yearly_mean)
    out[f"{prefix}_min"] = float(yearly_min)
    out[f"{prefix}_max"] = float(yearly_max)
    out[f"{prefix}_range"] = float(yearly_range)
    out[f"{prefix}_std"] = float(yearly_std)

    out[f"{prefix}_doy_max_sin"] = sin_max
    out[f"{prefix}_doy_max_cos"] = cos_max
    out[f"{prefix}_doy_min_sin"] = sin_min
    out[f"{prefix}_doy_min_cos"] = cos_min

    out[f"{prefix}_center_green_sin"] = center_green_sin
    out[f"{prefix}_center_green_cos"] = center_green_cos

    out[f"{prefix}_max_positive_slope"] = max_positive_slope
    out[f"{prefix}_max_negative_slope"] = max_negative_slope

    out[f"{prefix}_frac_above_yearly_mean"] = frac_above_yearly_mean
    out[f"{prefix}_frac_time_above_yearly_mean"] = frac_time_above_yearly_mean
    out[f"{prefix}_frac_time_above_80pct_max"] = frac_time_above_80pct_max

    out[f"{prefix}_summer_mean"] = summer_mean
    out[f"{prefix}_winter_mean"] = winter_mean
    out[f"{prefix}_spring_minus_winter"] = spring_minus_winter
    out[f"{prefix}_fall_minus_summer"] = fall_minus_summer

    return out


def build_phenology_feature_table(x_df: pd.DataFrame) -> pd.DataFrame:
    df = x_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["doy"] = df["date"].dt.dayofyear
    df["season"] = df["month"].apply(month_to_season)

    rows = []

    for (uid, year), g in df.groupby(["uniqueID", "year"], sort=False):
        row = {
            "uniqueID": uid,
            "year": int(year),
            "n_obs": int(len(g)),
        }
        row.update(compute_series_features(g, "NDVI"))
        row.update(compute_series_features(g, "CIre"))
        rows.append(row)

    return pd.DataFrame(rows)


# ============================================================
# Fusion across years
# ============================================================
def fuse_tree_year_probs(df_split, probs, class_names):
    tmp = df_split[["uniqueID", "year", "genus_idx"]].copy()
    tmp["row_idx"] = np.arange(len(tmp))

    rows = []
    for uid, g in tmp.groupby("uniqueID", sort=False):
        idx = g["row_idx"].to_numpy()
        p = probs[idx]

        fused_logp = np.mean(np.log(p + 1e-12), axis=0)
        pred = int(np.argmax(fused_logp))
        true = int(g["genus_idx"].iloc[0])

        rows.append({
            "uniqueID": uid,
            "n_years": len(idx),
            "genus_true": true,
            "genus_pred_fused": pred,
        })

    out = pd.DataFrame(rows)
    acc = (out["genus_true"] == out["genus_pred_fused"]).mean()
    out["genus_true_name"] = out["genus_true"].map(lambda i: class_names[i])
    out["genus_pred_fused_name"] = out["genus_pred_fused"].map(lambda i: class_names[i])
    return out, acc


# ============================================================
# Main
# ============================================================
def main():
    X_PATH = r"C:/users/larki/Desktop/PollenSense/xDataNormalized.csv"
    Y_PATH = r"C:/users/larki/Desktop/PollenSense/yDataInteger.csv"
    HEIGHT_PATH = r"C:/users/larki/Desktop/PollenSense/GIS/meta_trees_height_features.csv"
    OUT_DIR = r"C:/users/larki/Desktop/"

    USE_TWO_BROAD_CLASSES = False
    RANDOM_STATE = 42
    MIN_GENUS_SAMPLES = 1000

    N_ESTIMATORS = 1000
    MAX_DEPTH = 25
    MIN_SAMPLES_LEAF = 5
    N_JOBS = -1

    PHENO_PATH = r"C:/users/larki/Desktop/phenology_ndvi_cire.parquet"
    FORCE_REBUILD_PHENO = False

    X_INDEX_PATH = r"C:/users/larki/Desktop/x_with_indices.parquet"
    FORCE_REBUILD_X = False

    genus_name_col = "BOTANICALG"

    height_cols = [
        "ht_center_1m",
        "ht_max_3m",
        "ht_mean_3m",
        "ht_p25_5m",
        "ht_p50_5m",
        "ht_p90_5m",
        "ht_std_3m",
    ]

    os.makedirs(OUT_DIR, exist_ok=True)

    print("Loading data...")
    x_df = pd.read_csv(X_PATH)
    y_df = pd.read_csv(Y_PATH)
    h_df = pd.read_csv(HEIGHT_PATH)

    if os.path.exists(X_INDEX_PATH) and not FORCE_REBUILD_X:
        print("Loading cached Sentinel-2 indices...")
        x_df = pd.read_parquet(X_INDEX_PATH)
    else:
        print("Computing Sentinel-2 indices...")
        x_df = add_sentinel2_indices(x_df)
        x_df.to_parquet(X_INDEX_PATH, index=False)

    if "uniqueID" not in x_df.columns:
        raise ValueError("xDataNormalized.csv is missing 'uniqueID'")
    if "date" not in x_df.columns:
        raise ValueError("xDataNormalized.csv is missing 'date'")
    if "uniqueID" not in y_df.columns:
        raise ValueError("yDataInteger.csv is missing 'uniqueID'")
    if genus_name_col not in y_df.columns:
        raise ValueError(f"yDataInteger.csv is missing '{genus_name_col}'")

    # ------------------------------------------------
    # Labels
    # ------------------------------------------------
    y_tree, genus_meta = make_genus_labels_with_structured_other(
        y_df=y_df,
        unique_id_col="uniqueID",
        genus_col=genus_name_col,
        min_genus_samples=MIN_GENUS_SAMPLES,
        use_two_broad_classes=USE_TWO_BROAD_CLASSES,
    )

    genus_classes = genus_meta["genus_classes"]

    print("\nTree-level genus counts:")
    print(y_tree["genus_label"].value_counts())

    # ------------------------------------------------
    # Phenology features
    # ------------------------------------------------
    if os.path.exists(PHENO_PATH) and not FORCE_REBUILD_PHENO:
        print("\nLoading cached phenology table...")
        pheno_df = pd.read_parquet(PHENO_PATH)
    else:
        print("\nBuilding phenology feature table...")
        pheno_df = build_phenology_feature_table(x_df)

        print("Saving phenology table...")
        pheno_df.to_parquet(PHENO_PATH, index=False)

    print("Tree-year rows:", len(pheno_df))

    # ------------------------------------------------
    # Height features
    # ------------------------------------------------
    needed_height_cols = ["uniqueID"] + height_cols
    missing_height_cols = [c for c in needed_height_cols if c not in h_df.columns]
    if missing_height_cols:
        raise ValueError(f"Missing columns in height file: {missing_height_cols}")

    h_tree = h_df[needed_height_cols].copy().drop_duplicates(subset=["uniqueID"])

    # ------------------------------------------------
    # Merge
    # ------------------------------------------------
    model_df = pheno_df.merge(
        y_tree[["uniqueID", "genus_label", "genus_idx"]],
        on="uniqueID",
        how="inner"
    ).merge(
        h_tree,
        on="uniqueID",
        how="left"
    )

    if model_df.empty:
        raise ValueError("Merged modeling dataframe is empty")

    # ------------------------------------------------
    # Split by tree
    # ------------------------------------------------
    tree_labels = y_tree[["uniqueID", "genus_idx"]].copy()

    train_ids, temp_ids = train_test_split(
        tree_labels["uniqueID"].to_numpy(),
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=tree_labels["genus_idx"].to_numpy(),
    )

    temp_genus_idx = tree_labels.set_index("uniqueID").loc[temp_ids, "genus_idx"].to_numpy()

    val_ids, test_ids = train_test_split(
        temp_ids,
        test_size=0.50,
        random_state=RANDOM_STATE,
        stratify=temp_genus_idx,
    )

    train_df = model_df[model_df["uniqueID"].isin(train_ids)].copy()
    val_df = model_df[model_df["uniqueID"].isin(val_ids)].copy()
    test_df = model_df[model_df["uniqueID"].isin(test_ids)].copy()

    print("\nUnique trees by split:")
    print("train:", len(np.unique(train_df["uniqueID"])))
    print("val:  ", len(np.unique(val_df["uniqueID"])))
    print("test: ", len(np.unique(test_df["uniqueID"])))

    print("\nTree-year rows by split:")
    print("train:", len(train_df))
    print("val:  ", len(val_df))
    print("test: ", len(test_df))

    # ------------------------------------------------
    # Predictors
    # ------------------------------------------------
    predictor_cols = [
        c for c in model_df.columns
        if c not in ["uniqueID", "year", "genus_label", "genus_idx"]
    ]

    train_means = train_df[predictor_cols].mean(numeric_only=True)

    train_df[predictor_cols] = train_df[predictor_cols].fillna(train_means)
    val_df[predictor_cols] = val_df[predictor_cols].fillna(train_means)
    test_df[predictor_cols] = test_df[predictor_cols].fillna(train_means)

    X_train = train_df[predictor_cols].to_numpy(dtype=np.float32)
    y_train = train_df["genus_idx"].to_numpy(dtype=int)

    X_val = val_df[predictor_cols].to_numpy(dtype=np.float32)
    y_val = val_df["genus_idx"].to_numpy(dtype=int)

    X_test = test_df[predictor_cols].to_numpy(dtype=np.float32)
    y_test = test_df["genus_idx"].to_numpy(dtype=int)

    from sklearn.utils.class_weight import compute_class_weight

    classes = np.unique(y_train)
    weights = compute_class_weight("balanced", classes=classes, y=y_train)
    class_weight_dict = {c: w for c, w in zip(classes, weights)}

    # ------------------------------------------------
    # Train RF
    # ------------------------------------------------
    print("\nTraining Random Forest...")
    rf = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        min_samples_leaf=MIN_SAMPLES_LEAF,
        n_jobs=N_JOBS,
        random_state=RANDOM_STATE,
        class_weight=class_weight_dict,
    )


    rf.fit(X_train, y_train)

    # ------------------------------------------------
    # Evaluate tree-year rows
    # ------------------------------------------------
    val_pred = rf.predict(X_val)
    test_pred = rf.predict(X_test)

    val_proba = rf.predict_proba(X_val)
    test_proba = rf.predict_proba(X_test)

    val_acc = accuracy_score(y_val, val_pred)
    test_acc = accuracy_score(y_test, test_pred)

    val_top3 = top_k_accuracy_score(
        y_val, val_proba, k=min(3, len(genus_classes)), labels=list(range(len(genus_classes)))
    )
    test_top3 = top_k_accuracy_score(
        y_test, test_proba, k=min(3, len(genus_classes)), labels=list(range(len(genus_classes)))
    )

    print(f"\nValidation accuracy (tree-year rows): {val_acc:.4f}")
    print(f"Test accuracy (tree-year rows):       {test_acc:.4f}")
    print(f"Validation top-3 accuracy:            {val_top3:.4f}")
    print(f"Test top-3 accuracy:                  {test_top3:.4f}")

    print("\nValidation classification report:")
    print(classification_report(y_val, val_pred, target_names=genus_classes, digits=4))

    print("\nTest classification report:")
    print(classification_report(y_test, test_pred, target_names=genus_classes, digits=4))

    # ------------------------------------------------
    # Tree-level fusion
    # ------------------------------------------------
    val_tree_df, val_tree_acc = fuse_tree_year_probs(val_df, val_proba, genus_classes)
    test_tree_df, test_tree_acc = fuse_tree_year_probs(test_df, test_proba, genus_classes)

    print(f"\nValidation accuracy (tree-fused): {val_tree_acc:.4f}")
    print(f"Test accuracy (tree-fused):       {test_tree_acc:.4f}")

    # ------------------------------------------------
    # Save outputs
    # ------------------------------------------------
    prefix = os.path.join(OUT_DIR, "rf_genus_compact_phenology_ndvi_cire_structured_other")

    pd.DataFrame({
        "feature": predictor_cols,
        "importance": rf.feature_importances_
    }).sort_values("importance", ascending=False).to_csv(
        prefix + "_feature_importance.csv", index=False
    )

    pd.DataFrame([{
        "val_acc_treeyear": val_acc,
        "test_acc_treeyear": test_acc,
        "val_top3_treeyear": val_top3,
        "test_top3_treeyear": test_top3,
        "val_acc_treefused": val_tree_acc,
        "test_acc_treefused": test_tree_acc,
        "n_train_rows": len(train_df),
        "n_val_rows": len(val_df),
        "n_test_rows": len(test_df),
        "n_train_trees": len(np.unique(train_df["uniqueID"])),
        "n_val_trees": len(np.unique(val_df["uniqueID"])),
        "n_test_trees": len(np.unique(test_df["uniqueID"])),
        "n_estimators": N_ESTIMATORS,
        "max_depth": MAX_DEPTH,
        "min_samples_leaf": MIN_SAMPLES_LEAF,
        "min_genus_samples": MIN_GENUS_SAMPLES,
    }]).to_csv(prefix + "_results.csv", index=False)

    val_tree_df.to_csv(prefix + "_val_tree_fused_predictions.csv", index=False)
    test_tree_df.to_csv(prefix + "_test_tree_fused_predictions.csv", index=False)

    val_cm = confusion_matrix(y_val, val_pred, labels=list(range(len(genus_classes))))
    save_confusion_matrix(
        cm=val_cm,
        labels=genus_classes,
        out_csv=prefix + "_val_confusion.csv",
        out_png=prefix + "_val_confusion.png",
        title="Validation Confusion Matrix (Tree-Year Rows)"
    )

    test_cm = confusion_matrix(y_test, test_pred, labels=list(range(len(genus_classes))))
    save_confusion_matrix(
        cm=test_cm,
        labels=genus_classes,
        out_csv=prefix + "_test_confusion.csv",
        out_png=prefix + "_test_confusion.png",
        title="Test Confusion Matrix (Tree-Year Rows)"
    )

    test_tree_cm = confusion_matrix(
        test_tree_df["genus_true"],
        test_tree_df["genus_pred_fused"],
        labels=list(range(len(genus_classes)))
    )
    save_confusion_matrix(
        cm=test_tree_cm,
        labels=genus_classes,
        out_csv=prefix + "_test_tree_fused_confusion.csv",
        out_png=prefix + "_test_tree_fused_confusion.png",
        title="Test Confusion Matrix (Tree-Fused)"
    )

    with open(prefix + "_metadata.json", "w") as f:
        json.dump({
            "phenology_variables": ["NDVI", "CIre"],
            "height_cols": height_cols,
            "predictor_cols": predictor_cols,
            "genus_classes": genus_classes,
            "kept_genera": genus_meta["kept_genera"],
            "min_genus_samples": MIN_GENUS_SAMPLES,
            "n_estimators": N_ESTIMATORS,
            "max_depth": MAX_DEPTH,
            "min_samples_leaf": MIN_SAMPLES_LEAF,
            "random_state": RANDOM_STATE,
        }, f, indent=2)

    print(f"\nSaved outputs with prefix: {prefix}")


if __name__ == "__main__":
    main()