import os
import json
import time
import random

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torch.optim.lr_scheduler import ReduceLROnPlateau

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    precision_recall_fscore_support,
)

try:
    from .hybrid_meteor_data import (
        YearHybridDataset,
        add_sentinel2_indices,
        build_tree_year_records_with_annual_metrics_and_era5,
        collate_year_hybrid_batch,
        load_era5_folder,
        make_two_broad_labels,
    )
    from .hybrid_meteor_inference import (
        evaluate_with_probs,
        fuse_tree_year_probs_simple,
        fuse_tree_year_probs_temporal,
    )
    from .hybrid_meteor_io import load_preprocessed_cache, save_preprocessed_cache
    from .hybrid_meteor_model import HybridYearLSTMClassifier
except ImportError:
    from hybrid_meteor_data import (
        YearHybridDataset,
        add_sentinel2_indices,
        build_tree_year_records_with_annual_metrics_and_era5,
        collate_year_hybrid_batch,
        load_era5_folder,
        make_two_broad_labels,
    )
    from hybrid_meteor_inference import (
        evaluate_with_probs,
        fuse_tree_year_probs_simple,
        fuse_tree_year_probs_temporal,
    )
    from hybrid_meteor_io import load_preprocessed_cache, save_preprocessed_cache
    from hybrid_meteor_model import HybridYearLSTMClassifier

# ============================================================
# Reproducibility
# ============================================================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ============================================================
# Utilities
# ============================================================

def save_confusion_matrix(cm, labels, out_csv, out_png, title):
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)
    cm_df.to_csv(out_csv)

    cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-10)
    cm_norm_df = pd.DataFrame(cm_norm, index=labels, columns=labels)
    cm_norm_df.to_csv(out_csv.replace(".csv", "_normalized.csv"))

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm_norm_df, cmap="Blues", square=True, cbar=True, annot=True, fmt=".2f")
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()

# ============================================================
# Hybrid LSTM model
# ============================================================

class RecallAwareLoss(nn.Module):
    def __init__(self, base_loss, conifer_class_idx=0, fn_weight=0.5):
        super().__init__()
        self.base_loss = base_loss
        self.conifer_class_idx = conifer_class_idx
        self.fn_weight = fn_weight

    def forward(self, logits, targets):
        ce = self.base_loss(logits, targets)

        probs = torch.softmax(logits, dim=1)
        conifer_prob = probs[:, self.conifer_class_idx]

        # Penalize low conifer probability among true conifers.
        conifer_mask = (targets == self.conifer_class_idx).float()

        if conifer_mask.sum() > 0:
            fn_penalty = (conifer_mask * (1.0 - conifer_prob)).sum() / conifer_mask.sum()
        else:
            fn_penalty = torch.tensor(0.0, device=logits.device)

        return ce + self.fn_weight * fn_penalty

# ============================================================
# Training / evaluation
# ============================================================
def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()

    running_loss = 0.0
    n_total = 0
    n_correct = 0
    start_time = time.time()

    for batch in dataloader:
        x = batch["x"].to(device, non_blocking=True)
        lengths = batch["lengths"].to(device, non_blocking=True)
        annual_metrics = batch["annual_metrics"].to(device, non_blocking=True)
        y = batch["y_broad"].to(device, non_blocking=True)

        optimizer.zero_grad()
        logits, _ = model(x, lengths, annual_metrics)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        batch_size = y.size(0)
        running_loss += loss.item() * batch_size
        n_total += batch_size
        n_correct += (logits.argmax(dim=1) == y).sum().item()

    return {
        "loss": running_loss / n_total,
        "acc": n_correct / n_total,
        "time_sec": time.time() - start_time,
    }


def load_or_build_training_data(
    x_path,
    y_path,
    era5_folder,
    cache_path,
    force_rebuild_cache,
    s2_cols,
    era5_cols,
    s2_id_col,
    s2_date_col,
    era5_id_col,
    era5_date_col,
    use_sequence_branch,
    use_annual_metrics,
    ablation_tag,
):
    if os.path.exists(cache_path) and not force_rebuild_cache:
        print("Loading cached dataset...")
        cache = load_preprocessed_cache(cache_path)

        required_cache_keys = [
            "seqs",
            "lengths",
            "annual_metrics",
            "valid_measurements",
            "y_broad",
            "unique_ids",
            "years",
            "genus_names",
            "train_idx",
            "val_idx",
            "test_idx",
            "feature_names",
            "annual_metric_names",
            "broad2_classes",
            "idx_to_broad2",
            "seq_mean",
            "seq_std",
            "annual_metric_mean",
            "annual_metric_std",
        ]

        missing_cache_keys = [k for k in required_cache_keys if k not in cache]
        if missing_cache_keys:
            raise KeyError(
                f"Cache is missing required keys: {missing_cache_keys}. "
                "Set FORCE_REBUILD_CACHE=True once to rebuild the cache."
            )

        cached_ablation_tag = cache.get("ablation_tag")
        if cached_ablation_tag is not None and cached_ablation_tag != ablation_tag:
            raise ValueError(
                f"Loaded cache has ablation_tag={cached_ablation_tag}, "
                f"but current settings require {ablation_tag}. "
                "Use the tagged CACHE_PATH or set FORCE_REBUILD_CACHE=True."
            )

        return cache

    print("Building dataset from raw inputs...")

    x_df = pd.read_csv(x_path)
    x_df[s2_date_col] = pd.to_datetime(x_df[s2_date_col]).dt.normalize()
    x_df = add_sentinel2_indices(x_df)

    y_df = pd.read_csv(y_path)
    label_df, broad2_meta = make_two_broad_labels(
        y_df=y_df,
        unique_id_col=s2_id_col,
        genus_col="BOTANICALG",
    )

    print("\n2-class label counts:")
    print(label_df["broad2_label"].value_counts())

    era5_df = load_era5_folder(
        era5_folder=era5_folder,
        id_col=era5_id_col,
        date_col=era5_date_col,
    )

    era5_df = era5_df.rename(columns={era5_id_col: s2_id_col, era5_date_col: s2_date_col})

    missing_era5 = [c for c in era5_cols if c not in era5_df.columns]
    if missing_era5:
        raise ValueError(f"ERA5 data is missing columns: {missing_era5}")

    x_df = x_df.merge(
        era5_df[[s2_id_col, s2_date_col] + era5_cols],
        on=[s2_id_col, s2_date_col],
        how="inner",
    )

    print(f"\nRows after Sentinel-ERA5 inner join: {len(x_df)}")

    x_df = x_df[x_df[s2_id_col].isin(set(label_df[s2_id_col]))].copy()

    year_df, feature_names, annual_metric_names = build_tree_year_records_with_annual_metrics_and_era5(
        df=x_df,
        s2_cols=s2_cols,
        era5_cols=era5_cols,
        id_col=s2_id_col,
        date_col=s2_date_col,
        add_band_mask=True,
        fill_value=0.0,
    )

    year_df = year_df.merge(
        label_df[[s2_id_col, "BOTANICALG", "broad2_idx", "broad2_label"]],
        on=s2_id_col,
        how="inner",
    )

    broad2_classes = broad2_meta["broad2_classes"]
    idx_to_broad2 = broad2_meta["idx_to_broad2"]

    unique_tree_df = label_df[[s2_id_col, "broad2_idx"]].copy()

    train_ids, temp_ids = train_test_split(
        unique_tree_df[s2_id_col].to_numpy(),
        test_size=0.20,
        random_state=42,
        stratify=unique_tree_df["broad2_idx"].to_numpy(),
    )

    temp_broad_idx = unique_tree_df.set_index(s2_id_col).loc[temp_ids, "broad2_idx"].to_numpy()

    val_ids, test_ids = train_test_split(
        temp_ids,
        test_size=0.50,
        random_state=42,
        stratify=temp_broad_idx,
    )

    train_mask = year_df[s2_id_col].isin(train_ids)

    if train_mask.sum() == 0:
        raise ValueError("No training rows were found for the generated train split.")

    train_annual_rows = np.stack(year_df.loc[train_mask, "annual_metrics"].to_numpy())
    annual_metric_mean = train_annual_rows.mean(axis=0)
    annual_metric_std = train_annual_rows.std(axis=0)
    annual_metric_std[annual_metric_std == 0] = 1.0

    year_df["annual_metrics"] = year_df["annual_metrics"].apply(
        lambda arr: ((arr - annual_metric_mean) / annual_metric_std).astype(np.float32)
    )

    train_seq_list = year_df.loc[train_mask, "seq"].to_list()
    train_seq_rows = np.concatenate(train_seq_list, axis=0)

    seq_mean = train_seq_rows.mean(axis=0)
    seq_std = train_seq_rows.std(axis=0)
    seq_std[seq_std == 0] = 1.0

    year_df["seq"] = year_df["seq"].apply(
        lambda arr: ((arr - seq_mean) / seq_std).astype(np.float32)
    )

    seqs = year_df["seq"].tolist()
    lengths = year_df["length"].to_numpy(dtype=np.int64)
    n_rows = len(year_df)

    if use_annual_metrics:
        annual_metrics = np.stack(year_df["annual_metrics"].to_numpy()).astype(np.float32)
    else:
        annual_metrics = np.zeros((n_rows, 0), dtype=np.float32)
        annual_metric_names = []

    payload = {
        "seqs": seqs,
        "lengths": lengths,
        "annual_metrics": annual_metrics,
        "valid_measurements": year_df["valid_measurements"].to_numpy(dtype=np.float32),
        "y_broad": year_df["broad2_idx"].to_numpy(dtype=np.int64),
        "unique_ids": year_df[s2_id_col].to_numpy(),
        "years": year_df["year"].to_numpy(),
        "genus_names": year_df["BOTANICALG"].to_numpy(),
        "train_idx": np.where(year_df[s2_id_col].isin(train_ids))[0],
        "val_idx": np.where(year_df[s2_id_col].isin(val_ids))[0],
        "test_idx": np.where(year_df[s2_id_col].isin(test_ids))[0],
        "feature_names": feature_names,
        "annual_metric_names": annual_metric_names,
        "broad2_classes": broad2_classes,
        "idx_to_broad2": idx_to_broad2,
        "seq_mean": seq_mean,
        "seq_std": seq_std,
        "annual_metric_mean": annual_metric_mean,
        "annual_metric_std": annual_metric_std,
        "use_sequence_branch": use_sequence_branch,
        "use_annual_metrics": use_annual_metrics,
        "ablation_tag": ablation_tag,
    }

    save_preprocessed_cache(cache_path, payload)
    print(f"Saved cache to {cache_path}")
    return payload


def run_final_model_analysis(
    model,
    train_loader,
    val_loader,
    test_loader,
    criterion,
    device,
    broad2_classes,
    idx_to_broad2,
    num_classes,
    train_idx,
    val_idx,
    test_idx,
    input_dim,
    annual_metric_dim,
    s2_cols,
    era5_cols,
    batch_size,
    learning_rate,
    num_epochs,
    patience,
    use_recall_aware_loss,
    recall_fn_weight,
    use_sequence_branch,
    use_annual_metrics,
    ablation_tag,
    out_dir,
    history_path,
    results_path,
    val_row_pred_path,
    test_row_pred_path,
    val_tree_pred_path,
    test_tree_pred_path,
    metadata_path,
):
    val_metrics = evaluate_with_probs(
        model,
        val_loader,
        criterion,
        device,
    )

    test_metrics = evaluate_with_probs(
        model,
        test_loader,
        criterion,
        device,
    )

    val_tree_df, val_tree_acc = fuse_tree_year_probs_temporal(
        unique_ids=val_metrics["unique_ids"],
        years=val_metrics["years"],
        y_true=val_metrics["y_true"],
        probs=val_metrics["probs"],
        weights=val_metrics["valid_measurements"],
        class_names=broad2_classes
    )

    test_tree_df, test_tree_acc = fuse_tree_year_probs_temporal(
        unique_ids=test_metrics["unique_ids"],
        years=test_metrics["years"],
        y_true=test_metrics["y_true"],
        probs=test_metrics["probs"],
        weights=test_metrics["valid_measurements"],
        class_names=broad2_classes
    )

    print(f"\nValidation accuracy (row-level):  {val_metrics['acc']:.4f}")
    print(f"Validation accuracy (tree-fused): {val_tree_acc:.4f}")
    print(f"Test accuracy (row-level):        {test_metrics['acc']:.4f}")
    print(f"Test accuracy (tree-fused):       {test_tree_acc:.4f}")

    print("\nTest classification report (row-level):")
    print(classification_report(
        test_metrics["y_true"],
        test_metrics["y_pred"],
        target_names=broad2_classes,
        digits=4
    ))

    precision, recall, f1, support = precision_recall_fscore_support(
        test_tree_df["y_true"].to_numpy(),
        test_tree_df["y_pred_fused"].to_numpy(),
        labels=list(range(num_classes)),
        zero_division=0
    )
    metric_df_tree = pd.DataFrame({
        "class_idx": list(range(num_classes)),
        "class_name": broad2_classes,
        "precision_tree_fused": precision,
        "recall_tree_fused": recall,
        "f1_tree_fused": f1,
        "support_tree_fused": support,
    })
    print("\nPer-class metrics (tree-fused):")
    print(metric_df_tree)

    

    val_row_df = pd.DataFrame({
        "uniqueID": val_metrics["unique_ids"],
        "year": val_metrics["years"],
        "genus_name": val_metrics["genus_names"],
        "valid_measurements": val_metrics["valid_measurements"],
        "y_true": val_metrics["y_true"],
        "y_pred": val_metrics["y_pred"],
    })
    val_row_df["y_true_name"] = val_row_df["y_true"].map(idx_to_broad2)
    val_row_df["y_pred_name"] = val_row_df["y_pred"].map(idx_to_broad2)
    for i, cls in enumerate(broad2_classes):
        val_row_df[f"prob_{cls}"] = val_metrics["probs"][:, i]
    val_row_df.to_csv(val_row_pred_path, index=False)

    test_row_df = pd.DataFrame({
        "uniqueID": test_metrics["unique_ids"],
        "year": test_metrics["years"],
        "genus_name": test_metrics["genus_names"],
        "valid_measurements": test_metrics["valid_measurements"],
        "y_true": test_metrics["y_true"],
        "y_pred": test_metrics["y_pred"],
    })
    test_row_df["y_true_name"] = test_row_df["y_true"].map(idx_to_broad2)
    test_row_df["y_pred_name"] = test_row_df["y_pred"].map(idx_to_broad2)
    for i, cls in enumerate(broad2_classes):
        test_row_df[f"prob_{cls}"] = test_metrics["probs"][:, i]
    test_row_df.to_csv(test_row_pred_path, index=False)

    train_metrics = evaluate_with_probs(
        model,
        train_loader,
        criterion,
        device,
    )

    train_tree_df, train_tree_acc = fuse_tree_year_probs_simple(
        unique_ids=train_metrics["unique_ids"],
        years=train_metrics["years"],
        y_true=train_metrics["y_true"],
        probs=train_metrics["probs"],
        weights=train_metrics["valid_measurements"],
        class_names=broad2_classes,
    )

    if "unique_id" in train_tree_df.columns:
        train_tree_df = train_tree_df.rename(columns={"unique_id": "uniqueID"})

    TRAIN_ROW_PRED_PATH = os.path.join(
        out_dir,
        "hybrid_lstm_broad2_treefused_era5_train_row_predictionsv5.csv"
    )

    train_row_df = pd.DataFrame({
        "uniqueID": train_metrics["unique_ids"],
        "year": train_metrics["years"],
        "genus_name": train_metrics["genus_names"],
        "valid_measurements": train_metrics["valid_measurements"],
        "y_true": train_metrics["y_true"],
        "y_pred": train_metrics["y_pred"],
    })

    train_row_df["y_true_name"] = train_row_df["y_true"].map(idx_to_broad2)
    train_row_df["y_pred_name"] = train_row_df["y_pred"].map(idx_to_broad2)

    for i, cls in enumerate(broad2_classes):
        train_row_df[f"prob_{cls}"] = train_metrics["probs"][:, i]

    train_row_df.to_csv(TRAIN_ROW_PRED_PATH, index=False)

    TRAIN_TREE_PRED_PATH = os.path.join(
        out_dir,
        "hybrid_lstm_broad2_treefused_era5_train_tree_predictionsv5.csv"
    )

    train_tree_df.to_csv(TRAIN_TREE_PRED_PATH, index=False)

    if "unique_id" in val_tree_df.columns:
        val_tree_df = val_tree_df.rename(columns={"unique_id": "uniqueID"})
    if "unique_id" in test_tree_df.columns:
        test_tree_df = test_tree_df.rename(columns={"unique_id": "uniqueID"})

    val_tree_df.to_csv(val_tree_pred_path, index=False)
    test_tree_df.to_csv(test_tree_pred_path, index=False)

    PLOT_DIR = os.path.join(out_dir, "conifer_temporal_stability_plots")
    os.makedirs(PLOT_DIR, exist_ok=True)


    pd.DataFrame([{
        "val_acc_row": val_metrics["acc"],
        "val_acc_tree_fused": val_tree_acc,
        "test_acc_row": test_metrics["acc"],
        "test_acc_tree_fused": test_tree_acc,
        "n_train_rows": len(train_idx),
        "n_val_rows": len(val_idx),
        "n_test_rows": len(test_idx),
        "n_val_trees": len(val_tree_df),
        "n_test_trees": len(test_tree_df),
        "n_classes": num_classes,
        "input_dim": input_dim,
        "annual_metric_dim": annual_metric_dim,
    }]).to_csv(results_path, index=False)

    cm_val_tree = confusion_matrix(
        val_tree_df["y_true"].to_numpy(),
        val_tree_df["y_pred_fused"].to_numpy(),
        labels=list(range(num_classes))
    )
    save_confusion_matrix(
        cm=cm_val_tree,
        labels=broad2_classes,
        out_csv=os.path.join(out_dir, f"hybrid_lstm_broad2_treefused_era5_val_confusionv5_{ablation_tag}.csv"),
        out_png=os.path.join(out_dir, f"hybrid_lstm_broad2_treefused_era5_val_confusionv5_{ablation_tag}.png"),
        title="ERA5 Hybrid LSTM Tree-Fused Confusion Matrix - Validation",
    )

    cm_test_tree = confusion_matrix(
        test_tree_df["y_true"].to_numpy(),
        test_tree_df["y_pred_fused"].to_numpy(),
        labels=list(range(num_classes))
    )
    save_confusion_matrix(
        cm=cm_test_tree,
        labels=broad2_classes,
        out_csv=os.path.join(out_dir, f"hybrid_lstm_broad2_treefused_era5_test_confusionv5_{ablation_tag}.csv"),
        out_png=os.path.join(out_dir, f"hybrid_lstm_broad2_treefused_era5_test_confusionv5_{ablation_tag}.png"),
        title="ERA5 Hybrid LSTM Tree-Fused Confusion Matrix - Test",
    )

    with open(metadata_path, "w") as f:
        json.dump({
            "broad2_classes": broad2_classes,
            "s2_cols": s2_cols,
            "era5_cols": era5_cols,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "num_epochs": num_epochs,
            "patience": patience,
            "input_dim": int(input_dim),
            "annual_metric_dim": int(annual_metric_dim),
            "fusion_weighting": "year predictions weighted by valid Sentinel-2 measurements",
            "use_recall_aware_loss": use_recall_aware_loss,
            "recall_fn_weight": recall_fn_weight,
            "sequence_pooling": "attention + mean + max pooling",
            "use_sequence_branch": use_sequence_branch,
            "use_annual_metrics": use_annual_metrics,
            "ablation_tag": ablation_tag,
        }, f, indent=2)

    print("\nSaved files:")
    print(history_path)
    print(results_path)
    print(val_row_pred_path)
    print(test_row_pred_path)
    print(val_tree_pred_path)
    print(test_tree_pred_path)
    print(metadata_path)



# ============================================================
# Main
# ============================================================
def main():
    SEED = 42

    X_PATH = r"C:/users/larki/Desktop/PollenSense/xDataNormalized.csv"
    Y_PATH = r"C:/users/larki/Desktop/PollenSense/yDataInteger.csv"
    ERA5_FOLDER = r"C:/users/larki/Desktop/PollenSense/GIS/ERA5"
    OUT_DIR = r"C:/users/larki/Desktop/"

    S2_ID_COL = "uniqueID"
    S2_DATE_COL = "date"
    ERA5_ID_COL = "uniqueID"
    ERA5_DATE_COL = "date"

    BATCH_SIZE = 256
    LEARNING_RATE = 1e-3
    NUM_EPOCHS = 80
    PATIENCE = 10

    FORCE_REBUILD_CACHE = True
    RUN_TRAINING = True

    USE_RECALL_AWARE_LOSS = True
    RECALL_FN_WEIGHT = 0.5
    CONIFER_EXTRA_WEIGHT = 1.5

    # ------------------------------------------------------------
    # Feature ablation toggles
    # Defaults preserve current behavior
    # ------------------------------------------------------------
    USE_SEQUENCE_BRANCH = False
    USE_ANNUAL_METRICS = True

    ABLATION_TAG = (
        f"seq{int(USE_SEQUENCE_BRANCH)}_"
        f"ann{int(USE_ANNUAL_METRICS)}"
    )

    s2_cols = [
        "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12",
        "NDVI", "GNDVI", "CIg", "CIre", "MTCI", "MCARI",
        "NDVIre1", "NDVIre2", "REPI", "NDII", "MSAVI", "LAI_re", "LAI_ndvi"
    ]

    era5_cols = [
        "temp_mean_7d_c",
        "temp_mean_14d_c",
        "temp_mean_30d_c",
        "precip_sum_7d_mm",
        "precip_sum_14d_mm",
        "precip_sum_30d_mm",
        "srad_sum_7d_j_m2",
        "srad_sum_14d_j_m2",
        "srad_sum_30d_j_m2",
        "gdd_cum_ytd_base10_c",
    ]

    LSTM_HIDDEN_DIM = 128
    LSTM_LAYERS = 2
    SEQ_EMBED_DIM = 128
    ANNUAL_EMBED_DIM = 64
    HYBRID_EMBED_DIM = 128
    DROPOUT = 0.3
    BIDIRECTIONAL = True

    SCHEDULER_FACTOR = 0.5
    SCHEDULER_PATIENCE = 4
    SCHEDULER_THRESHOLD = 1e-3

    CACHE_PATH = os.path.join(OUT_DIR, f"hybrid_lstm_broad2_treefused_era5_cachev5_{ABLATION_TAG}.pkl")
    BEST_MODEL_PATH = os.path.join(OUT_DIR, f"best_hybrid_lstm_broad2_treefused_era5v5_{ABLATION_TAG}.pt")
    HISTORY_PATH = os.path.join(OUT_DIR, f"hybrid_lstm_broad2_treefused_era5_training_historyv5_{ABLATION_TAG}.csv")
    RESULTS_PATH = os.path.join(OUT_DIR, f"hybrid_lstm_broad2_treefused_era5_resultsv5_{ABLATION_TAG}.csv")
    VAL_ROW_PRED_PATH = os.path.join(OUT_DIR, f"hybrid_lstm_broad2_treefused_era5_val_row_predictionsv5_{ABLATION_TAG}.csv")
    TEST_ROW_PRED_PATH = os.path.join(OUT_DIR, f"hybrid_lstm_broad2_treefused_era5_test_row_predictionsv5_{ABLATION_TAG}.csv")
    VAL_TREE_PRED_PATH = os.path.join(OUT_DIR, f"hybrid_lstm_broad2_treefused_era5_val_tree_predictionsv5_{ABLATION_TAG}.csv")
    TEST_TREE_PRED_PATH = os.path.join(OUT_DIR, f"hybrid_lstm_broad2_treefused_era5_test_tree_predictionsv5_{ABLATION_TAG}.csv")
    METADATA_PATH = os.path.join(OUT_DIR, f"hybrid_lstm_broad2_treefused_era5_metadatav5_{ABLATION_TAG}.json")

    set_seed(SEED)
    os.makedirs(OUT_DIR, exist_ok=True)

    data = load_or_build_training_data(
        x_path=X_PATH,
        y_path=Y_PATH,
        era5_folder=ERA5_FOLDER,
        cache_path=CACHE_PATH,
        force_rebuild_cache=FORCE_REBUILD_CACHE,
        s2_cols=s2_cols,
        era5_cols=era5_cols,
        s2_id_col=S2_ID_COL,
        s2_date_col=S2_DATE_COL,
        era5_id_col=ERA5_ID_COL,
        era5_date_col=ERA5_DATE_COL,
        use_sequence_branch=USE_SEQUENCE_BRANCH,
        use_annual_metrics=USE_ANNUAL_METRICS,
        ablation_tag=ABLATION_TAG,
    )

    seqs = data["seqs"]
    lengths = data["lengths"]
    annual_metrics = data["annual_metrics"]
    valid_measurements = data["valid_measurements"]
    y_broad = data["y_broad"]
    unique_ids = data["unique_ids"]
    years = data["years"]
    genus_names = data["genus_names"]

    train_idx = data["train_idx"]
    val_idx = data["val_idx"]
    test_idx = data["test_idx"]

    feature_names = data["feature_names"]
    annual_metric_names = data["annual_metric_names"]
    broad2_classes = data["broad2_classes"]
    idx_to_broad2 = data["idx_to_broad2"]

    input_dim = seqs[0].shape[1]
    annual_metric_dim = annual_metrics.shape[1]
    num_classes = len(broad2_classes)

    dataset = YearHybridDataset(
        seqs=seqs,
        lengths=lengths,
        annual_metrics=annual_metrics,
        valid_measurements=valid_measurements,
        y_broad=y_broad,
        unique_ids=unique_ids,
        years=years,
        genus_names=genus_names,
    )

    train_dataset = Subset(dataset, train_idx)
    val_dataset = Subset(dataset, val_idx)
    test_dataset = Subset(dataset, test_idx)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        collate_fn=collate_year_hybrid_batch,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        collate_fn=collate_year_hybrid_batch,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        collate_fn=collate_year_hybrid_batch,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = HybridYearLSTMClassifier(
        input_dim=input_dim,
        annual_metric_dim=annual_metric_dim,
        num_classes=num_classes,
        lstm_hidden_dim=LSTM_HIDDEN_DIM,
        lstm_layers=LSTM_LAYERS,
        seq_embed_dim=SEQ_EMBED_DIM,
        annual_embed_dim=ANNUAL_EMBED_DIM,
        hybrid_embed_dim=HYBRID_EMBED_DIM,
        dropout=DROPOUT,
        bidirectional=BIDIRECTIONAL,
        use_sequence_branch=USE_SEQUENCE_BRANCH,
        use_annual_metrics=USE_ANNUAL_METRICS,
    ).to(device)

    train_labels = y_broad[train_idx]
    class_counts = pd.Series(train_labels).value_counts().reindex(range(num_classes), fill_value=0)

    class_weights = np.zeros(num_classes, dtype=np.float32)
    nz = class_counts > 0

    # Base inverse-sqrt weighting
    class_weights[nz] = 1.0 / np.sqrt(class_counts[nz])
    class_weights[nz] = class_weights[nz] / class_weights[nz].sum() * nz.sum()

    class_weights[0] *= CONIFER_EXTRA_WEIGHT

    # Renormalize so average weight stays near 1
    class_weights[nz] = class_weights[nz] / class_weights[nz].sum() * nz.sum()

    print("Class counts:")
    print(class_counts)
    print("Class weights:", class_weights)

    base_criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(class_weights, dtype=torch.float32, device=device)
    )

    if USE_RECALL_AWARE_LOSS:
        criterion = RecallAwareLoss(
            base_loss=base_criterion,
            conifer_class_idx=0,
            fn_weight=RECALL_FN_WEIGHT,
        )
    else:
        criterion = base_criterion

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=SCHEDULER_FACTOR,
        patience=SCHEDULER_PATIENCE,
        threshold=SCHEDULER_THRESHOLD,
    )

    best_val_tree_acc = -float("inf")
    epochs_without_improvement = 0
    history = []

    best_val_conifer_f1 = -float("inf")

    if RUN_TRAINING:
        print("\nTraining ERA5-enhanced hybrid LSTM...")
        for epoch in range(NUM_EPOCHS):
            train_metrics = train_one_epoch(model, train_loader, optimizer, criterion, device)
            val_metrics = evaluate_with_probs(model, val_loader, criterion, device)
            val_tree_df, val_tree_acc = fuse_tree_year_probs_temporal(
                unique_ids=val_metrics["unique_ids"],
                years=val_metrics["years"],
                y_true=val_metrics["y_true"],
                probs=val_metrics["probs"],
                weights=val_metrics["valid_measurements"],
                class_names=broad2_classes,
            )

            precision, recall, f1, support = precision_recall_fscore_support(
                val_tree_df["y_true"].to_numpy(),
                val_tree_df["y_pred_fused"].to_numpy(),
                labels=list(range(num_classes)),
                zero_division=0,
            )

            val_conifer_f1 = f1[0]

            scheduler.step(val_conifer_f1)
            current_lr = optimizer.param_groups[0]["lr"]

            print(
                f"Epoch {epoch+1:03d}/{NUM_EPOCHS} | "
                f"lr={current_lr:.6f} | "
                f"train_loss={train_metrics['loss']:.4f} | "
                f"train_acc={train_metrics['acc']:.4f} | "
                f"val_loss={val_metrics['loss']:.4f} | "
                f"val_acc_row={val_metrics['acc']:.4f} | "
                f"val_acc_tree_fused={val_tree_acc:.4f}"
            )

            history.append({
                "epoch": epoch + 1,
                "lr": current_lr,
                "train_loss": train_metrics["loss"],
                "train_acc_row": train_metrics["acc"],
                "val_loss": val_metrics["loss"],
                "val_acc_row": val_metrics["acc"],
                "val_acc_tree_fused": val_tree_acc,
                "val_conifer_f1": val_conifer_f1,
            })

            if val_conifer_f1 > best_val_conifer_f1:
                best_val_conifer_f1 = val_conifer_f1
                best_val_tree_acc = val_tree_acc
                epochs_without_improvement = 0

                torch.save({
                    "model_state_dict": model.state_dict(),
                    "input_dim": int(input_dim),
                    "annual_metric_dim": int(annual_metric_dim),
                    "num_classes": int(num_classes),
                    "feature_names": list(feature_names),
                    "annual_metric_names": list(annual_metric_names),
                    "broad2_classes": list(broad2_classes),
                    "best_val_tree_acc": float(best_val_tree_acc),
                    "best_val_conifer_f1": float(best_val_conifer_f1),
                    "use_sequence_branch": USE_SEQUENCE_BRANCH,
                    "use_annual_metrics": USE_ANNUAL_METRICS,
                    "ablation_tag": ABLATION_TAG,
                }, BEST_MODEL_PATH)
                print(
                    f"  Saved new best model. "
                    f"best_val_conifer_f1={best_val_conifer_f1:.4f} | "
                    f"best_val_tree_acc={best_val_tree_acc:.4f}"
                )
            else:
                epochs_without_improvement += 1
                print(f"  No improvement for {epochs_without_improvement} epoch(s).")

            if epochs_without_improvement >= PATIENCE:
                print(f"Early stopping triggered after {PATIENCE} epochs without improvement.")
                break

        val_tree_df, val_tree_acc = fuse_tree_year_probs_temporal(
                unique_ids=val_metrics["unique_ids"],
                years=val_metrics["years"],
                y_true=val_metrics["y_true"],
                probs=val_metrics["probs"],
                weights=val_metrics["valid_measurements"],
                class_names=broad2_classes
        )

        pd.DataFrame(history).to_csv(HISTORY_PATH, index=False)

    if not os.path.exists(BEST_MODEL_PATH):
        raise FileNotFoundError(
            f"No checkpoint found for this ablation: {BEST_MODEL_PATH}\n"
            "Set RUN_TRAINING=True to train this configuration, or point BEST_MODEL_PATH "
            "to a checkpoint created with the same USE_SEQUENCE_BRANCH and "
            "USE_ANNUAL_METRICS settings."
        )

    checkpoint = torch.load(BEST_MODEL_PATH, map_location=device, weights_only=False)

    checkpoint_tag = checkpoint.get("ablation_tag")
    if checkpoint_tag is not None and checkpoint_tag != ABLATION_TAG:
        raise ValueError(
            f"Checkpoint ablation_tag={checkpoint_tag}, but current settings require {ABLATION_TAG}. "
            "Use a matching checkpoint or retrain this ablation."
        )

    model.load_state_dict(checkpoint["model_state_dict"])

    run_final_model_analysis(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        criterion=criterion,
        device=device,
        broad2_classes=broad2_classes,
        idx_to_broad2=idx_to_broad2,
        num_classes=num_classes,
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        input_dim=input_dim,
        annual_metric_dim=annual_metric_dim,
        s2_cols=s2_cols,
        era5_cols=era5_cols,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        num_epochs=NUM_EPOCHS,
        patience=PATIENCE,
        use_recall_aware_loss=USE_RECALL_AWARE_LOSS,
        recall_fn_weight=RECALL_FN_WEIGHT,
        use_sequence_branch=USE_SEQUENCE_BRANCH,
        use_annual_metrics=USE_ANNUAL_METRICS,
        ablation_tag=ABLATION_TAG,
        out_dir=OUT_DIR,
        history_path=HISTORY_PATH,
        results_path=RESULTS_PATH,
        val_row_pred_path=VAL_ROW_PRED_PATH,
        test_row_pred_path=TEST_ROW_PRED_PATH,
        val_tree_pred_path=VAL_TREE_PRED_PATH,
        test_tree_pred_path=TEST_TREE_PRED_PATH,
        metadata_path=METADATA_PATH,
    )


if __name__ == "__main__":
    main()
