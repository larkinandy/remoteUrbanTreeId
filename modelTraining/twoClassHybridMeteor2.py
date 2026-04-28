import os
import json
import time
import pickle
import random
import glob

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.optim.lr_scheduler import ReduceLROnPlateau

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    precision_recall_fscore_support,
)

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
def save_preprocessed_cache(cache_path: str, payload: dict):
    with open(cache_path, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_preprocessed_cache(cache_path: str) -> dict:
    with open(cache_path, "rb") as f:
        return pickle.load(f)


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
# Height features
# ============================================================
def load_and_prepare_height_features(
    height_path: str,
    unique_ids_train: np.ndarray,
    height_cols: list[str],
    id_col: str = "uniqueID",
):
    height_df = pd.read_csv(height_path)

    needed = [id_col] + height_cols
    missing = [c for c in needed if c not in height_df.columns]
    if missing:
        raise ValueError(f"Missing columns in height file: {missing}")

    height_df = height_df[needed].copy()
    height_df = height_df.drop_duplicates(subset=[id_col])

    for c in height_cols:
        height_df[c] = pd.to_numeric(height_df[c], errors="coerce")

    train_mask = height_df[id_col].isin(set(unique_ids_train))

    train_means = height_df.loc[train_mask, height_cols].mean()
    height_df[height_cols] = height_df[height_cols].fillna(train_means)

    train_means = height_df.loc[train_mask, height_cols].mean()
    train_stds = height_df.loc[train_mask, height_cols].std().replace(0, 1.0)

    height_df[height_cols] = (height_df[height_cols] - train_means) / train_stds

    meta = {
        "height_cols": height_cols,
        "height_train_mean": train_means.to_dict(),
        "height_train_std": train_stds.to_dict(),
    }
    return height_df, meta


# ============================================================
# 2-class broad mapping
# ============================================================
def genus_to_two_broad(genus: str):
    genus = str(genus).strip()

    conifer = {
        "Picea",
        "Pinus",
    }

    broadleaf = {
        "Acer",
        "Tilia",
        "Ulmus",
        "Zelkova",
        "Gleditsia",
        "Carya",
        "Juglans",
        "Carpinus",
        "Celtis",
        "Cercis",
        "Cladrastis",
        "Gymnocladus",
        "Liriodendron",
        "Prunus",
        "Quercus",
        "Ginkgo",
        "Pyrus",
    }

    if genus in conifer:
        return "conifer"
    elif genus in broadleaf:
        return "broadleaf"
    else:
        return None


def make_two_broad_labels(
    y_df: pd.DataFrame,
    unique_id_col: str = "uniqueID",
    genus_col: str = "BOTANICALG",
):
    df = y_df[[unique_id_col, genus_col]].copy()
    df = df.drop_duplicates(subset=[unique_id_col])

    df[genus_col] = df[genus_col].astype(str).str.strip()
    df["broad2_label"] = df[genus_col].apply(genus_to_two_broad)
    df = df[df["broad2_label"].notna()].copy()

    broad2_classes = [
        "conifer",
        "broadleaf",
    ]
    broad2_to_idx = {label: i for i, label in enumerate(broad2_classes)}
    idx_to_broad2 = {i: label for label, i in broad2_to_idx.items()}

    df["broad2_idx"] = df["broad2_label"].map(broad2_to_idx).astype(np.int64)

    meta = {
        "broad2_classes": broad2_classes,
        "broad2_to_idx": broad2_to_idx,
        "idx_to_broad2": idx_to_broad2,
    }
    return df, meta


# ============================================================
# ERA5 loading / joining
# ============================================================
def load_era5_folder(
    era5_folder: str,
    id_col: str = "uniqueID",
    date_col: str = "date",
):
    csv_files = sorted(glob.glob(os.path.join(era5_folder, "*.csv")))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in ERA5 folder: {era5_folder}")

    dfs = []
    for fp in csv_files:
        df = pd.read_csv(fp)
        if id_col not in df.columns or date_col not in df.columns:
            raise ValueError(f"{fp} is missing required columns: {id_col}, {date_col}")
        dfs.append(df)

    era5_df = pd.concat(dfs, ignore_index=True)

    era5_df[date_col] = pd.to_datetime(era5_df[date_col]).dt.normalize()
    era5_df[id_col] = era5_df[id_col]

    era5_df = era5_df.drop_duplicates(subset=[id_col, date_col]).copy()

    return era5_df



# ============================================================
# Build yearly sequences + annual metrics with ERA5
# ============================================================
def build_tree_year_records_with_annual_metrics_and_era5(
    df: pd.DataFrame,
    s2_cols: list[str],
    era5_cols: list[str],
    id_col: str = "uniqueID",
    date_col: str = "date",
    add_band_mask: bool = True,
    fill_value: float = 0.0,
):
    df = df.copy()

    df[date_col] = pd.to_datetime(df[date_col])
    df["year"] = df[date_col].dt.year
    df["month"] = df[date_col].dt.month

    def month_to_season(m):
        if m in [12, 1, 2]:
            return "winter"
        elif m in [3, 4, 5]:
            return "spring"
        elif m in [6, 7, 8]:
            return "summer"
        else:
            return "fall"

    df["season"] = df["month"].apply(month_to_season)

    df = df.sort_values([id_col, "year", date_col])

    # timestep features
    feature_names = s2_cols.copy()
    feature_names += era5_cols.copy()
    feature_names += ["delta_days", "doy_sin", "doy_cos"]
    if add_band_mask:
        feature_names += [f"{c}_mask" for c in s2_cols]

    # annual branch uses mean/min/max for both S2 and ERA5 features
    annual_source_cols = s2_cols + era5_cols
    annual_metric_names = []
    for c in annual_source_cols:
        annual_metric_names.extend([
            f"{c}_mean",
            f"{c}_min",
            f"{c}_max",
            f"{c}_range",
            f"{c}_std",
        ])

    annual_metric_names.extend([
    "NDVI_winter_mean",
    "NDVI_spring_mean",
    "NDVI_summer_mean",
    "NDVI_fall_mean",
    "NDVI_spring_minus_winter",
    "NDVI_fall_minus_summer",
    "NDVI_frac_above_year_mean",
    ])

    
    needed_cols = [id_col, date_col, "year", "season"] + s2_cols + era5_cols
    df = df[needed_cols].copy()

    rows = []

    for uid, g_tree in df.groupby(id_col, sort=False):
        for year, g in g_tree.groupby("year", sort=True):
            g = g.sort_values(date_col)

            s2_vals = g[s2_cols].to_numpy(dtype=np.float32)
            era5_vals = g[era5_cols].to_numpy(dtype=np.float32)

            if add_band_mask:
                band_mask = (~np.isnan(s2_vals)).astype(np.float32)
                valid_measurements = float(band_mask.sum())
            else:
                valid_measurements = float(np.isfinite(s2_vals).sum())

            s2_vals_filled = np.nan_to_num(s2_vals, nan=fill_value)
            era5_vals_filled = np.nan_to_num(era5_vals, nan=fill_value)

            # ------------------------------------------------
            # Optional log-transform for skewed ERA5 variables
            # ------------------------------------------------
            era5_df_tmp = pd.DataFrame(era5_vals_filled, columns=era5_cols)

            log_cols = [
                "precip_sum_7d_mm",
                "precip_sum_14d_mm",
                "precip_sum_30d_mm",
                "srad_sum_7d_j_m2",
                "srad_sum_14d_j_m2",
                "srad_sum_30d_j_m2",
                "gdd_cum_ytd_base10_c",
            ]

            for col in log_cols:
                if col in era5_df_tmp.columns:
                    era5_df_tmp[col] = np.log1p(era5_df_tmp[col].clip(lower=0))

            era5_vals_filled = era5_df_tmp.to_numpy(dtype=np.float32)

            delta_days = g[date_col].diff().dt.days.fillna(0).to_numpy(dtype=np.float32).reshape(-1, 1)
            delta_days = delta_days / 30.0

            doy = g[date_col].dt.dayofyear.to_numpy(dtype=np.float32)
            doy_sin = np.sin(2.0 * np.pi * doy / 365.25).reshape(-1, 1).astype(np.float32)
            doy_cos = np.cos(2.0 * np.pi * doy / 365.25).reshape(-1, 1).astype(np.float32)

            feats = [s2_vals_filled, era5_vals_filled, delta_days, doy_sin, doy_cos]
            if add_band_mask:
                feats.append(band_mask)

            x_i = np.concatenate(feats, axis=1).astype(np.float32)

            metric_parts = []

            for col in annual_source_cols:
                arr = pd.to_numeric(g[col], errors="coerce").to_numpy(dtype=np.float32)
                arr = arr[~np.isnan(arr)]

                if len(arr) == 0:
                    metric_parts.extend([0.0, 0.0, 0.0, 0.0, 0.0])
                else:
                    col_mean = float(arr.mean())
                    col_min = float(arr.min())
                    col_max = float(arr.max())
                    col_range = float(col_max - col_min)
                    col_std = float(arr.std())

                    metric_parts.extend([
                        col_mean,
                        col_min,
                        col_max,
                        col_range,
                        col_std,
                    ])

            # Extra NDVI seasonal/phenology features
            ndvi_tmp = g[["season", "NDVI"]].copy()
            ndvi_tmp["NDVI"] = pd.to_numeric(ndvi_tmp["NDVI"], errors="coerce")
            ndvi_tmp = ndvi_tmp.dropna(subset=["NDVI"])

            if len(ndvi_tmp) == 0:
                metric_parts.extend([0.0] * 7)
            else:
                year_mean = float(ndvi_tmp["NDVI"].mean())
                season_means = ndvi_tmp.groupby("season")["NDVI"].mean()

                winter = float(season_means.get("winter", year_mean))
                spring = float(season_means.get("spring", year_mean))
                summer = float(season_means.get("summer", year_mean))
                fall = float(season_means.get("fall", year_mean))

                metric_parts.extend([
                    winter,
                    spring,
                    summer,
                    fall,
                    spring - winter,
                    fall - summer,
                    float((ndvi_tmp["NDVI"] > year_mean).mean()),
                ])

            annual_metrics = np.asarray(metric_parts, dtype=np.float32)

            rows.append({
                "uniqueID": uid,
                "year": int(year),
                "seq": x_i,
                "length": int(len(g)),
                "coverage": float(len(g)),
                "valid_measurements": valid_measurements,
                "annual_metrics": annual_metrics,
            })

    year_df = pd.DataFrame(rows)
    return year_df, feature_names, annual_metric_names


# ============================================================
# Dataset
# ============================================================
class YearHybridDataset(Dataset):
    def __init__(
        self,
        seqs,
        lengths,
        annual_metrics,
        static_feats,
        valid_measurements,
        y_broad,
        unique_ids,
        years,
        genus_names,
    ):
        self.seqs = seqs
        self.lengths = np.asarray(lengths, dtype=np.int64)
        self.annual_metrics = np.asarray(annual_metrics, dtype=np.float32)
        self.static_feats = np.asarray(static_feats, dtype=np.float32)
        self.valid_measurements = np.asarray(valid_measurements, dtype=np.float32)
        self.y_broad = np.asarray(y_broad, dtype=np.int64)
        self.unique_ids = np.asarray(unique_ids)
        self.years = np.asarray(years)
        self.genus_names = np.asarray(genus_names)

        n = len(self.y_broad)
        if not (
            len(self.seqs) == len(self.lengths) == len(self.annual_metrics) ==
            len(self.static_feats) == len(self.valid_measurements) ==
            len(self.unique_ids) == len(self.years) == len(self.genus_names) == n
        ):
            raise ValueError("All inputs must have the same length")

    def __len__(self):
        return len(self.y_broad)

    def __getitem__(self, idx):
        return {
            "seq": torch.tensor(self.seqs[idx], dtype=torch.float32),
            "length": torch.tensor(self.lengths[idx], dtype=torch.long),
            "annual_metrics": torch.tensor(self.annual_metrics[idx], dtype=torch.float32),
            "static_feats": torch.tensor(self.static_feats[idx], dtype=torch.float32),
            "valid_measurements": torch.tensor(self.valid_measurements[idx], dtype=torch.float32),
            "y_broad": torch.tensor(self.y_broad[idx], dtype=torch.long),
            "uniqueID": self.unique_ids[idx],
            "year": self.years[idx],
            "genus_name": self.genus_names[idx],
        }


def collate_year_hybrid_batch(batch):
    batch_size = len(batch)
    feat_dim = batch[0]["seq"].shape[1]
    max_len = max(item["seq"].shape[0] for item in batch)
    annual_dim = batch[0]["annual_metrics"].shape[0]
    static_dim = batch[0]["static_feats"].shape[0]

    x = torch.zeros(batch_size, max_len, feat_dim, dtype=torch.float32)
    lengths = torch.zeros(batch_size, dtype=torch.long)
    annual_metrics = torch.zeros(batch_size, annual_dim, dtype=torch.float32)
    static_feats = torch.zeros(batch_size, static_dim, dtype=torch.float32)
    valid_measurements = torch.zeros(batch_size, dtype=torch.float32)
    y_broad = torch.zeros(batch_size, dtype=torch.long)

    unique_ids = []
    years = []
    genus_names = []

    for i, item in enumerate(batch):
        t = item["seq"].shape[0]
        x[i, :t, :] = item["seq"]
        lengths[i] = item["length"]
        annual_metrics[i] = item["annual_metrics"]
        static_feats[i] = item["static_feats"]
        valid_measurements[i] = item["valid_measurements"]
        y_broad[i] = item["y_broad"]
        unique_ids.append(item["uniqueID"])
        years.append(item["year"])
        genus_names.append(item["genus_name"])

    return {
        "x": x,
        "lengths": lengths,
        "annual_metrics": annual_metrics,
        "static_feats": static_feats,
        "valid_measurements": valid_measurements,
        "y_broad": y_broad,
        "unique_ids": unique_ids,
        "years": years,
        "genus_names": genus_names,
    }


# ============================================================
# Hybrid LSTM model
# ============================================================
class HybridYearLSTMClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int,
        annual_metric_dim: int,
        static_input_dim: int,
        num_classes: int,
        lstm_hidden_dim: int = 128,
        lstm_layers: int = 2,
        seq_embed_dim: int = 128,
        annual_embed_dim: int = 64,
        static_embed_dim: int = 16,
        hybrid_embed_dim: int = 128,
        dropout: float = 0.3,
        bidirectional: bool = True,
    ):
        super().__init__()

        self.bidirectional = bidirectional

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=lstm_hidden_dim,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        lstm_base_dim = lstm_hidden_dim * (2 if bidirectional else 1)

        self.attn = nn.Sequential(
            nn.Linear(lstm_base_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )

        # attention pooled + mean pooled + max pooled
        pooled_dim = lstm_base_dim * 3

        self.seq_proj = nn.Sequential(
            nn.Linear(pooled_dim, seq_embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.annual_mlp = nn.Sequential(
            nn.Linear(annual_metric_dim, annual_embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.static_mlp = nn.Sequential(
            nn.Linear(static_input_dim, static_embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        combined_dim = seq_embed_dim + annual_embed_dim + static_embed_dim

        self.hybrid_proj = nn.Sequential(
            nn.Linear(combined_dim, hybrid_embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.classifier = nn.Linear(hybrid_embed_dim, num_classes)

    def encode(self, x, lengths, annual_metrics, static_feats):
        packed = pack_padded_sequence(
            x,
            lengths=lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )

        packed_out, _ = self.lstm(packed)
        out, _ = pad_packed_sequence(packed_out, batch_first=True)

        # mask shape: [batch, time]
        mask = torch.arange(out.size(1), device=out.device)[None, :] < lengths[:, None].to(out.device)

        # mean pooling
        mask_f = mask.unsqueeze(-1).float()
        mean_pool = (out * mask_f).sum(dim=1) / lengths.to(out.device).unsqueeze(1).float()

        # max pooling
        out_masked = out.masked_fill(~mask.unsqueeze(-1), -1e9)
        max_pool, _ = out_masked.max(dim=1)

        # attention pooling
        attn_logits = self.attn(out).squeeze(-1)
        attn_logits = attn_logits.masked_fill(~mask, -1e9)
        attn_weights = torch.softmax(attn_logits, dim=1)
        attn_pool = (out * attn_weights.unsqueeze(-1)).sum(dim=1)

        z_seq = torch.cat([attn_pool, mean_pool, max_pool], dim=1)

        z_seq = self.seq_proj(z_seq)
        z_annual = self.annual_mlp(annual_metrics)
        z_static = self.static_mlp(static_feats)

        z = torch.cat([z_seq, z_annual, z_static], dim=1)
        hybrid_emb = self.hybrid_proj(z)
        return hybrid_emb

    def forward(self, x, lengths, annual_metrics, static_feats):
        hybrid_emb = self.encode(x, lengths, annual_metrics, static_feats)
        logits = self.classifier(hybrid_emb)
        return logits, hybrid_emb

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
        static_feats = batch["static_feats"].to(device, non_blocking=True)
        y = batch["y_broad"].to(device, non_blocking=True)

        optimizer.zero_grad()
        logits, _ = model(x, lengths, annual_metrics, static_feats)
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

def predict_from_probs(
    probs,
    use_conifer_threshold=False,
    conifer_threshold=0.5,
    conifer_class_idx=0,
    broadleaf_class_idx=1,
):
    if use_conifer_threshold:
        conifer_prob = probs[:, conifer_class_idx]
        preds = np.where(
            conifer_prob >= conifer_threshold,
            conifer_class_idx,
            broadleaf_class_idx,
        )
        return preds.astype(np.int64)

    return np.argmax(probs, axis=1).astype(np.int64)


@torch.no_grad()
def evaluate_with_probs(
    model,
    dataloader,
    criterion,
    device,
    use_conifer_threshold=False,
    conifer_threshold=0.5,
):
    model.eval()

    running_loss = 0.0
    n_total = 0
    n_correct = 0

    all_true = []
    all_pred = []
    all_probs = []
    all_unique_ids = []
    all_years = []
    all_genus_names = []
    all_valid_measurements = []

    for batch in dataloader:
        x = batch["x"].to(device, non_blocking=True)
        lengths = batch["lengths"].to(device, non_blocking=True)
        annual_metrics = batch["annual_metrics"].to(device, non_blocking=True)
        static_feats = batch["static_feats"].to(device, non_blocking=True)
        y = batch["y_broad"].to(device, non_blocking=True)

        logits, _ = model(x, lengths, annual_metrics, static_feats)
        probs = torch.softmax(logits, dim=1)
        loss = criterion(logits, y)

        probs_np = probs.cpu().numpy()
        preds_np = predict_from_probs(
            probs_np,
            use_conifer_threshold=use_conifer_threshold,
            conifer_threshold=conifer_threshold,
        )

        preds = torch.tensor(preds_np, device=device, dtype=torch.long)

        batch_size = y.size(0)
        running_loss += loss.item() * batch_size
        n_total += batch_size
        n_correct += (preds == y).sum().item()

        all_true.append(y.cpu().numpy())
        all_pred.append(preds.cpu().numpy())
        all_probs.append(probs.cpu().numpy())

        all_unique_ids.extend(batch["unique_ids"])
        all_years.extend(batch["years"])
        all_genus_names.extend(batch["genus_names"])
        all_valid_measurements.extend(batch["valid_measurements"].cpu().numpy().tolist())

    return {
        "loss": running_loss / n_total,
        "acc": n_correct / n_total,
        "y_true": np.concatenate(all_true),
        "y_pred": np.concatenate(all_pred),
        "probs": np.concatenate(all_probs, axis=0),
        "unique_ids": all_unique_ids,
        "years": all_years,
        "genus_names": all_genus_names,
        "valid_measurements": all_valid_measurements,
    }

def predict_tree_conifer_rule(
        fused_prob,
        year_probs,
        params,
        conifer_class_idx=0,
        broadleaf_class_idx=1,
    ):
        """
        fused_prob: (num_classes,)
        year_probs: (num_years, num_classes)
        params: dict of fusion hyperparameters
        """

        conifer_prob = fused_prob[conifer_class_idx]
        year_conifer = year_probs[:, conifer_class_idx]

        max_year = year_conifer.max()
        n_moderate = np.sum(year_conifer >= params["moderate_thr"])
        n_strong = np.sum(year_conifer >= params["strong_thr"])

        if (
            conifer_prob >= params["fused_thr"]
            or max_year >= params["max_thr"]
            or n_strong >= params["min_strong_years"]
            or n_moderate >= params["min_moderate_years"]
        ):
            return conifer_class_idx
        else:
            return broadleaf_class_idx

def tune_tree_fusion_rules(
    unique_ids,
    years,
    y_true,
    probs,
    weights,
    class_names,
):
    grid = []

    fused_thr_vals = [0.45, 0.50, 0.55, 0.60]
    max_thr_vals = [0.90, 0.95, 0.98]
    moderate_thr_vals = [0.50, 0.55, 0.60]
    strong_thr_vals = [0.70, 0.75, 0.80]
    min_moderate_vals = [3, 4, 5]
    min_strong_vals = [1, 2]

    for fused_thr in fused_thr_vals:
        for max_thr in max_thr_vals:
            for moderate_thr in moderate_thr_vals:
                for strong_thr in strong_thr_vals:
                    for min_moderate in min_moderate_vals:
                        for min_strong in min_strong_vals:

                            params = {
                                "fused_thr": fused_thr,
                                "max_thr": max_thr,
                                "moderate_thr": moderate_thr,
                                "strong_thr": strong_thr,
                                "min_moderate_years": min_moderate,
                                "min_strong_years": min_strong,
                            }

                            tree_df, _ = fuse_tree_year_probs_temporal(
                                unique_ids=unique_ids,
                                years=years,
                                y_true=y_true,
                                probs=probs,
                                weights=weights,
                                class_names=class_names,
                            )

                            precision, recall, f1, _ = precision_recall_fscore_support(
                                tree_df["y_true"].to_numpy(),
                                tree_df["y_pred_fused"].to_numpy(),
                                labels=list(range(len(class_names))),
                                zero_division=0,
                            )

                            grid.append({
                                **params,
                                "conifer_precision": precision[0],
                                "conifer_recall": recall[0],
                                "conifer_f1": f1[0],
                                "macro_f1": np.mean(f1),
                            })

    df = pd.DataFrame(grid)

    # 🔥 NEW: enforce minimum precision
    df = df[df["conifer_precision"] >= 0.65]

    if len(df) == 0:
        df = pd.DataFrame(grid)  # fallback if too strict

    best = df.sort_values(
        ["conifer_f1", "conifer_recall"],
        ascending=False,
    ).iloc[0]


    return best.to_dict(), df

# ============================================================
# Tree-fused evaluation weighted by valid Sentinel-2 measurements
# ============================================================
def fuse_tree_year_probs_simple(
    unique_ids,
    years,
    y_true,
    probs,
    weights,
    class_names,
):
    df = pd.DataFrame({
        "unique_id": unique_ids,
        "year": years,
        "y_true": y_true,
        "weight": weights,
    })

    for i in range(probs.shape[1]):
        df[f"p_{i}"] = probs[:, i]

    tree_rows = []

    for uid, g in df.groupby("unique_id"):
        p = g[[f"p_{i}" for i in range(probs.shape[1])]].values
        w = g["weight"].values.astype(float)

        # Avoid divide-by-zero
        if w.sum() == 0:
            w = np.ones_like(w)

        w = w / w.sum()

        # 🔥 Core logic: weighted log-probability fusion
        logp = np.log(np.clip(p, 1e-8, 1.0))
        fused_logp = (w[:, None] * logp).sum(axis=0)

        # Convert back to probabilities (optional but clean)
        fused_prob = np.exp(fused_logp)
        fused_prob = fused_prob / fused_prob.sum()

        pred = int(np.argmax(fused_prob))

        tree_rows.append({
            "unique_id": uid,
            "y_true": int(g["y_true"].iloc[0]),
            "y_pred_fused": pred,
            "prob_conifer": fused_prob[0],
            "total_weight": float(g["weight"].sum()),
        })

    tree_df = pd.DataFrame(tree_rows)

    acc = (tree_df["y_true"] == tree_df["y_pred_fused"]).mean()

    return tree_df, acc

def fuse_tree_year_probs_temporal(
    unique_ids,
    years,
    y_true,
    probs,
    weights,
    class_names,
    conifer_class_idx=0,
    conifer_log_bias=0.05,
    conifer_max_weight=0.0,
):
    df = pd.DataFrame({
        "unique_id": unique_ids,
        "year": years,
        "y_true": y_true,
        "weight": weights,
    })

    probs = np.asarray(probs, dtype=np.float64)

    for i in range(probs.shape[1]):
        df[f"p_{i}"] = probs[:, i]

    rows = []
    prob_cols = [f"p_{i}" for i in range(probs.shape[1])]

    for uid, g in df.groupby("unique_id", sort=False):
        p = g[prob_cols].to_numpy(dtype=np.float64)
        w = g["weight"].to_numpy(dtype=np.float64)

        if w.sum() <= 0:
            w = np.ones_like(w)

        w = w / w.sum()

        logp = np.log(np.clip(p, 1e-8, 1.0))

        # Original stable coverage-weighted log fusion
        fused_logp = (w[:, None] * logp).sum(axis=0)

        # Conifer-only max-year boost
        max_conifer_logp = logp[:, conifer_class_idx].max()
        fused_logp[conifer_class_idx] = (
            (1.0 - conifer_max_weight) * fused_logp[conifer_class_idx]
            + conifer_max_weight * max_conifer_logp
        )

        # Small conifer bias
        fused_logp[conifer_class_idx] += conifer_log_bias

        fused_prob = np.exp(fused_logp - np.max(fused_logp))
        fused_prob = fused_prob / fused_prob.sum()

        pred = int(np.argmax(fused_logp))
        true = int(g["y_true"].iloc[0])

        rows.append({
            "unique_id": uid,
            "n_years": int(len(g)),
            "total_weight": float(g["weight"].sum()),
            "y_true": true,
            "y_pred_fused": pred,
            "y_true_name": class_names[true],
            "y_pred_fused_name": class_names[pred],
            "prob_conifer": float(fused_prob[conifer_class_idx]),
        })

    tree_df = pd.DataFrame(rows)
    acc = (tree_df["y_true"] == tree_df["y_pred_fused"]).mean()

    return tree_df, acc

# ============================================================
# Main
# ============================================================
def main():
    set_seed(42)

    X_PATH = r"C:/users/larki/Desktop/PollenSense/xDataNormalized.csv"
    Y_PATH = r"C:/users/larki/Desktop/PollenSense/yDataInteger.csv"
    HEIGHT_PATH = r"C:/users/larki/Desktop/PollenSense/GIS/meta_trees_height_features.csv"
    DIST_PATH = r"C:/users/larki/Desktop/PollenSense/nearest_opposite_tree_distance.csv"
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

    USE_AUTO_CONIFER_THRESHOLD = False
    MIN_BROADLEAF_RECALL_FOR_THRESHOLD = 0.95

    USE_FIXED_CONIFER_THRESHOLD = False
    FIXED_CONIFER_THRESHOLD = 0.40

    height_cols = [
        "ht_center_1m", "ht_max_3m", "ht_mean_3m", "ht_p25_5m",
        "ht_p50_5m", "ht_p90_5m", "ht_std_3m",
    ]

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

    CACHE_PATH = os.path.join(OUT_DIR, "hybrid_lstm_broad2_treefused_era5_cachev4.pkl")
    BEST_MODEL_PATH = os.path.join(OUT_DIR, "best_hybrid_lstm_broad2_treefused_era5v4.pt")
    HISTORY_PATH = os.path.join(OUT_DIR, "hybrid_lstm_broad2_treefused_era5_training_historyv4.csv")
    RESULTS_PATH = os.path.join(OUT_DIR, "hybrid_lstm_broad2_treefused_era5_resultsv4.csv")
    VAL_ROW_PRED_PATH = os.path.join(OUT_DIR, "hybrid_lstm_broad2_treefused_era5_val_row_predictionsv4.csv")
    TEST_ROW_PRED_PATH = os.path.join(OUT_DIR, "hybrid_lstm_broad2_treefused_era5_test_row_predictionsv4.csv")
    VAL_TREE_PRED_PATH = os.path.join(OUT_DIR, "hybrid_lstm_broad2_treefused_era5_val_tree_predictionsv4.csv")
    TEST_TREE_PRED_PATH = os.path.join(OUT_DIR, "hybrid_lstm_broad2_treefused_era5_test_tree_predictionsv4.csv")
    METADATA_PATH = os.path.join(OUT_DIR, "hybrid_lstm_broad2_treefused_era5_metadatav4.json")
    BEST_FUSION_PARAMS_PATH = os.path.join(OUT_DIR, "best_fusion_params.json")

    os.makedirs(OUT_DIR, exist_ok=True)

    if os.path.exists(CACHE_PATH) and not FORCE_REBUILD_CACHE:
        print("Loading cached dataset...")
        cache = load_preprocessed_cache(CACHE_PATH)

        seqs = cache["seqs"]
        lengths = cache["lengths"]
        annual_metrics = cache["annual_metrics"]
        static_feats = cache["static_feats"]
        valid_measurements = cache["valid_measurements"]
        y_broad = cache["y_broad"]
        unique_ids = cache["unique_ids"]
        years = cache["years"]
        genus_names = cache["genus_names"]

        train_idx = cache["train_idx"]
        val_idx = cache["val_idx"]
        test_idx = cache["test_idx"]

        feature_names = cache["feature_names"]
        annual_metric_names = cache["annual_metric_names"]
        broad2_classes = cache["broad2_classes"]
        idx_to_broad2 = cache["idx_to_broad2"]

    else:
        print("Building dataset from raw inputs...")

        # Sentinel-2
        x_df = pd.read_csv(X_PATH)
        x_df[S2_DATE_COL] = pd.to_datetime(x_df[S2_DATE_COL]).dt.normalize()
        x_df = add_sentinel2_indices(x_df)

        # Labels
        y_df = pd.read_csv(Y_PATH)
        label_df, broad2_meta = make_two_broad_labels(
            y_df=y_df,
            unique_id_col=S2_ID_COL,
            genus_col="BOTANICALG",
        )

        print("\n2-class label counts:")
        print(label_df["broad2_label"].value_counts())

        # ERA5
        era5_df = load_era5_folder(
            era5_folder=ERA5_FOLDER,
            id_col=ERA5_ID_COL,
            date_col=ERA5_DATE_COL,
        )

        # Standardize join columns
        era5_df = era5_df.rename(columns={ERA5_ID_COL: S2_ID_COL, ERA5_DATE_COL: S2_DATE_COL})

        missing_era5 = [c for c in era5_cols if c not in era5_df.columns]
        if missing_era5:
            raise ValueError(f"ERA5 data is missing columns: {missing_era5}")

        # Inner join S2 <-> ERA5 on uniqueID/date
        x_df = x_df.merge(
            era5_df[[S2_ID_COL, S2_DATE_COL] + era5_cols],
            on=[S2_ID_COL, S2_DATE_COL],
            how="inner",
        )

        print(f"\nRows after Sentinel-ERA5 inner join: {len(x_df)}")

        # Keep only labeled trees
        x_df = x_df[x_df[S2_ID_COL].isin(set(label_df[S2_ID_COL]))].copy()

        # Build year records
        year_df, feature_names, annual_metric_names = build_tree_year_records_with_annual_metrics_and_era5(
            df=x_df,
            s2_cols=s2_cols,
            era5_cols=era5_cols,
            id_col=S2_ID_COL,
            date_col=S2_DATE_COL,
            add_band_mask=True,
            fill_value=0.0,
        )

        year_df = year_df.merge(
            label_df[[S2_ID_COL, "BOTANICALG", "broad2_idx", "broad2_label"]],
            on=S2_ID_COL,
            how="inner",
        )

        broad2_classes = broad2_meta["broad2_classes"]
        idx_to_broad2 = broad2_meta["idx_to_broad2"]

        unique_tree_df = label_df[[S2_ID_COL, "broad2_idx"]].copy()

        train_ids, temp_ids = train_test_split(
            unique_tree_df[S2_ID_COL].to_numpy(),
            test_size=0.20,
            random_state=42,
            stratify=unique_tree_df["broad2_idx"].to_numpy(),
        )

        temp_broad_idx = unique_tree_df.set_index(S2_ID_COL).loc[temp_ids, "broad2_idx"].to_numpy()

        val_ids, test_ids = train_test_split(
            temp_ids,
            test_size=0.50,
            random_state=42,
            stratify=temp_broad_idx,
        )

        height_df, _ = load_and_prepare_height_features(
            height_path=HEIGHT_PATH,
            unique_ids_train=train_ids,
            height_cols=height_cols,
            id_col=S2_ID_COL,
        )

        year_df = year_df.merge(
            height_df[[S2_ID_COL] + height_cols],
            on=S2_ID_COL,
            how="left",
        )

        if year_df[height_cols].isna().any().any():
            raise ValueError("Missing height features after merge")
        
        # ------------------------------------------------------------
        # Optional: filter training/evaluation records by distance to
        # nearest opposite-label tree
        # ------------------------------------------------------------
        FILTER_BY_OPPOSITE_DISTANCE = True
        MIN_OPPOSITE_DISTANCE_M = 20.0

        OPPOSITE_DIST_PATH = r"C:/users/larki/Desktop/PollenSense/nearest_opposite_tree_distance.csv"

        if FILTER_BY_OPPOSITE_DISTANCE:
            opp_df = pd.read_csv(OPPOSITE_DIST_PATH)

            if "uniqueId" in opp_df.columns and "uniqueID" not in opp_df.columns:
                opp_df = opp_df.rename(columns={"uniqueId": "uniqueID"})

            required_cols = ["uniqueID", "nearest_opposite_tree_distance_m"]
            missing = [c for c in required_cols if c not in opp_df.columns]
            if missing:
                raise ValueError(f"Opposite-distance file missing columns: {missing}")

            opp_df = opp_df[required_cols].drop_duplicates(subset=["uniqueID"])

            year_df = year_df.merge(
                opp_df,
                on="uniqueID",
                how="left",
            )

            before_rows = len(year_df)
            before_trees = year_df["uniqueID"].nunique()

            year_df = year_df[
                year_df["nearest_opposite_tree_distance_m"].notna()
                & (year_df["nearest_opposite_tree_distance_m"] >= MIN_OPPOSITE_DISTANCE_M)
            ].copy()

            after_rows = len(year_df)
            after_trees = year_df["uniqueID"].nunique()

            print(
                f"\nFiltered by nearest opposite-label tree distance "
                f">= {MIN_OPPOSITE_DISTANCE_M} m"
            )
            print(f"Rows:  {before_rows} -> {after_rows}")
            print(f"Trees: {before_trees} -> {after_trees}")

        # Normalize annual metrics using training trees
        train_mask = year_df[S2_ID_COL].isin(train_ids)

        train_annual_rows = np.stack(year_df.loc[train_mask, "annual_metrics"].to_numpy())
        annual_metric_mean = train_annual_rows.mean(axis=0)
        annual_metric_std = train_annual_rows.std(axis=0)
        annual_metric_std[annual_metric_std == 0] = 1.0

        year_df["annual_metrics"] = year_df["annual_metrics"].apply(
            lambda arr: ((arr - annual_metric_mean) / annual_metric_std).astype(np.float32)
        )

        # ------------------------------------------------
        # Normalize sequence features using training trees
        # ------------------------------------------------
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
        annual_metrics = np.stack(year_df["annual_metrics"].to_numpy())
        static_feats = year_df[height_cols].to_numpy(dtype=np.float32)
        valid_measurements = year_df["valid_measurements"].to_numpy(dtype=np.float32)
        y_broad = year_df["broad2_idx"].to_numpy(dtype=np.int64)
        unique_ids = year_df[S2_ID_COL].to_numpy()
        years = year_df["year"].to_numpy()
        genus_names = year_df["BOTANICALG"].to_numpy()

        train_idx = np.where(year_df[S2_ID_COL].isin(train_ids))[0]
        val_idx = np.where(year_df[S2_ID_COL].isin(val_ids))[0]
        test_idx = np.where(year_df[S2_ID_COL].isin(test_ids))[0]

        payload = {
            "seqs": seqs,
            "lengths": lengths,
            "annual_metrics": annual_metrics,
            "static_feats": static_feats,
            "valid_measurements": valid_measurements,
            "y_broad": y_broad,
            "unique_ids": unique_ids,
            "years": years,
            "genus_names": genus_names,
            "train_idx": train_idx,
            "val_idx": val_idx,
            "test_idx": test_idx,
            "feature_names": feature_names,
            "annual_metric_names": annual_metric_names,
            "broad2_classes": broad2_classes,
            "idx_to_broad2": idx_to_broad2,
            "seq_mean": seq_mean,
            "seq_std": seq_std,
            "annual_metric_mean": annual_metric_mean,
            "annual_metric_std": annual_metric_std,
        }
        save_preprocessed_cache(CACHE_PATH, payload)
        print(f"Saved cache to {CACHE_PATH}")

    input_dim = seqs[0].shape[1]
    annual_metric_dim = annual_metrics.shape[1]
    static_input_dim = static_feats.shape[1]
    num_classes = len(broad2_classes)

    dataset = YearHybridDataset(
        seqs=seqs,
        lengths=lengths,
        annual_metrics=annual_metrics,
        static_feats=static_feats,
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
        static_input_dim=static_input_dim,
        num_classes=num_classes,
        lstm_hidden_dim=128,
        lstm_layers=2,
        seq_embed_dim=128,
        annual_embed_dim=64,
        static_embed_dim=16,
        hybrid_embed_dim=128,
        dropout=0.3,
        bidirectional=True,
    ).to(device)

    train_labels = y_broad[train_idx]
    class_counts = pd.Series(train_labels).value_counts().reindex(range(num_classes), fill_value=0)

    class_weights = np.zeros(num_classes, dtype=np.float32)
    nz = class_counts > 0

    # Base inverse-sqrt weighting
    class_weights[nz] = 1.0 / np.sqrt(class_counts[nz])
    class_weights[nz] = class_weights[nz] / class_weights[nz].sum() * nz.sum()

    # Extra conifer emphasis
    CONIFER_EXTRA_WEIGHT = 1.5  # try 1.25, 1.5, then 2.0
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
        factor=0.5,
        patience=4,
        threshold=1e-3,
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
                    "static_input_dim": int(static_input_dim),
                    "num_classes": int(num_classes),
                    "feature_names": list(feature_names),
                    "annual_metric_names": list(annual_metric_names),
                    "broad2_classes": list(broad2_classes),
                    "best_val_tree_acc": float(best_val_tree_acc),
                    "best_val_conifer_f1": float(best_val_conifer_f1),
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

    checkpoint = torch.load(BEST_MODEL_PATH, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    # ------------------------------------------------------------
    # Tune conifer threshold on validation set
    # ------------------------------------------------------------
    
    selected_conifer_threshold = 0.5

    print(f"\nSelected conifer threshold: {selected_conifer_threshold:.2f}")

    use_threshold_for_eval = USE_AUTO_CONIFER_THRESHOLD or USE_FIXED_CONIFER_THRESHOLD

    val_metrics = evaluate_with_probs(
        model,
        val_loader,
        criterion,
        device,
        use_conifer_threshold=use_threshold_for_eval,
        conifer_threshold=selected_conifer_threshold,
    )

    test_metrics = evaluate_with_probs(
        model,
        test_loader,
        criterion,
        device,
        use_conifer_threshold=use_threshold_for_eval,
        conifer_threshold=selected_conifer_threshold,
    )

    val_tree_df, val_tree_acc = fuse_tree_year_probs_temporal(
        unique_ids=val_metrics["unique_ids"],
        years=val_metrics["years"],
        y_true=val_metrics["y_true"],
        probs=val_metrics["probs"],
        weights=val_metrics["valid_measurements"],
        class_names=broad2_classes
    )

    precision, recall, f1, support = precision_recall_fscore_support(
        val_tree_df["y_true"].to_numpy(),
        val_tree_df["y_pred_fused"].to_numpy(),
        labels=list(range(num_classes)),
        zero_division=0,
    )
    val_conifer_f1 = f1[0]

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
    val_row_df.to_csv(VAL_ROW_PRED_PATH, index=False)

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
    test_row_df.to_csv(TEST_ROW_PRED_PATH, index=False)

    train_metrics = evaluate_with_probs(
        model,
        train_loader,
        criterion,
        device,
        use_conifer_threshold=use_threshold_for_eval,
        conifer_threshold=selected_conifer_threshold,
    )

    train_tree_df, train_tree_acc = fuse_tree_year_probs_simple(
        unique_ids=train_metrics["unique_ids"],
        years=train_metrics["years"],
        y_true=train_metrics["y_true"],
        probs=train_metrics["probs"],
        weights=train_metrics["valid_measurements"],
        class_names=broad2_classes,
    )

    TRAIN_ROW_PRED_PATH = os.path.join(
        OUT_DIR,
        "hybrid_lstm_broad2_treefused_era5_train_row_predictionsv2.csv"
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
        OUT_DIR,
        "hybrid_lstm_broad2_treefused_era5_train_tree_predictionsv2.csv"
    )

    train_tree_df.to_csv(TRAIN_TREE_PRED_PATH, index=False)

    val_tree_df.to_csv(VAL_TREE_PRED_PATH, index=False)
    test_tree_df.to_csv(TEST_TREE_PRED_PATH, index=False)

    PLOT_DIR = os.path.join(OUT_DIR, "conifer_temporal_stability_plots")
    os.makedirs(PLOT_DIR, exist_ok=True)

    def plot_noisy_vs_stable_conifers(row_df, out_dir, n_examples=12):
        df = row_df.copy()
        df = df[df["y_true_name"] == "conifer"].copy()
        df = df.sort_values(["uniqueID", "year"])

        # One row per tree-year expected
        summary = (
            df.groupby("uniqueID")
            .agg(
                genus_name=("genus_name", "first"),
                n_years=("year", "nunique"),
                mean_prob_conifer=("prob_conifer", "mean"),
                std_prob_conifer=("prob_conifer", "std"),
                min_prob_conifer=("prob_conifer", "min"),
                max_prob_conifer=("prob_conifer", "max"),
            )
            .reset_index()
        )

        summary["range_prob_conifer"] = (
            summary["max_prob_conifer"] - summary["min_prob_conifer"]
        )

        summary = summary[summary["n_years"] >= 3].copy()

        stable_ids = (
            summary.sort_values(["std_prob_conifer", "range_prob_conifer"])
            .head(n_examples)["uniqueID"]
            .tolist()
        )

        noisy_ids = (
            summary.sort_values(["std_prob_conifer", "range_prob_conifer"], ascending=False)
            .head(n_examples)["uniqueID"]
            .tolist()
        )

        def plot_group(tree_ids, title, filename):
            plt.figure(figsize=(10, 6))

            for uid in tree_ids:
                g = df[df["uniqueID"] == uid].sort_values("year")
                plt.plot(
                    g["year"],
                    g["prob_conifer"],
                    marker="o",
                    alpha=0.75,
                    label=str(uid),
                )

            plt.axhline(0.5, linestyle="--", linewidth=1)
            plt.ylim(-0.02, 1.02)
            plt.xlabel("Year")
            plt.ylabel("Predicted conifer probability")
            plt.title(title)
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, filename), dpi=300)
            plt.close()

        plot_group(
            stable_ids,
            "Stable true conifers: conifer probability across years",
            "stable_true_conifers.png",
        )

        plot_group(
            noisy_ids,
            "Noisy true conifers: conifer probability across years",
            "noisy_true_conifers.png",
        )

        summary.to_csv(
            os.path.join(out_dir, "conifer_temporal_stability_summary.csv"),
            index=False,
        )

        print("Saved:")
        print(os.path.join(out_dir, "stable_true_conifers.png"))
        print(os.path.join(out_dir, "noisy_true_conifers.png"))
        print(os.path.join(out_dir, "conifer_temporal_stability_summary.csv"))


    plot_noisy_vs_stable_conifers(
        test_row_df,
        PLOT_DIR,
        n_examples=12,
    )

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
    }]).to_csv(RESULTS_PATH, index=False)

    cm_val_tree = confusion_matrix(
        val_tree_df["y_true"].to_numpy(),
        val_tree_df["y_pred_fused"].to_numpy(),
        labels=list(range(num_classes))
    )
    save_confusion_matrix(
        cm=cm_val_tree,
        labels=broad2_classes,
        out_csv=os.path.join(OUT_DIR, "hybrid_lstm_broad2_treefused_era5_val_confusionv2.csv"),
        out_png=os.path.join(OUT_DIR, "hybrid_lstm_broad2_treefused_era5_val_confusionv2.png"),
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
        out_csv=os.path.join(OUT_DIR, "hybrid_lstm_broad2_treefused_era5_test_confusion.csv"),
        out_png=os.path.join(OUT_DIR, "hybrid_lstm_broad2_treefused_era5_test_confusion.png"),
        title="ERA5 Hybrid LSTM Tree-Fused Confusion Matrix - Test",
    )

    with open(METADATA_PATH, "w") as f:
        json.dump({
            "broad2_classes": broad2_classes,
            "height_cols": height_cols,
            "s2_cols": s2_cols,
            "era5_cols": era5_cols,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "num_epochs": NUM_EPOCHS,
            "patience": PATIENCE,
            "input_dim": int(input_dim),
            "annual_metric_dim": int(annual_metric_dim),
            "static_input_dim": int(static_input_dim),
            "fusion_weighting": "year predictions weighted by valid Sentinel-2 measurements",
            "use_recall_aware_loss": USE_RECALL_AWARE_LOSS,
            "recall_fn_weight": RECALL_FN_WEIGHT,
            "use_auto_conifer_threshold": USE_AUTO_CONIFER_THRESHOLD,
            "use_fixed_conifer_threshold": USE_FIXED_CONIFER_THRESHOLD,
            "selected_conifer_threshold": selected_conifer_threshold,
            "min_broadleaf_recall_for_threshold": MIN_BROADLEAF_RECALL_FOR_THRESHOLD,
            "sequence_pooling": "attention + mean + max pooling",
        }, f, indent=2)

    print("\nSaved files:")
    print(HISTORY_PATH)
    print(RESULTS_PATH)
    print(VAL_ROW_PRED_PATH)
    print(TEST_ROW_PRED_PATH)
    print(VAL_TREE_PRED_PATH)
    print(TEST_TREE_PRED_PATH)
    print(METADATA_PATH)


if __name__ == "__main__":
    main()