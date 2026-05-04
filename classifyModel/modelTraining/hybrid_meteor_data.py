import glob
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

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
        valid_measurements,
        y_broad=None,
        unique_ids=None,
        years=None,
        genus_names=None,
    ):
        self.seqs = seqs
        self.lengths = np.asarray(lengths, dtype=np.int64)
        self.annual_metrics = np.asarray(annual_metrics, dtype=np.float32)
        self.valid_measurements = np.asarray(valid_measurements, dtype=np.float32)
        n = len(self.seqs)
        self.y_broad = np.asarray(
            np.zeros(n, dtype=np.int64) if y_broad is None else y_broad,
            dtype=np.int64,
        )
        self.unique_ids = np.asarray(
            np.arange(n) if unique_ids is None else unique_ids,
        )
        self.years = np.asarray(
            np.zeros(n, dtype=np.int64) if years is None else years,
        )
        self.genus_names = np.asarray(
            np.full(n, "", dtype=object) if genus_names is None else genus_names,
        )

        if not (
            len(self.seqs) == len(self.lengths) == len(self.annual_metrics) ==
            len(self.valid_measurements) == len(self.unique_ids) ==
            len(self.years) == len(self.genus_names) == n
        ):
            raise ValueError("All inputs must have the same length")

    def __len__(self):
        return len(self.y_broad)

    def __getitem__(self, idx):
        return {
            "seq": torch.tensor(self.seqs[idx], dtype=torch.float32),
            "length": torch.tensor(self.lengths[idx], dtype=torch.long),
            "annual_metrics": torch.tensor(self.annual_metrics[idx], dtype=torch.float32),
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

    x = torch.zeros(batch_size, max_len, feat_dim, dtype=torch.float32)
    lengths = torch.zeros(batch_size, dtype=torch.long)
    annual_metrics = torch.zeros(batch_size, annual_dim, dtype=torch.float32)
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
        valid_measurements[i] = item["valid_measurements"]
        y_broad[i] = item["y_broad"]
        unique_ids.append(item["uniqueID"])
        years.append(item["year"])
        genus_names.append(item["genus_name"])

    return {
        "x": x,
        "lengths": lengths,
        "annual_metrics": annual_metrics,
        "valid_measurements": valid_measurements,
        "y_broad": y_broad,
        "unique_ids": unique_ids,
        "years": years,
        "genus_names": genus_names,
    }

