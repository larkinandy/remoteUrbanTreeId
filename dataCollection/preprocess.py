# preprocess.py
# Author: Andrew Larkin
# Date Created: April 13, 2026
# Summary: Preprocess records for deep learning
# Note: ChatGPT was used for code assist

# import libraries
import ee
from datetime import datetime, timedelta
import sys
import os
import pandas as ps
import numpy as np
from dotenv import load_dotenv

# import custom classes and environments
GIT_PATH = "C:/Users/larki/Documents/GitHub/remoteUrbanTreeId/dataCollection/"
sys.path.append(GIT_PATH)

load_dotenv(dotenv_path=GIT_PATH + ".env")

# get list of weekly Sentinel-2 files to combine
# OUTPUTS:
#    df (pandas dataframe) - contains combined Sentinel-2 time series
def getFilesToCombine():
    individualFiles = os.listdir(os.getenv("GEE_LOCAL_FOLDER"))
    pandasArr = []
    for file in individualFiles:
        data = ps.read_csv(os.getenv("GEE_LOCAL_FOLDER")+file)
        pandasArr.append(data)
    df = ps.concat(pandasArr)
    df['date'] = ps.to_datetime(df['date'])
    print(df.head())
    return(df)

# interpolate Sentinel-2 bands when cloud cover results in missing values
# OUTPUTS:
#    df (pandas dataframe) - Sentinel-2 time series with interpolated values 
def interpolateSentinel():
    df = df.sort_values(['uniqueID', 'date']).set_index('date')
    cols = ['B2', 'B3', 'B4', 'B5', 'B6', 'B7','B8','B8A','B11','B12']
    df = (
        df.groupby('uniqueID').apply(lambda g: g.resample('5D').mean())
    )
    df[cols] = (
        df.groupby('uniqueID')[cols].apply(lambda g: g.interpolate(
            method='spline',
            limit=10,              # max gap size
            limit_area='inside'   # no extrapolation
        ))
    )
    df = df.reset_index()
    return(df)

def build_padded_sequences(
        df: ps.DataFrame,
        band_cols: list[str],
        id_col: str = "uniqueID",
        date_col: str = "date",
        add_band_mask: bool = False,
        fill_value: float = 0.0,
    ):
    """
    Build padded sequence tensors from long-format Sentinel-2 dataframe.

    Returns:
        X:         (n_samples, max_time, n_features) float32
        lengths:   (n_samples,) int64
        uids:      (n_samples,) array
        feature_names: list[str]
    """
    df = df.copy()
    df[date_col] = ps.to_datetime(df[date_col])
    df = df.sort_values([id_col, date_col])

    seqs = []
    lengths = []
    uids = []

    feature_names = band_cols.copy()
    feature_names += ["delta_days", "doy_sin", "doy_cos"]

    if add_band_mask:
        feature_names += [f"{c}_mask" for c in band_cols]

    for uid, g in df.groupby(id_col):
        g = g.sort_values(date_col).copy()

        # Raw band values
        vals = g[band_cols].to_numpy(dtype=np.float32)

        # Optional per-band mask before filling NaNs
        if add_band_mask:
            band_mask = (~np.isnan(vals)).astype(np.float32)

        # Fill NaNs in features so tensors are finite
        vals = np.nan_to_num(vals, nan=fill_value)

        # Time features
        delta_days = (
            g[date_col].diff().dt.days.fillna(0).to_numpy(dtype=np.float32).reshape(-1, 1)
        )

        doy = g[date_col].dt.dayofyear.to_numpy(dtype=np.float32)
        doy_sin = np.sin(2.0 * np.pi * doy / 365.25).reshape(-1, 1).astype(np.float32)
        doy_cos = np.cos(2.0 * np.pi * doy / 365.25).reshape(-1, 1).astype(np.float32)

        feats = [vals, delta_days, doy_sin, doy_cos]

        if add_band_mask:
            feats.append(band_mask)

        x_i = np.concatenate(feats, axis=1).astype(np.float32)

        seqs.append(x_i)
        lengths.append(len(g))
        uids.append(uid)

    lengths = np.asarray(lengths, dtype=np.int64)
    uids = np.asarray(uids)

    max_len = int(lengths.max())
    n_samples = len(seqs)
    n_features = seqs[0].shape[1]

    X = np.full((n_samples, max_len, n_features), fill_value, dtype=np.float32)

    for i, arr in enumerate(seqs):
        T = arr.shape[0]
        X[i, :T, :] = arr

    return X, lengths, uids, feature_names

