import ee
import pandas as pd
import numpy as np
from datetime import timedelta

# ============================================================
# USER SETTINGS
# ============================================================
S2_CSV_PATH = r"C:/users/larki/Desktop/PollenSense/xDataNormalized.csv"
POINT_ASSET = "users/your_username/your_point_asset"   # <-- replace
UNIQUE_ID_FIELD = "uniqueID"                           # <-- replace if needed
DATE_COL = "date"

# Folder/name for Drive export
EXPORT_FOLDER = "GEE_exports"
EXPORT_DESC = "era5land_features_for_valid_s2_days"
EXPORT_PREFIX = "era5land_features_for_valid_s2_days"

# Choose columns used to decide whether an S2 row is "valid"
# If your CSV already only contains valid observations, you can simplify this.
VALIDITY_COLS = ["B2", "B3", "B4", "B8"]

# ERA5-Land settings
ERA5_ID = "ECMWF/ERA5_LAND/HOURLY"
GDD_BASE_C = 10.0
ERA5_SCALE_M = 11132  # native pixel size from catalog

# ============================================================
# INIT EARTH ENGINE
# ============================================================
ee.Initialize()

# ============================================================
# 1) LOAD S2 DATAFRAME AND IDENTIFY VALID DAYS
# ============================================================
s2 = pd.read_csv(S2_CSV_PATH)
s2[DATE_COL] = pd.to_datetime(s2[DATE_COL])

# Keep rows where at least one chosen S2 column is non-null
valid_mask = s2[VALIDITY_COLS].notna().any(axis=1)
s2_valid = s2.loc[valid_mask].copy()

# Unique valid dates across the city
valid_dates = sorted(s2_valid[DATE_COL].dt.normalize().drop_duplicates().tolist())

if len(valid_dates) == 0:
    raise ValueError("No valid Sentinel-2 dates found.")

min_valid_date = valid_dates[0]
max_valid_date = valid_dates[-1]

# Add a 30-day buffer because you want 30-day trailing windows
era5_start = (min_valid_date - timedelta(days=30)).strftime("%Y-%m-%d")
era5_end = (max_valid_date + timedelta(days=1)).strftime("%Y-%m-%d")

print(f"Valid S2 dates: {len(valid_dates)}")
print(f"ERA5 fetch window: {era5_start} to {era5_end}")

# Convert valid dates to strings for EE
valid_date_strs = [d.strftime("%Y-%m-%d") for d in valid_dates]
valid_dates_ee = ee.List(valid_date_strs)

# ============================================================
# 2) LOAD POINT ASSET
# ============================================================
points = ee.FeatureCollection(POINT_ASSET)

# Optional sanity check: keep only the unique ID + geometry
# points = points.select([UNIQUE_ID_FIELD], None, False)

# ============================================================
# 3) LOAD ERA5-LAND HOURLY AND BUILD DAILY COLLECTION
# ============================================================
era5_hourly = (
    ee.ImageCollection(ERA5_ID)
    .filterDate(era5_start, era5_end)
)

# Helper to make one daily image from hourly ERA5-Land
def make_daily_image(date_str):
    """
    Builds one daily image for a YYYY-MM-DD date.
    Outputs:
      - tmean_c   : daily mean 2m air temp (C)
      - precip_mm : daily total precip (mm)
      - srad_j    : daily total downward shortwave radiation (J/m2)
      - gdd_day   : daily growing degree days (base 10 C)
    """
    date = ee.Date(date_str)
    next_date = date.advance(1, "day")

    daily = era5_hourly.filterDate(date, next_date)

    # Daily mean temperature in Celsius
    tmean_c = (
        daily.select("temperature_2m")
        .mean()
        .subtract(273.15)
        .rename("tmean_c")
    )

    # Daily precipitation total in mm from hourly disaggregated precip band
    precip_mm = (
        daily.select("total_precipitation_hourly")
        .sum()
        .multiply(1000.0)
        .rename("precip_mm")
    )

    # Daily downward shortwave radiation total in J/m2 from hourly disaggregated band
    srad_j = (
        daily.select("surface_solar_radiation_downwards_hourly")
        .sum()
        .rename("srad_j")
    )

    # Daily GDD = max(0, tmean_c - base)
    gdd_day = tmean_c.subtract(GDD_BASE_C).max(0).rename("gdd_day")

    return (
        ee.Image.cat([tmean_c, precip_mm, srad_j, gdd_day])
        .set("system:time_start", date.millis())
        .set("date", date.format("YYYY-MM-dd"))
        .set("year", date.get("year"))
    )

# Build daily collection over the full buffered period
all_daily_dates = pd.date_range(start=era5_start, end=(max_valid_date).strftime("%Y-%m-%d"), freq="D")
all_daily_date_strs = [d.strftime("%Y-%m-%d") for d in all_daily_dates]

daily_ic = ee.ImageCollection(
    ee.List(all_daily_date_strs).map(lambda d: make_daily_image(d))
)

# ============================================================
# 4) FUNCTION TO BUILD FEATURE IMAGE FOR ONE VALID S2 DATE
# ============================================================
def build_feature_image_for_valid_date(date_str):
    """
    For one valid Sentinel-2 date, compute:
      temp_mean_7d, 14d, 30d
      precip_sum_7d, 14d, 30d
      srad_sum_7d, 14d, 30d
      gdd_cum_ytd
    """
    date = ee.Date(date_str)
    year_start = ee.Date.fromYMD(date.get("year"), 1, 1)
    next_date = date.advance(1, "day")

    def trailing_mean(band_name, n_days):
        start = date.advance(-(n_days - 1), "day")
        return (
            daily_ic
            .filterDate(start, next_date)
            .select(band_name)
            .mean()
        )

    def trailing_sum(band_name, n_days):
        start = date.advance(-(n_days - 1), "day")
        return (
            daily_ic
            .filterDate(start, next_date)
            .select(band_name)
            .sum()
        )

    # trailing mean temperature
    t_7 = trailing_mean("tmean_c", 7).rename("temp_mean_7d_c")
    t_14 = trailing_mean("tmean_c", 14).rename("temp_mean_14d_c")
    t_30 = trailing_mean("tmean_c", 30).rename("temp_mean_30d_c")

    # trailing precipitation sums
    p_7 = trailing_sum("precip_mm", 7).rename("precip_sum_7d_mm")
    p_14 = trailing_sum("precip_mm", 14).rename("precip_sum_14d_mm")
    p_30 = trailing_sum("precip_mm", 30).rename("precip_sum_30d_mm")

    # trailing downward radiation sums
    r_7 = trailing_sum("srad_j", 7).rename("srad_sum_7d_j_m2")
    r_14 = trailing_sum("srad_j", 14).rename("srad_sum_14d_j_m2")
    r_30 = trailing_sum("srad_j", 30).rename("srad_sum_30d_j_m2")

    # cumulative GDD from Jan 1 through this date
    gdd_cum = (
        daily_ic
        .filterDate(year_start, next_date)
        .select("gdd_day")
        .sum()
        .rename("gdd_cum_ytd_base10_c")
    )

    return (
        ee.Image.cat([t_7, t_14, t_30, p_7, p_14, p_30, r_7, r_14, r_30, gdd_cum])
        .set("date", date.format("YYYY-MM-dd"))
        .set("year", date.get("year"))
    )

# ============================================================
# 5) EXTRACT FEATURES FOR EACH VALID DATE AND EACH POINT
# ============================================================
def extract_for_one_date(date_str):
    feature_img = build_feature_image_for_valid_date(date_str)

    # Sample image values at each point
    sampled = feature_img.sampleRegions(
        collection=points,
        properties=[UNIQUE_ID_FIELD],
        scale=ERA5_SCALE_M,
        geometries=False
    )

    # Add date as a property to every sampled feature
    sampled = sampled.map(
        lambda f: f.set("date", ee.String(date_str))
    )
    return sampled

fc_list = valid_dates_ee.map(lambda d: extract_for_one_date(d))
out_fc = ee.FeatureCollection(fc_list).flatten()

# ============================================================
# 6) EXPORT TO GOOGLE DRIVE
# ============================================================
selectors = [
    UNIQUE_ID_FIELD,
    "date",
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

task = ee.batch.Export.table.toDrive(
    collection=out_fc,
    description=EXPORT_DESC,
    folder=EXPORT_FOLDER,
    fileNamePrefix=EXPORT_PREFIX,
    fileFormat="CSV",
    selectors=selectors,
)

task.start()
print("Export started.")
print("Check the Earth Engine Tasks panel or task status in Python.")