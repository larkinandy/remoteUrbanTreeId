"""Canonical Sentinel-2 sequence schema shared with model-shard assembly."""

SENTINEL_FEATURE_COLUMNS = [
    "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12",
    "NDVI", "GNDVI", "CIg", "CIre", "MTCI", "MCARI", "NDVIre1", "NDVIre2",
    "REPI", "NDII", "MSAVI", "LAI_re", "LAI_ndvi",
]

SENTINEL_META_COLUMNS = [
    "sentinel_observed",
    "sentinel_interpolated",
    "data_quality_mask",
    "source_image_count",
    "interpolation_gap_days",
    "interpolation_fraction",
    "days_since_previous_observation",
    "days_until_next_observation",
    "delta_days",
    "doy_sin",
    "doy_cos",
]

SENTINEL_SEQUENCE_COLUMNS = SENTINEL_FEATURE_COLUMNS + SENTINEL_META_COLUMNS

SENTINEL_QUALITY_COLUMNS = [
    "sentinel_sequence_length",
    "sentinel_observed_fraction",
    "sentinel_interpolated_fraction",
    "sentinel_mean_data_quality_mask",
    "sentinel_max_interpolation_gap_days",
    "sentinel_mean_interpolation_gap_days",
    "sentinel_mean_source_image_count",
]
