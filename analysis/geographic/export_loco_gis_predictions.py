#!/usr/bin/env python3
"""Export QA-screened LOCO predictions with coordinates for GIS mapping."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from geographic_utils import boolean_series, record_keys


def parse_args() -> argparse.Namespace:
    """Parse prediction, metadata, QA, and output paths."""
    parser = argparse.ArgumentParser(
        description="Join a LOCO ensemble prediction artifact to projected tree coordinates."
    )
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--tree-matches", type=Path, required=True)
    parser.add_argument("--qa-metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    """Write a seven-column, QA-screened GIS table using one-based group IDs."""
    args = parse_args()
    with np.load(args.predictions, allow_pickle=False) as artifact:
        city = np.asarray(artifact["city_token"], dtype=str)
        prediction_frame = pd.DataFrame(
            {
                "record_key": record_keys(city, artifact["tree_id"], artifact["crop_index"]),
                "tree_id": np.asarray(artifact["tree_id"], dtype=np.int64),
                "labeled_k6_group": np.asarray(artifact["collapsed_labels"], dtype=np.int64) + 1,
                "predicted_k6_group": np.asarray(artifact["collapsed_predictions"], dtype=np.int64) + 1,
                "labeled_k11_group": np.asarray(artifact["labels"], dtype=np.int64) + 1,
                "predicted_k11_group": np.asarray(artifact["predictions"], dtype=np.int64) + 1,
            }
        )

    qa = pd.read_csv(
        args.qa_metadata,
        usecols=["record_key", "qa_flag_low_vegetated_height_with_lidar"],
    )
    qa["qa_flag_low_vegetated_height_with_lidar"] = boolean_series(
        qa["qa_flag_low_vegetated_height_with_lidar"]
    )
    prediction_frame = prediction_frame.merge(qa, on="record_key", validate="one_to_one")
    prediction_frame = prediction_frame.loc[
        ~prediction_frame["qa_flag_low_vegetated_height_with_lidar"]
    ].copy()

    tree_metadata = pd.read_csv(
        args.tree_matches,
        usecols=[
            "tree_id",
            "tree_x",
            "tree_y",
            "scientific_name",
            "crown_epsg",
        ],
    ).rename(columns={"scientific_name": "species"})
    if tree_metadata["tree_id"].duplicated().any():
        raise ValueError("Tree-match metadata contains duplicate tree_id values")
    coordinate_systems = tree_metadata["crown_epsg"].dropna().astype(int).unique()
    if len(coordinate_systems) != 1:
        raise ValueError(f"Expected one projected CRS, found {coordinate_systems.tolist()}")

    output = prediction_frame.merge(tree_metadata, on="tree_id", validate="one_to_one")
    output["k6_match"] = (
        output["labeled_k6_group"] == output["predicted_k6_group"]
    ).astype("int8")
    output["k11_match"] = (
        output["labeled_k11_group"] == output["predicted_k11_group"]
    ).astype("int8")
    columns = [
        "tree_x",
        "tree_y",
        "labeled_k6_group",
        "predicted_k6_group",
        "labeled_k11_group",
        "predicted_k11_group",
        "k6_match",
        "k11_match",
        "species",
    ]
    output = output.loc[:, columns]
    if output.isna().any().any():
        missing = output.isna().sum()
        raise ValueError(f"Missing GIS export values:\n{missing[missing.gt(0)]}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)
    print(f"Wrote {len(output):,} QA-screened records to {args.output}")
    print(f"Coordinate reference system: EPSG:{coordinate_systems[0]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
