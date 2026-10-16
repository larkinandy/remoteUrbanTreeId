"""Load and cross-check the frozen figure tables against scientific mappings."""
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def load_taxa(root=ROOT):
    figures = root / "analysis/figures"
    fine = pd.read_csv(figures / "provenance/manuscript_k11_discrimination_space.csv")
    coarse = pd.read_csv(figures / "provenance/manuscript_collapsed_k6_discrimination_space.csv")
    mapping = pd.read_csv(root / "config/frozen_taxon_group_mapping.csv")
    display = json.loads((figures / "figure_grouping.json").read_text())
    scientific = json.loads((root / "config/final_grouping.json").read_text())
    joined = fine.merge(coarse, on="taxon", suffixes=("", "_coarse"), validate="one_to_one")
    joined = joined.merge(mapping, on="taxon", validate="one_to_one")
    if len(joined) != len(fine) or len(joined) != len(coarse) or len(joined) != len(mapping):
        raise ValueError("Frozen tables do not contain the same taxa")
    for column in ("umap_1", "umap_2", "records", "taxon_type"):
        if not joined[column].equals(joined[column + "_coarse"]):
            raise ValueError(f"Panel tables disagree on {column}")
    for row in joined.itertuples():
        expected = display["analytical_to_manuscript_one_based"][row.k11_group - 1]
        if row.group != expected or row.group_coarse != row.collapsed_k6_group:
            raise ValueError(f"Figure group disagrees with frozen mapping: {row.taxon}")
        if scientific["collapse_map_one_based"][row.k11_group - 1] != row.collapsed_k6_group:
            raise ValueError(f"Invalid collapse: {row.taxon}")
        expected_records = (row.core_validation_records if row.taxon_type == "core" else
                            row.rare_development_records + row.rare_held_out_test_records)
        if row.records != expected_records:
            raise ValueError(f"Support disagrees with mapping: {row.taxon}")
    joined["name"] = joined.taxon.str.replace("_", " ").str.capitalize()
    joined = joined.rename(columns={"group": "manuscript_k11_group"})
    return joined, display
