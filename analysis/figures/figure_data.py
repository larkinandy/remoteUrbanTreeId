"""Shared data preparation and export helpers for publication figures."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.figure
import pandas as pd


TRUE_VALUES = frozenset({"1", "true", "t", "yes", "y"})
FALSE_VALUES = frozenset({"0", "false", "f", "no", "n", "", "none", "nan"})


def boolean_series(values: pd.Series) -> pd.Series:
    """Parse common boolean encodings without treating ``'False'`` as true."""
    if pd.api.types.is_bool_dtype(values.dtype):
        return values.fillna(False).astype(bool)
    normalized = values.fillna("").astype(str).str.strip().str.lower()
    unknown = sorted(set(normalized) - TRUE_VALUES - FALSE_VALUES)
    if unknown:
        raise ValueError(f"Unrecognized boolean values: {unknown[:5]}")
    return normalized.isin(TRUE_VALUES)


@dataclass(frozen=True)
class GateWeightTables:
    """Prepared tables shared by gate-weight figure variants."""

    overall: pd.Series
    city: pd.DataFrame
    height: pd.DataFrame
    provenance: pd.DataFrame
    city_ranges: pd.DataFrame


def load_gate_weight_tables(input_dir: Path, minimum_city_support: int) -> GateWeightTables:
    """Load frozen gate summaries and calculate deviations consistently."""
    overall = pd.read_csv(input_dir / "gate_weights_overall.csv").set_index("modality")["mean"]
    city = pd.read_csv(input_dir / "gate_weights_by_city.csv")
    height = pd.read_csv(input_dir / "gate_weights_by_height_class_qa_screened.csv")
    provenance = pd.read_csv(input_dir / "gate_weights_by_taxon_provenance.csv")
    city = city.loc[city["support"].ge(minimum_city_support)].copy()
    city["deviation_pp"] = 100.0 * (city["mean"] - city["modality"].map(overall))
    height = height.loc[height["height_class"].isin(["2–5 m", "5–10 m", ">=10 m"])].copy()
    height["deviation_pp"] = 100.0 * (height["mean"] - height["modality"].map(overall))
    provenance["deviation_pp"] = 100.0 * (provenance["mean"] - provenance["modality"].map(overall))
    city_ranges = city.groupby("modality")["deviation_pp"].agg(["min", "max", "count"])
    return GateWeightTables(overall, city, height, provenance, city_ranges)


def save_figure_formats(
    figure: matplotlib.figure.Figure, output_prefix: Path, dpi: int = 600
) -> list[Path]:
    """Save the repository-standard SVG, PDF, and 600-dpi PNG outputs."""
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    outputs = []
    for extension in ("svg", "pdf", "png"):
        path = output_prefix.with_suffix(f".{extension}")
        figure.savefig(
            path,
            dpi=dpi if extension == "png" else None,
            bbox_inches="tight",
            facecolor="white",
        )
        outputs.append(path)
    return outputs
