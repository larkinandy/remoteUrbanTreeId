"""Stratify final ensemble predictions by taxon provenance and city."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

DISCRIMINATOR_DIR = Path(__file__).resolve().parents[2] / "model" / "discriminator"
if str(DISCRIMINATOR_DIR) not in sys.path:
    sys.path.insert(0, str(DISCRIMINATOR_DIR))

import train_taxon_discriminator as clean


K11_TO_K6 = np.asarray([0, 0, 1, 1, 2, 2, 3, 4, 2, 5, 1], dtype=np.int64)


def metrics(y, p, k):
    y = np.asarray(y, dtype=np.int64)
    p = np.asarray(p, dtype=np.int64)
    cm = np.bincount(y * k + p, minlength=k * k).reshape(k, k)
    support = cm.sum(axis=1)
    predicted = cm.sum(axis=0)
    tp = np.diag(cm)
    denom = 2 * tp + (predicted - tp) + (support - tp)
    per_class_f1 = np.divide(2 * tp, denom, out=np.zeros(k, dtype=float), where=denom > 0)
    represented = support > 0
    return {
        "accuracy": float(tp.sum() / support.sum()),
        "macro_f1": float(per_class_f1[represented].mean()),
        "weighted_f1": float(np.average(per_class_f1[represented], weights=support[represented])),
        "represented_groups": int(represented.sum()),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--predictions", type=Path, required=True)
    p.add_argument("--shard-root", type=Path, required=True)
    p.add_argument("--core-partition", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    a = p.parse_args(); a.output_dir.mkdir(parents=True, exist_ok=True)
    pred = np.load(a.predictions, allow_pickle=True)
    wanted = pd.DataFrame({
        "city_token": pred["city_token"].astype(str), "tree_id": pred["tree_id"].astype(np.int64),
        "crop_index": pred["crop_index"].astype(np.int64), "row": np.arange(len(pred["labels"])),
    })
    pieces = []
    for path in sorted(a.shard_root.glob("*/*_metadata.csv")):
        city = path.name.split("_part", 1)[0].lower()
        if not bool((wanted.city_token == city).any()): continue
        frame = pd.read_csv(path, usecols=lambda c: c in {"tree_id", "crop_index", "scientific_name"}, low_memory=False)
        frame["city_token"] = city; pieces.append(frame)
    lookup = pd.concat(pieces, ignore_index=True).drop_duplicates(["city_token", "tree_id", "crop_index"])
    frame = wanted.merge(lookup, on=["city_token", "tree_id", "crop_index"], how="left", validate="one_to_one").sort_values("row")
    if frame.scientific_name.isna().any(): raise RuntimeError("Missing taxon metadata")
    frame["taxon"] = clean.scientific_label_from_name(frame.scientific_name, "scientific_binomial_or_genus")
    core = set(np.load(a.core_partition, allow_pickle=True)["label_columns"].astype(str))
    frame["taxon_provenance"] = np.where(frame.taxon.isin(core), "core_discriminator_taxon", "rare_assigned_taxon")
    frame["k11_true"] = pred["labels"].astype(np.int64); frame["k11_pred"] = pred["predictions"].astype(np.int64)
    frame["k6_true"] = pred["collapsed_labels"].astype(np.int64); frame["k6_pred"] = pred["collapsed_predictions"].astype(np.int64)
    rows = []
    for provenance, group in frame.groupby("taxon_provenance", sort=True):
        rows.append({"stratum": provenance, "records": len(group), "taxa": group.taxon.nunique(),
                     **{f"k11_{k}": v for k, v in metrics(group.k11_true, group.k11_pred, 11).items()},
                     **{f"k6_{k}": v for k, v in metrics(group.k6_true, group.k6_pred, 6).items()}})
    pd.DataFrame(rows).to_csv(a.output_dir / "performance_by_taxon_provenance.csv", index=False)
    city_rows = []
    for city, group in frame.groupby("city_token", sort=True):
        city_rows.append({"city": city, "records": len(group), "taxa": group.taxon.nunique(),
                          **{f"k11_{k}": v for k, v in metrics(group.k11_true, group.k11_pred, 11).items()},
                          **{f"k6_{k}": v for k, v in metrics(group.k6_true, group.k6_pred, 6).items()}})
    pd.DataFrame(city_rows).sort_values("k6_macro_f1", ascending=False).to_csv(a.output_dir / "performance_by_city.csv", index=False)
    cm = np.bincount(frame.k6_true.to_numpy() * 6 + frame.k6_pred.to_numpy(), minlength=36).reshape(6, 6)
    pd.DataFrame(cm, index=[f"true_{i+1}" for i in range(6)], columns=[f"pred_{i+1}" for i in range(6)]).to_csv(a.output_dir / "collapsed_k6_confusion_counts.csv")
    print(pd.DataFrame(rows).to_string(index=False))
    city = pd.DataFrame(city_rows)
    print("\nCity k6 macro-F1 summary:", city.k6_macro_f1.describe().to_dict())
    print("lowest cities:\n", city.nsmallest(5, "k6_macro_f1")[["city", "records", "taxa", "k6_accuracy", "k6_macro_f1"]].to_string(index=False))


if __name__ == "__main__": main()
