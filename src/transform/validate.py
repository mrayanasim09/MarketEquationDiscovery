"""Validate v2 processed forecasting data before any model training."""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pandas as pd

from src.transform.common import COUNTRIES_FILE, V2_PROCESSED, require_validated_raw

FEATURES_FILE = V2_PROCESSED / "quarterly_feature_panel.csv"
ORIGINS_FILE = V2_PROCESSED / "forecast_origins.csv"
SAMPLES_FILE = V2_PROCESSED / "forecast_samples.csv"
QUARTERS_FILE = V2_PROCESSED / "quarters.json"
GRAPH_MANIFEST = V2_PROCESSED / "graph_manifest.json"
OUT_FILE = V2_PROCESSED / "transformation_validation.json"


def main() -> int:
    require_validated_raw()
    errors: list[str] = []
    countries = json.loads(COUNTRIES_FILE.read_text())
    graph = json.loads(GRAPH_MANIFEST.read_text())
    graph_quarters = set(json.loads(QUARTERS_FILE.read_text()))
    features = pd.read_csv(FEATURES_FILE)
    origins = pd.read_csv(ORIGINS_FILE)
    samples = pd.read_csv(SAMPLES_FILE)

    if graph["countries"] != countries:
        errors.append("graph country order differs from processed countries.json")
    if set(samples["country"]) != set(countries):
        errors.append("forecast samples do not use exactly the processed country set")
    if samples["macro_feature_quarter"].ge(samples["origin_quarter"]).any():
        errors.append("macro feature quarter is not strictly earlier than forecast origin")
    if samples["trade_graph_quarter"].ge(samples["origin_quarter"]).any():
        errors.append("trade graph quarter is not strictly earlier than forecast origin")
    if samples["target_quarter"].le(samples["origin_quarter"]).any():
        errors.append("target quarter is not strictly later than forecast origin")
    if not set(samples["trade_graph_quarter"]).issubset(graph_quarters):
        errors.append("forecast samples reference a graph quarter without a snapshot")
    per_origin = samples.groupby(["horizon_quarters", "origin_quarter"])["country"].nunique()
    if not per_origin.eq(len(countries)).all():
        errors.append("at least one forecast origin lacks a complete country panel")
    if samples[["cpi_yoy_input", "energy_cpi_yoy_input", "target_cpi_yoy"]].isna().any().any():
        errors.append("forecast samples contain a missing lagged input or target value")

    origin_counts = origins.groupby(["horizon_quarters", "split"]).size().unstack(fill_value=0)
    for horizon in (1, 2, 4):
        if int(origin_counts.loc[horizon].get("test", 0)) < 32:
            errors.append(f"horizon {horizon} has fewer than 32 test origins")
    if errors:
        print("V2 TRANSFORMATION VALIDATION FAILED")
        print("\n".join(f"- {error}" for error in errors))
        return 1

    report = {
        "validated_at": datetime.now(timezone.utc).isoformat(),
        "passed": True,
        "country_count": len(countries),
        "feature_rows": len(features),
        "graph_snapshot_count": len(graph_quarters),
        "forecast_origin_rows": len(origins),
        "forecast_sample_rows": len(samples),
        "origin_counts": {str(h): {split: int(value) for split, value in origin_counts.loc[h].items()} for h in origin_counts.index},
        "leakage_checks": {
            "macro_features_strictly_lagged": True,
            "trade_graphs_strictly_lagged": True,
            "targets_strictly_future": True,
            "all_graph_references_exist": True,
            "every_origin_has_complete_country_panel": True,
        },
        "graph_missingness": {
            "complete_directed_edges": graph["complete_edges"],
            "unobserved_nonself_cells": graph["unobserved_nonself_cells"],
        },
    }
    OUT_FILE.write_text(json.dumps(report, indent=2) + "\n")
    print(f"V2 TRANSFORMATION VALIDATION PASSED: {len(origins)} origins, {len(samples)} country-origin samples")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
