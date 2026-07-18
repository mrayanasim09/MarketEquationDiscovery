"""Build directed quarterly trade snapshots from complete monthly Comext exports.

A directed edge i->j is the sum of i's three observed monthly exports to j in a
calendar quarter, in current euros. Edges with a missing source month are marked
unobserved and are not converted to zero. Raw and log1p weights are saved without
row/symmetric normalization; normalization is a later model-variant decision.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src.transform.common import COUNTRIES_FILE, TRADE_RAW, V2_PROCESSED, ensure_output_dir, require_validated_raw

EDGES_FILE = V2_PROCESSED / "quarterly_trade_edges_directed.csv"
QUARTERS_FILE = V2_PROCESSED / "quarters.json"
ADJACENCY_FILE = V2_PROCESSED / "adjacency_directed_trade_eur.npy"
OBSERVED_MASK_FILE = V2_PROCESSED / "adjacency_directed_observed_mask.npy"
MANIFEST_FILE = V2_PROCESSED / "graph_manifest.json"


def joined(values: pd.Series) -> str:
    return "|".join(sorted(set(values.astype(str))))


def main() -> int:
    require_validated_raw()
    ensure_output_dir()
    countries = json.loads(COUNTRIES_FILE.read_text())
    country_set = set(countries)
    raw = pd.read_csv(TRADE_RAW)
    raw = raw[raw["exporter"].isin(country_set) & raw["importer"].isin(country_set) & raw["exporter"].ne(raw["importer"])].copy()
    raw["period"] = pd.PeriodIndex(raw["observation_period"], freq="M")
    raw["quarter"] = pd.PeriodIndex(raw["period"].array, freq="M").asfreq("Q-DEC")
    grouped = raw.groupby(["exporter", "importer", "quarter"], as_index=False).agg(
        trade_value_eur=("trade_value", "sum"),
        months_observed=("trade_value", "count"),
        monthly_periods=("observation_period", joined),
        source_identifiers=("source_identifier", joined),
        source_release_dates=("source_release_date", joined),
        retrieved_at=("retrieved_at", joined),
    )
    grouped["is_complete_quarter"] = grouped["months_observed"].eq(3)
    grouped.loc[~grouped["is_complete_quarter"], "trade_value_eur"] = float("nan")
    grouped["trade_log1p"] = np.log1p(grouped["trade_value_eur"])
    grouped["quarter"] = grouped["quarter"].astype(str)
    grouped = grouped.sort_values(["quarter", "exporter", "importer"])
    grouped.to_csv(EDGES_FILE, index=False)

    quarters = sorted(grouped["quarter"].unique())
    index = {country: i for i, country in enumerate(countries)}
    adjacency = np.zeros((len(quarters), len(countries), len(countries)), dtype=np.float64)
    observed = np.zeros((len(quarters), len(countries), len(countries)), dtype=bool)
    for t, quarter in enumerate(quarters):
        snapshot = grouped[grouped["quarter"].eq(quarter) & grouped["is_complete_quarter"]]
        for row in snapshot.itertuples(index=False):
            i, j = index[row.exporter], index[row.importer]
            adjacency[t, i, j] = row.trade_value_eur
            observed[t, i, j] = True
    np.save(ADJACENCY_FILE, adjacency)
    np.save(OBSERVED_MASK_FILE, observed)
    manifest = {
        "countries": countries,
        "quarters": quarters,
        "edge_definition": "Directed exporter-to-importer sum of three monthly Comext VALUE_IN_EUROS observations.",
        "weight_files": {
            "trade_eur": str(ADJACENCY_FILE),
            "observed_mask": str(OBSERVED_MASK_FILE),
            "edge_table": str(EDGES_FILE),
        },
        "normalization": "None at transformation stage; raw euro and log1p edge weights are retained for later pre-specified model variants.",
        "missing_pairs": "A false observed-mask entry means no complete three-month source observation; it is not a zero-trade imputation.",
        "directed": True,
        "complete_edges": int(observed.sum()),
        "unobserved_nonself_cells": int(len(quarters) * len(countries) * (len(countries) - 1) - observed.sum()),
    }
    QUARTERS_FILE.write_text(json.dumps(quarters, indent=2) + "\n")
    MANIFEST_FILE.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Wrote {EDGES_FILE}: {len(grouped):,} directed pair-quarter rows; {int(observed.sum()):,} complete edges")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
