"""Aggregate validated monthly HICP source indices to complete quarterly averages.

No monthly value is filled. A quarterly average is emitted only when all three
native monthly observations are present for a country, variable, and quarter.
Source identifiers, release dates, and retrieval timestamps remain in the output.
"""

from __future__ import annotations

import json

import pandas as pd

from src.transform.common import COUNTRIES_FILE, MACRO_RAW, TRADE_RAW, V2_PROCESSED, ensure_output_dir, require_validated_raw

REQUIRED_VARIABLES = ("HICP_CP00_INDEX", "HICP_CP045_ENERGY_INDEX")
OUT_FILE = V2_PROCESSED / "quarterly_hicp_panel.csv"


def joined(values: pd.Series) -> str:
    return "|".join(sorted(set(values.astype(str))))


def main() -> int:
    require_validated_raw()
    ensure_output_dir()
    raw = pd.read_csv(MACRO_RAW)
    raw = raw[raw["variable"].isin(REQUIRED_VARIABLES)].copy()
    raw["observation_date"] = pd.to_datetime(raw["observation_date"], errors="raise")
    raw["quarter"] = raw["observation_date"].dt.to_period("Q")

    grouped = raw.groupby(["entity_id", "variable", "quarter"], as_index=False).agg(
        index_quarterly_average=("value", "mean"),
        months_observed=("value", "count"),
        source_identifiers=("source_identifier", joined),
        source_release_dates=("source_release_date", joined),
        retrieved_at=("retrieved_at", joined),
    )
    grouped["is_complete_quarter"] = grouped["months_observed"].eq(3)
    grouped.loc[~grouped["is_complete_quarter"], "index_quarterly_average"] = float("nan")
    grouped["quarter"] = grouped["quarter"].astype(str)

    value_wide = grouped.pivot(index=["entity_id", "quarter"], columns="variable", values="index_quarterly_average")
    count_wide = grouped.pivot(index=["entity_id", "quarter"], columns="variable", values="months_observed")
    complete_wide = grouped.pivot(index=["entity_id", "quarter"], columns="variable", values="is_complete_quarter")
    provenance = grouped.groupby(["entity_id", "quarter"], as_index=True).agg(
        source_identifiers=("source_identifiers", joined),
        source_release_dates=("source_release_dates", joined),
        retrieved_at=("retrieved_at", joined),
    )
    panel = provenance.join(value_wide).join(count_wide.add_suffix("_months_observed")).join(complete_wide.add_suffix("_complete"))
    panel = panel.reset_index().sort_values(["entity_id", "quarter"])
    panel = panel.rename(columns={
        "HICP_CP00_INDEX": "hicp_all_items_quarterly_average",
        "HICP_CP045_ENERGY_INDEX": "hicp_energy_quarterly_average",
        "HICP_CP00_INDEX_months_observed": "hicp_all_items_months_observed",
        "HICP_CP045_ENERGY_INDEX_months_observed": "hicp_energy_months_observed",
        "HICP_CP00_INDEX_complete": "hicp_all_items_complete",
        "HICP_CP045_ENERGY_INDEX_complete": "hicp_energy_complete",
    })
    panel.to_csv(OUT_FILE, index=False)

    macro_countries = set(panel.loc[panel["hicp_all_items_complete"] & panel["hicp_energy_complete"], "entity_id"])
    trade = pd.read_csv(TRADE_RAW)
    trade_countries = set(trade["exporter"]) & set(trade["importer"])
    countries = sorted(macro_countries & trade_countries)
    if len(countries) < 20:
        raise ValueError(f"fewer than 20 countries have both complete HICP components and bilateral trade: {len(countries)}")
    COUNTRIES_FILE.write_text(json.dumps(countries, indent=2) + "\n")
    print(f"Wrote {OUT_FILE}: {len(panel):,} country-quarter rows; {len(countries)} aligned macro-trade countries")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
