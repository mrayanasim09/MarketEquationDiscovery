"""Create leakage-safe expanding rolling-origin metadata and forecast samples.

At origin t, macro features and the directed trade graph both use reference quarter
t-1. Targets are CPI YoY at t+h for h in {1,2,4}. The raw sources are revised
snapshots, so this is a reference-period-lag pseudo-real-time design, not a
historical-vintage reconstruction.
"""

from __future__ import annotations

import json

import pandas as pd

from src.transform.common import COUNTRIES_FILE, HORIZONS, MACRO_LAG_QUARTERS, TRADE_LAG_QUARTERS, V2_PROCESSED, ensure_output_dir, require_validated_raw

FEATURES_FILE = V2_PROCESSED / "quarterly_feature_panel.csv"
GRAPH_QUARTERS_FILE = V2_PROCESSED / "quarters.json"
ORIGINS_FILE = V2_PROCESSED / "forecast_origins.csv"
SAMPLES_FILE = V2_PROCESSED / "forecast_samples.csv"
SPLIT_REPORT_FILE = V2_PROCESSED / "split_report.json"

# Fixed before inspecting model performance. Test has the protocol-minimum 32
# origins; validation is entirely pre-test, and each test fit may expand through
# its preceding origin.
TRAIN_END = pd.Period("2014Q4", freq="Q")
VALIDATION_END = pd.Period("2016Q4", freq="Q")
TEST_START = pd.Period("2017Q1", freq="Q")


def split_for(origin: pd.Period) -> str:
    if origin <= TRAIN_END:
        return "train"
    if origin <= VALIDATION_END:
        return "validation"
    return "test"


def main() -> int:
    require_validated_raw()
    ensure_output_dir()
    features = pd.read_csv(FEATURES_FILE)
    features["period"] = pd.PeriodIndex(features["quarter"], freq="Q")
    countries = json.loads(COUNTRIES_FILE.read_text())
    graph_quarters = {pd.Period(value, freq="Q") for value in json.loads(GRAPH_QUARTERS_FILE.read_text())}
    cpi = features.set_index(["entity_id", "period"])["cpi_yoy"]
    energy = features.set_index(["entity_id", "period"])["energy_cpi_yoy"]
    all_periods = sorted(features["period"].unique())

    origins: list[dict] = []
    samples: list[dict] = []
    for horizon in HORIZONS:
        for origin in all_periods:
            macro_quarter = origin - MACRO_LAG_QUARTERS
            graph_quarter = origin - TRADE_LAG_QUARTERS
            target_quarter = origin + horizon
            if graph_quarter not in graph_quarters:
                continue
            rows = []
            for country in countries:
                macro_cpi = cpi.get((country, macro_quarter))
                macro_energy = energy.get((country, macro_quarter))
                target = cpi.get((country, target_quarter))
                if pd.isna(macro_cpi) or pd.isna(macro_energy) or pd.isna(target):
                    rows = []
                    break
                rows.append({
                    "country": country,
                    "origin_quarter": str(origin),
                    "horizon_quarters": horizon,
                    "target_quarter": str(target_quarter),
                    "macro_feature_quarter": str(macro_quarter),
                    "trade_graph_quarter": str(graph_quarter),
                    "cpi_yoy_input": float(macro_cpi),
                    "energy_cpi_yoy_input": float(macro_energy),
                    "target_cpi_yoy": float(target),
                    "split": split_for(origin),
                })
            if not rows:
                continue
            origins.append({
                "origin_quarter": str(origin),
                "horizon_quarters": horizon,
                "target_quarter": str(target_quarter),
                "macro_feature_quarter": str(macro_quarter),
                "trade_graph_quarter": str(graph_quarter),
                "split": split_for(origin),
                "expanding_train_end": str(origin - 1),
                "country_count": len(rows),
            })
            samples.extend(rows)
    origin_df = pd.DataFrame(origins).sort_values(["horizon_quarters", "origin_quarter"])
    sample_df = pd.DataFrame(samples).sort_values(["horizon_quarters", "origin_quarter", "country"])
    origin_df.to_csv(ORIGINS_FILE, index=False)
    sample_df.to_csv(SAMPLES_FILE, index=False)
    counts = origin_df.groupby(["horizon_quarters", "split"]).size().unstack(fill_value=0)
    report = {
        "design": "Expanding rolling-origin pseudo-real-time forecasts with fixed reference-period lags.",
        "macro_lag_quarters": MACRO_LAG_QUARTERS,
        "trade_lag_quarters": TRADE_LAG_QUARTERS,
        "horizons": list(HORIZONS),
        "fixed_split_boundaries": {"train_end": str(TRAIN_END), "validation_end": str(VALIDATION_END), "test_start": str(TEST_START)},
        "origin_counts": {str(horizon): {split: int(value) for split, value in counts.loc[horizon].items()} for horizon in counts.index},
        "test_origins_per_horizon": {str(horizon): int(counts.loc[horizon].get("test", 0)) for horizon in counts.index},
        "country_count": len(countries),
        "forecast_sample_rows": len(sample_df),
        "leakage_rule": "For every row, macro_feature_quarter and trade_graph_quarter are strictly earlier than origin_quarter; target_quarter is strictly later than origin_quarter.",
    }
    SPLIT_REPORT_FILE.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Wrote {ORIGINS_FILE}: {len(origin_df):,} origins; {SAMPLES_FILE}: {len(sample_df):,} country-origin samples")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
