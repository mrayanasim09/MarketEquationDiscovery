"""Milestone 5: ARIMA and VAR baseline forecasts on frozen dataset_v1."""

from __future__ import annotations

import json
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.vector_ar.var_model import VAR

from src.config import DATA_PROCESSED, RESULTS, ensure_dirs, load_protocol

DATASET_DIR = DATA_PROCESSED / "dataset_v1"
OUT_DIR = RESULTS / "baselines"

ARIMA_ORDER = (0, 1, 1)
VAR_LAGS = 4
VAR_EXCLUDE = {"ARG"}
VAR_COLS = ["cpi_yoy", "gdp_yoy", "neer_chg", "energy_idx"]


def load_nodes() -> pd.DataFrame:
    df = pd.read_csv(DATASET_DIR / "nodes.csv")
    df["quarter_period"] = df["quarter"].apply(lambda q: pd.Period(q, freq="Q"))
    return df.sort_values(["iso3", "quarter_period"])


def quarter_sort_key(q: str) -> pd.Period:
    return pd.Period(q, freq="Q")


def compute_metrics(actual: pd.Series, predicted: pd.Series) -> dict[str, float]:
    mask = actual.notna() & predicted.notna()
    if mask.sum() == 0:
        return {"rmse": np.nan, "mae": np.nan, "mape": np.nan, "n": 0}

    err = actual[mask] - predicted[mask]
    rmse = float(np.sqrt(np.mean(err**2)))
    mae = float(np.mean(np.abs(err)))
    denom = actual[mask].abs()
    mape_mask = denom >= 0.5
    mape = float(np.mean(np.abs(err[mape_mask] / denom[mape_mask]) * 100)) if mape_mask.any() else np.nan
    return {"rmse": rmse, "mae": mae, "mape": mape, "n": int(mask.sum())}


def prepare_var_frame(country_df: pd.DataFrame) -> pd.DataFrame:
    """Complete VAR matrix: interpolate short gaps, drop policy_rate by design."""
    frame = country_df.set_index("quarter_period")[VAR_COLS].copy()
    frame = frame.interpolate(method="linear", limit=2).ffill(limit=2)
    return frame.dropna()


def rolling_arima_forecasts(
    country_df: pd.DataFrame,
    train_end: pd.Period,
) -> pd.DataFrame:
    """One-step-ahead ARIMA(0,1,1) from first val quarter through test."""
    records = []
    series = country_df.set_index("quarter_period")["cpi_yoy"].astype(float)
    eval_quarters = country_df.loc[country_df["split"].isin(["val", "test"]), "quarter_period"].unique()
    eval_quarters = sorted(eval_quarters)

    for q in eval_quarters:
        history = series.loc[series.index <= q].dropna()
        if len(history) < 8:
            continue
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = ARIMA(history, order=ARIMA_ORDER)
                res = model.fit()
                fc = res.forecast(steps=1)
            pred = float(fc.iloc[0]) if hasattr(fc, "iloc") else float(fc[0])
            actual = country_df.loc[country_df["quarter_period"] == q, "cpi_yoy_next"]
            actual_val = float(actual.iloc[0]) if len(actual) else np.nan
            records.append(
                {
                    "iso3": country_df["iso3"].iloc[0],
                    "quarter": str(q),
                    "split": country_df.loc[country_df["quarter_period"] == q, "split"].iloc[0],
                    "actual": actual_val,
                    "arima_pred": pred,
                }
            )
        except Exception as exc:
            records.append(
                {
                    "iso3": country_df["iso3"].iloc[0],
                    "quarter": str(q),
                    "split": country_df.loc[country_df["quarter_period"] == q, "split"].iloc[0],
                    "actual": np.nan,
                    "arima_pred": np.nan,
                    "error": str(exc),
                }
            )
    return pd.DataFrame(records)


def rolling_var_forecasts(country_df: pd.DataFrame) -> pd.DataFrame:
    """One-step-ahead VAR(4) on [cpi_yoy, gdp_yoy, neer_chg, energy_idx]; no policy_rate."""
    records = []
    frame = prepare_var_frame(country_df)
    eval_quarters = country_df.loc[country_df["split"].isin(["val", "test"]), "quarter_period"].unique()
    eval_quarters = sorted(q for q in eval_quarters if q in frame.index)

    for q in eval_quarters:
        history = frame.loc[frame.index <= q]
        if len(history) < VAR_LAGS + 8:
            continue
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = VAR(history)
                res = model.fit(VAR_LAGS)
                fc = res.forecast(history.values[-VAR_LAGS:], steps=1)
            pred = float(fc[0, 0])  # cpi_yoy is first column
            actual = country_df.loc[country_df["quarter_period"] == q, "cpi_yoy_next"]
            actual_val = float(actual.iloc[0]) if len(actual) else np.nan
            records.append(
                {
                    "iso3": country_df["iso3"].iloc[0],
                    "quarter": str(q),
                    "split": country_df.loc[country_df["quarter_period"] == q, "split"].iloc[0],
                    "actual": actual_val,
                    "var_pred": pred,
                }
            )
        except Exception as exc:
            records.append(
                {
                    "iso3": country_df["iso3"].iloc[0],
                    "quarter": str(q),
                    "split": country_df.loc[country_df["quarter_period"] == q, "split"].iloc[0],
                    "actual": np.nan,
                    "var_pred": np.nan,
                    "error": str(exc),
                }
            )
    return pd.DataFrame(records)


def summarize_by_split(forecasts: pd.DataFrame, pred_col: str, exclude_iso3: set | None = None) -> dict:
    out = {}
    for split in ("val", "test", "all"):
        sub = forecasts if split == "all" else forecasts[forecasts["split"] == split]
        if exclude_iso3 is not None:
            sub = sub[~sub["iso3"].isin(exclude_iso3)]
        out[split] = compute_metrics(sub["actual"], sub[pred_col])
    return out


def main() -> None:
    ensure_dirs()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    protocol = load_protocol()
    nodes = load_nodes()

    arima_parts = []
    var_parts = []

    print("Training ARIMA(0,1,1) per country (all 23, including ARG)...")
    for iso3, grp in nodes.groupby("iso3"):
        fc = rolling_arima_forecasts(grp, pd.Period(protocol["splits"]["train_end"], freq="Q"))
        if not fc.empty:
            arima_parts.append(fc)
        print(f"  {iso3}: {len(fc)} forecasts")

    print(f"\nTraining VAR({VAR_LAGS}) per country (excludes {sorted(VAR_EXCLUDE)}, no policy_rate)...")
    print(f"  VAR columns: {VAR_COLS}")
    for iso3, grp in nodes.groupby("iso3"):
        if iso3 in VAR_EXCLUDE:
            print(f"  {iso3}: skipped (VAR exclusion)")
            continue
        fc = rolling_var_forecasts(grp)
        if not fc.empty:
            var_parts.append(fc)
        print(f"  {iso3}: {len(fc)} forecasts")

    arima_fc = pd.concat(arima_parts, ignore_index=True) if arima_parts else pd.DataFrame()
    var_fc = pd.concat(var_parts, ignore_index=True) if var_parts else pd.DataFrame()

    forecasts = arima_fc.merge(
        var_fc[["iso3", "quarter", "var_pred"]],
        on=["iso3", "quarter"],
        how="outer",
    )
    forecasts_path = OUT_DIR / "forecasts.csv"
    forecasts.to_csv(forecasts_path, index=False)

    metrics = {
        "run_at": datetime.now(timezone.utc).isoformat(),
        "milestone": 5,
        "arima": {
            "order": list(ARIMA_ORDER),
            "countries": sorted(nodes["iso3"].unique()),
            "metrics": summarize_by_split(arima_fc, "arima_pred"),
            "metrics_excl_arg": summarize_by_split(arima_fc, "arima_pred", exclude_iso3=VAR_EXCLUDE),
        },
        "var": {
            "lags": VAR_LAGS,
            "columns": VAR_COLS,
            "excluded_countries": sorted(VAR_EXCLUDE),
            "note": "policy_rate dropped; ARG excluded for missing CPI",
            "countries": sorted(set(nodes["iso3"].unique()) - VAR_EXCLUDE),
            "metrics": summarize_by_split(var_fc, "var_pred"),
        },
    }

    metrics_path = OUT_DIR / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))

    print(f"\nSaved {forecasts_path} ({len(forecasts)} rows)")
    print(f"Saved {metrics_path}")
    print("\nTest-set metrics:")
    for model in ("arima", "var"):
        m = metrics[model]["metrics"]["test"]
        print(f"  {model.upper()}: RMSE={m['rmse']:.3f} MAE={m['mae']:.3f} MAPE={m['mape']:.1f}% n={m['n']}")
    print("\nApples-to-apples comparison (excluding Argentina):")
    arima_excl = metrics["arima"]["metrics_excl_arg"]["test"]
    var_excl = metrics["var"]["metrics"]["test"]
    print(f"  ARIMA (excl ARG): RMSE={arima_excl['rmse']:.3f} MAE={arima_excl['mae']:.3f} n={arima_excl['n']}")
    print(f"  VAR (excl ARG):    RMSE={var_excl['rmse']:.3f} MAE={var_excl['mae']:.3f} n={var_excl['n']}")


if __name__ == "__main__":
    main()
