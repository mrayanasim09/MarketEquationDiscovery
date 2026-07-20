"""Journal metrics and origin-level inference for completed v2.1 forecast transactions.

This module implements the pre-registered evaluation protocol for the v2.1
prospective benchmark, including:

- Deterministic scoring (RMSE, MAE, sMAPE) with origin-level aggregation
- Probabilistic scoring (CRPS, interval coverage and width at 80%/95%)
- Diebold-Mariano tests with Harvey-Leybourne-Newbold finite-sample
  correction and Bartlett HAC kernel
- Moving-block bootstrap confidence intervals
- Benjamini-Hochberg FDR correction for multiplicity
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from src.models.storage_v2_1 import DM_COLUMNS, METRIC_COLUMNS, PROVENANCE


def _smape(actual: np.ndarray, mean: np.ndarray) -> np.ndarray:
    """Symmetric Mean Absolute Percentage Error, omitting zero-denominator pairs."""
    denominator = np.abs(actual) + np.abs(mean)
    values = np.full(len(actual), np.nan)
    valid = denominator > 0
    values[valid] = 200.0 * np.abs(actual[valid] - mean[valid]) / denominator[valid]
    return values


def _normal_crps(actual: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """Closed-form CRPS for Normal(mean, scale), with positive finite scales required."""
    if np.any(scale <= 0) or not np.isfinite(scale).all():
        raise ValueError("CRPS requires finite positive predictive scales")
    z = (actual - mean) / scale
    return scale * (z * (2 * stats.norm.cdf(z) - 1) + 2 * stats.norm.pdf(z) - 1 / math.sqrt(math.pi))


def _origin_mean(frame: pd.DataFrame, values: np.ndarray) -> pd.Series:
    """Compute the cross-country mean of *values* within each forecast origin."""
    return pd.Series(values, index=frame.index).groupby(frame["forecast_origin"]).mean().sort_index()


def score_forecasts(forecasts: pd.DataFrame) -> pd.DataFrame:
    """Score a completed forecast table against the v2.1 metric registry.

    Returns a long-format DataFrame with one row per (model, variant, seed,
    horizon, metric) combination, conforming to METRIC_COLUMNS.
    """
    rows: list[dict[str, Any]] = []
    keys = PROVENANCE + ["model", "graph_variant", "seed", "horizon"]
    for values, group in forecasts.groupby(keys, dropna=False):
        provenance = dict(zip(keys, values, strict=True))
        actual, mean = group.actual.to_numpy(float), group["mean"].to_numpy(float)
        abs_loss, square_loss = np.abs(actual - mean), (actual - mean) ** 2
        smape = _smape(actual, mean)
        # Infer a normal scale from the locked 80% interval for proper scoring.
        scale = (group.upper_80.to_numpy(float) - group.lower_80.to_numpy(float)) / (2 * stats.norm.ppf(0.9))
        crps = _normal_crps(actual, mean, scale)
        coverage80 = ((actual >= group.lower_80.to_numpy(float)) & (actual <= group.upper_80.to_numpy(float))).astype(float)
        coverage95 = ((actual >= group.lower_95.to_numpy(float)) & (actual <= group.upper_95.to_numpy(float))).astype(float)
        width80 = group.upper_80.to_numpy(float) - group.lower_80.to_numpy(float)
        width95 = group.upper_95.to_numpy(float) - group.lower_95.to_numpy(float)
        origin_count = int(group.forecast_origin.nunique())
        metrics = {
            "rmse": float(np.sqrt(_origin_mean(group, square_loss).mean())),
            "mae": float(_origin_mean(group, abs_loss).mean()),
            "smape": float(_origin_mean(group, smape).mean()),
            "crps": float(_origin_mean(group, crps).mean()),
            "interval_coverage_80": float(_origin_mean(group, coverage80).mean()),
            "interval_coverage_95": float(_origin_mean(group, coverage95).mean()),
            "interval_width_80": float(_origin_mean(group, width80).mean()),
            "interval_width_95": float(_origin_mean(group, width95).mean()),
        }
        rows.extend({**provenance, "metric": name, "value": value, "origin_count": origin_count} for name, value in metrics.items())
    return pd.DataFrame(rows, columns=METRIC_COLUMNS)


def _dm_hln(difference: np.ndarray, horizon: int) -> tuple[float, float]:
    """Diebold-Mariano test with Harvey-Leybourne-Newbold correction.

    Uses a Bartlett kernel with bandwidth equal to ``horizon - 1`` for
    HAC variance estimation. Returns (statistic, two-sided p-value).
    """
    n, lag = len(difference), min(horizon - 1, len(difference) - 1)
    if n < 3:
        return float("nan"), float("nan")
    centered = difference - difference.mean()
    variance = float(np.mean(centered * centered))
    for step in range(1, lag + 1):
        covariance = float(np.mean(centered[step:] * centered[:-step]))
        variance += 2 * (1 - step / (lag + 1)) * covariance
    if variance <= 0:
        return float("nan"), float("nan")
    statistic = difference.mean() / math.sqrt(variance / n)
    hln = math.sqrt((n + 1 - 2 * horizon + horizon * (horizon - 1) / n) / n)
    statistic *= hln
    return float(statistic), float(2 * stats.t.sf(abs(statistic), df=n - 1))


def _moving_block_ci(values: np.ndarray, block: int, draws: int, rng: np.random.Generator) -> tuple[float, float]:
    """Moving-block bootstrap 95% confidence interval for the mean of *values*.

    Resamples contiguous blocks of length *block* with replacement and
    returns the (2.5th, 97.5th) percentiles of the bootstrap distribution.
    """
    n = len(values)
    if n < 2:
        return float("nan"), float("nan")
    block = min(block, n)
    means = np.empty(draws)
    for draw in range(draws):
        sample: list[float] = []
        while len(sample) < n:
            start = int(rng.integers(0, n - block + 1))
            sample.extend(values[start:start + block])
        means[draw] = np.mean(sample[:n])
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


def _bh_monotone(pvalues: pd.Series) -> pd.Series:
    """Benjamini–Hochberg adjusted p-values with the required monotonicity step."""
    adjusted = pd.Series(np.nan, index=pvalues.index, dtype=float)
    valid = pvalues.dropna().sort_values()
    if valid.empty:
        return adjusted
    raw = valid.to_numpy(float) * len(valid) / np.arange(1, len(valid) + 1)
    monotone = np.minimum.accumulate(raw[::-1])[::-1]
    adjusted.loc[valid.index] = np.minimum(monotone, 1.0)
    return adjusted


def dm_tests(forecasts: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    """Run pairwise Diebold-Mariano tests for all graph models vs. comparators.

    Each graph model × variant × seed × horizon is tested against every
    pre-specified comparator. Seeded comparators (LSTM, TCN) are matched
    by seed to ensure a valid one-to-one merge. Results include HLN-corrected
    DM statistics, moving-block bootstrap CIs, and BH-adjusted p-values.
    """
    rows: list[dict[str, Any]] = []
    comparators = set(config["statistical_testing"]["comparators"])
    graph_models = set(config["graph_models"])
    bootstrap = config["statistical_testing"]["bootstrap"]
    groups = forecasts[forecasts.model.isin(graph_models)].groupby(PROVENANCE + ["model", "graph_variant", "seed", "horizon"], dropna=False)
    for keys, graph in groups:
        provenance = dict(zip(PROVENANCE + ["model", "graph_variant", "seed", "horizon"], keys, strict=True))
        for comparator in sorted(comparators):
            base = forecasts[(forecasts.run_id == provenance["run_id"]) & (forecasts.model == comparator) & (forecasts.horizon == provenance["horizon"])]
            if base.empty:
                raise RuntimeError(f"missing pre-specified comparator {comparator}")
            if comparator in {"lstm", "tcn"}:
                base = base[base.seed == str(provenance["seed"])]
            joined = graph.merge(base, on=["country", "forecast_origin", "target_quarter", "horizon"], suffixes=("_graph", "_base"), validate="one_to_one")
            graph_loss = _origin_mean(joined.rename(columns={"forecast_origin": "forecast_origin"}), np.abs(joined.actual_graph - joined.mean_graph))
            base_loss = _origin_mean(joined.rename(columns={"forecast_origin": "forecast_origin"}), np.abs(joined.actual_base - joined.mean_base))
            difference = (graph_loss - base_loss).dropna().to_numpy(float)
            statistic, pvalue = _dm_hln(difference, int(provenance["horizon"]))
            ci_low, ci_high = _moving_block_ci(difference, int(bootstrap["block_length"]), int(bootstrap["draws"]), np.random.default_rng(20260718))
            rows.append({**provenance, "comparator": comparator, "loss": "absolute_error", "dm_stat": statistic, "p_value": pvalue, "p_value_bh": float("nan"), "loss_difference": float(np.mean(difference)), "ci_low": ci_low, "ci_high": ci_high, "origin_count": len(difference)})
    result = pd.DataFrame(rows, columns=DM_COLUMNS)
    if not result.empty:
        result["p_value_bh"] = _bh_monotone(result.p_value)
    return result
