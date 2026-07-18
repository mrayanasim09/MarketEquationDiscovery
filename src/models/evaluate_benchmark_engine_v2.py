"""Post-run seed-aware scoring and inference for benchmark engine v2.

This module is deliberately not called by the runner.  It requires the complete
locked 20-seed forecast store and must be invoked only after post-run validation.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
from scipy import stats

from src.models.storage import DM_COLUMNS, METRIC_COLUMNS, PROVENANCE_COLUMNS, append_parquet

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "experiments/results/v2"
EVALUATOR_OUTPUTS = ("metrics.parquet", "dm_tests.parquet")


def _origin_losses(frame: pd.DataFrame, loss: str) -> pd.Series:
    error = frame.actual - frame.prediction
    values = error.abs() if loss == "absolute" else error.pow(2)
    return values.groupby(frame.forecast_origin).mean().sort_index()


def _block_bootstrap(values: pd.Series, rng: np.random.Generator, draws: int = 2_000, block: int = 4) -> tuple[float, float]:
    """Moving-block interval on origin-level losses; countries remain jointly grouped."""
    data = values.to_numpy(dtype=float)
    n = len(data)
    if n < 2:
        return (float("nan"), float("nan"))
    block = min(block, n)
    estimates = []
    for _ in range(draws):
        selected: list[float] = []
        while len(selected) < n:
            start = int(rng.integers(0, n - block + 1))
            selected.extend(data[start:start + block])
        estimates.append(float(np.mean(selected[:n])))
    interval = np.quantile(estimates, [0.025, 0.975])
    return float(interval[0]), float(interval[1])


def _dm_hln(diff: np.ndarray, horizon: int) -> tuple[float, float]:
    """HAC Diebold--Mariano with Harvey--Leybourne--Newbold small-sample correction."""
    n = len(diff)
    if n < 3:
        return float("nan"), float("nan")
    centered = diff - diff.mean()
    lag = min(horizon - 1, n - 1)
    long_run = float(np.mean(centered * centered))
    for term in range(1, lag + 1):
        covariance = float(np.mean(centered[term:] * centered[:-term]))
        long_run += 2 * (1 - term / (lag + 1)) * covariance
    if long_run <= 0:
        return float("nan"), float("nan")
    statistic = diff.mean() / math.sqrt(long_run / n)
    factor = math.sqrt((n + 1 - 2 * horizon + horizon * (horizon - 1) / n) / n)
    statistic *= factor
    return float(statistic), float(2 * stats.t.sf(abs(statistic), df=n - 1))


def _bh_adjust(pvalues: pd.Series) -> pd.Series:
    valid = pvalues.notna()
    usable = cast(pd.Series, pvalues.loc[valid])
    rank = usable.rank(method="first")
    adjusted = (usable * int(valid.sum()) / rank).clip(upper=1.0)
    return adjusted.reindex(pvalues.index)


def score_forecasts(forecasts: pd.DataFrame) -> pd.DataFrame:
    rows = []
    key = PROVENANCE_COLUMNS + ["model_name", "model_variant", "graph_variant", "seed", "horizon", "split", "feature_set", "graph_type"]
    for values, group in forecasts.groupby(key, dropna=False):
        run_id, git_commit, dataset_version, configuration_id, execution_timestamp, model, variant, graph_variant, seed, horizon, split, feature_set, graph_type = cast(tuple[object, object, object, object, object, object, object, object, object, object, object, object, object], values)
        origin_abs = _origin_losses(group, "absolute")
        origin_sq = _origin_losses(group, "squared")
        # The persistence forecast is used as the available scale for MASE.
        persistence_rows = cast(pd.DataFrame, forecasts[(forecasts.model_name == "persistence") & (forecasts.seed == "deterministic") & (forecasts.run_id == run_id)])
        persistence = group.merge(
            persistence_rows[["horizon", "forecast_origin", "country", "prediction"]],
            on=["horizon", "forecast_origin", "country"], suffixes=("", "_persistence"), how="left",
        )
        scale = (persistence.actual - persistence.prediction_persistence).abs().mean()
        mase = float(origin_abs.mean() / scale) if scale > 0 else float("nan")
        # Directional accuracy is assessed relative to the observed persistence value.
        direction_base = persistence.prediction_persistence
        directional = (np.sign(persistence.prediction - direction_base) == np.sign(persistence.actual - direction_base)).mean()
        lo, hi = _block_bootstrap(origin_abs, np.random.default_rng(20260718))
        common = {"run_id": run_id, "git_commit": git_commit, "dataset_version": dataset_version, "configuration_id": configuration_id, "execution_timestamp": execution_timestamp, "model_name": model, "model_variant": variant, "graph_variant": graph_variant, "seed": seed, "horizon": horizon, "split": split, "origin_count": len(origin_abs), "feature_set": feature_set, "graph_type": graph_type}
        rows.extend([
            {**common, "metric": "mae", "value": float(origin_abs.mean())},
            {**common, "metric": "rmse", "value": float(np.sqrt(origin_sq.mean()))},
            {**common, "metric": "mase", "value": mase},
            {**common, "metric": "directional_accuracy", "value": float(directional)},
            {**common, "metric": "mae_block_bootstrap_ci_low", "value": lo},
            {**common, "metric": "mae_block_bootstrap_ci_high", "value": hi},
        ])
    return cast(pd.DataFrame, pd.DataFrame(rows, columns=METRIC_COLUMNS))


def dm_tests(forecasts: pd.DataFrame, comparator: str = "ridge") -> pd.DataFrame:
    rows = []
    candidates = forecasts[forecasts.model_name.isin(["gcn", "temporal_graph"])]
    for values, group in candidates.groupby(PROVENANCE_COLUMNS + ["model_name", "model_variant", "seed", "horizon"]):
        run_id, git_commit, dataset_version, configuration_id, execution_timestamp, model, variant, seed, horizon = cast(tuple[object, object, object, object, object, object, object, object, object], values)
        base = forecasts[(forecasts.model_name == comparator) & (forecasts.horizon == horizon) & (forecasts.run_id == run_id)]
        if base.empty:
            continue
        # A seeded graph model is compared with the deterministic comparator at identical origins/countries.
        joined = group.merge(base, on=["horizon", "forecast_origin", "country"], suffixes=("_model", "_base"))
        loss_model = (joined.actual_model - joined.prediction_model).abs().groupby(joined.forecast_origin).mean()
        loss_base = (joined.actual_base - joined.prediction_base).abs().groupby(joined.forecast_origin).mean()
        difference = (loss_model - loss_base).dropna()
        statistic, pvalue = _dm_hln(difference.to_numpy(), int(cast(int | str, horizon)))
        rows.append({"run_id": run_id, "git_commit": git_commit, "dataset_version": dataset_version, "configuration_id": configuration_id, "execution_timestamp": execution_timestamp, "model_name": model, "model_variant": variant, "graph_variant": variant, "seed": seed, "horizon": horizon, "comparator_name": comparator, "comparator_variant": "macro_trade_exposure", "dm_stat": statistic, "p_value": pvalue, "origins": len(difference), "loss_difference": float(difference.mean())})
    result = cast(pd.DataFrame, pd.DataFrame(rows, columns=DM_COLUMNS))
    if not result.empty:
        result["p_value_bh"] = _bh_adjust(result.p_value)
    return result


def seed_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    """Aggregate seed-level estimates; deterministic models remain explicitly singleton rows."""
    base = metrics[metrics.metric.isin(["mae", "rmse", "mase", "directional_accuracy"])]
    grouped = base.groupby(["run_id", "model_name", "model_variant", "horizon", "metric"], dropna=False).value
    summary = grouped.agg(["mean", "std", "count"]).reset_index()
    summary["ci95_low"] = summary["mean"] - 1.96 * summary["std"].fillna(0) / np.sqrt(summary["count"])
    summary["ci95_high"] = summary["mean"] + 1.96 * summary["std"].fillna(0) / np.sqrt(summary["count"])
    return summary


def main() -> int:
    forecast_path = RESULTS / "forecasts.parquet"
    if not forecast_path.is_file():
        raise SystemExit("Refusing to score: forecasts.parquet does not exist; evaluator never trains models.")
    forecasts = pd.read_parquet(forecast_path)
    metrics = score_forecasts(forecasts)
    append_parquet("metrics.parquet", metrics, METRIC_COLUMNS, ["run_id", "model_name", "model_variant", "seed", "horizon", "split", "metric"])
    dm = dm_tests(forecasts)
    if not dm.empty:
        # p_value_bh is retained in the analysis companion table; canonical DM rows preserve the locked schema.
        append_parquet("dm_tests.parquet", cast(pd.DataFrame, dm.loc[:, DM_COLUMNS]), DM_COLUMNS, ["run_id", "model_name", "model_variant", "seed", "horizon", "comparator_name"])
        dm.to_parquet(RESULTS / "dm_tests_adjusted.parquet", index=False)
    seed_summary(metrics).to_parquet(RESULTS / "metrics_seed_summary.parquet", index=False)
    print("V2 post-run scoring completed with seed-level metrics and HAC/HLN DM tests.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
