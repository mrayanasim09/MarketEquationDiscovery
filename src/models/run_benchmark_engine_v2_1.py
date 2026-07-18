"""Isolated, transactional v2.1 benchmark engine.

No code in this module writes raw/processed artifacts or archived v2 results.  It
requires a completed immutable validation-only tuning manifest before it permits
any final-test model fitting.
"""
from __future__ import annotations

import argparse
import json
import platform
import shutil
import sys
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from scipy import stats
from statsmodels.tsa.api import VAR
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.exponential_smoothing.ets import ETSModel
from statsmodels.tsa.statespace.dynamic_factor import DynamicFactor

from src.models.baselines import gradient_boosting_regressor
from src.models.evaluate_benchmark_engine_v2_1 import dm_tests, score_forecasts
from src.models.graphs.factory import GRAPH_VARIANTS, build as build_graph
from src.models.neural import GraphConvolutionForecaster, SequenceLSTM, TemporalConvNet, TemporalGraphForecaster
from src.models.provenance import build_execution_provenance, sha256_file
from src.models.run_benchmark_engine_v2 import (
    MLP, fit_graph_neural, fit_neural, graph_panels, history_matrix, quarter,
    sequence_eligible_rows, sequence_features, standardize, tabular_features,
)
from src.models.storage_v2_1 import DM_COLUMNS, FORECAST_COLUMNS, METRIC_COLUMNS, RESULTS, append_failure_record, write_parquet_exact
from src.models.training import eligible_training, quarterly_step_count, seed_everything
from src.models.tuning.manifest import TUNING_MANIFEST, require_tuning_manifest
from src.models.validate_v2_1_contract import REQUIRED_MODELS, load_config, validation_report
from src.transform.common import V2_PROCESSED, require_validated_raw

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = RESULTS / "configs/benchmark_engine_v2_1.json"
RUNNER_OUTPUTS = ("forecasts.parquet", "metrics.parquet", "dm_tests.parquet", "run_manifest.json", "checkpoints", "environment_manifest.json")
SUPPORTED_MODELS = frozenset(REQUIRED_MODELS)


def _load_inputs() -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, list[str], dict[str, int], np.ndarray]:
    cfg = load_config()
    samples = pd.read_csv(V2_PROCESSED / "forecast_samples.csv")
    panel = pd.read_csv(V2_PROCESSED / "quarterly_feature_panel.csv")
    for column in ("origin_quarter", "target_quarter", "macro_feature_quarter", "trade_graph_quarter"):
        samples[column] = samples[column].map(quarter)
    panel["period"] = panel["quarter"].map(quarter)
    countries = json.loads((V2_PROCESSED / "countries.json").read_text())
    quarters = json.loads((V2_PROCESSED / "quarters.json").read_text())
    return cfg, samples, panel, countries, {name: index for index, name in enumerate(quarters)}, np.load(V2_PROCESSED / "adjacency_directed_trade_eur.npy")


def _normal_intervals(mean: np.ndarray, scale: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not np.isfinite(scale) or scale <= 0:
        raise RuntimeError("validation-calibrated predictive scale must be positive and finite")
    z80, z95 = stats.norm.ppf(0.9), stats.norm.ppf(0.975)
    return mean - z80 * scale, mean + z80 * scale, mean - z95 * scale, mean + z95 * scale


def _classical_predictions(name: str, test: pd.DataFrame, panel: pd.DataFrame, countries: list[str], horizon: int) -> np.ndarray:
    end = test.macro_feature_quarter.iloc[0]
    steps = quarterly_step_count(end, test.target_quarter.iloc[0])
    matrix = history_matrix(panel, countries, end)
    if name == "persistence":
        return test.cpi_yoy_input.to_numpy(float)
    if name == "var":
        try:
            return VAR(matrix).fit(maxlags=1, trend="c").forecast(matrix[-1:], steps)[-1]
        except Exception:
            return test.cpi_yoy_input.to_numpy(float)
    if name == "dynamic_factor":
        try:
            fitted = DynamicFactor(matrix, k_factors=1, factor_order=1, error_order=0).fit(disp=False)
            return np.asarray(fitted.forecast(steps=steps).iloc[-1] if hasattr(fitted.forecast(steps=steps), "iloc") else fitted.forecast(steps=steps)[-1], dtype=float)
        except Exception:
            return test.cpi_yoy_input.to_numpy(float)
    output: list[float] = []
    for row in test.to_dict("records"):
        values = panel[(panel.entity_id == row["country"]) & (panel.period <= row["macro_feature_quarter"])].sort_values("period").cpi_yoy.dropna().to_numpy(float)
        try:
            if name == "arima":
                prediction = ARIMA(values, order=(1, 0, 0)).fit().forecast(steps)[-1]
            elif name == "ets":
                prediction = ETSModel(values, error="add", trend=None, seasonal=None).fit(disp=False).forecast(steps)[-1]
            else:
                raise ValueError(f"unsupported classical model {name}")
        except Exception:
            prediction = row["cpi_yoy_input"]
        output.append(float(prediction))
    return np.asarray(output)


def _checkpoint(model: torch.nn.Module, directory: Path, name: str) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), directory / f"{name}.pt")


def _neural_predictions(name: str, variant: str, seed: int, train: pd.DataFrame, test: pd.DataFrame, panel: pd.DataFrame, countries: list[str], qidx: dict[str, int], adjacency: np.ndarray, cfg: dict[str, Any], checkpoint_dir: Path | None, checkpoint_id: str) -> np.ndarray:
    seed_everything(seed)
    train_features = sequence_eligible_rows(train, panel).sort_values(["origin_quarter", "country"])
    y = train_features.target_cpi_yoy.to_numpy(float)
    if name == "mlp":
        x_train = tabular_features(train_features, panel, countries, qidx, adjacency)
        x_test = tabular_features(test, panel, countries, qidx, adjacency)
        scaled_train, scaled_test, _ = standardize(x_train, x_test)
        model = fit_neural(MLP(x_train.shape[1], cfg["hidden_dim"]), scaled_train, y, cfg["epochs"], cfg["learning_rate"])
        prediction = model(torch.tensor(scaled_test, dtype=torch.float32)).detach().numpy()
    elif name in {"lstm", "tcn"}:
        train_seq, test_seq = sequence_features(train_features, panel), sequence_features(test, panel)
        flat_train, flat_test = train_seq.reshape(-1, 2), test_seq.reshape(-1, 2)
        scaled_train, scaled_test, _ = standardize(flat_train, flat_test)
        model_type = SequenceLSTM if name == "lstm" else TemporalConvNet
        model = fit_neural(model_type(2, cfg["hidden_dim"]), scaled_train.reshape(train_seq.shape), y, cfg["epochs"], cfg["learning_rate"])
        prediction = model(torch.tensor(scaled_test.reshape(test_seq.shape), dtype=torch.float32)).detach().numpy()
    else:
        temporal = name == "temporal_graph"
        graph_train = train_features if temporal else train
        panels = graph_panels(graph_train, panel, countries, qidx, adjacency, variant, seed) if not temporal else _temporal_graph_panels(graph_train, panel, countries, qidx, adjacency, variant, seed)
        raw = np.concatenate([item[0].reshape(-1, 2) for item in panels])
        scaled, _, scaler = standardize(raw, raw)
        scaled_panels, position = [], 0
        for x, graph, target in panels:
            count = x.shape[0] * (x.shape[1] if temporal else 1)
            scaled_panels.append((scaled[position:position + count].reshape(x.shape), graph, target))
            position += count
        test_panels = graph_panels(test, panel, countries, qidx, adjacency, variant, seed) if not temporal else _temporal_graph_panels(test, panel, countries, qidx, adjacency, variant, seed)
        test_x, test_graph, _ = test_panels[0]
        model = TemporalGraphForecaster(2, cfg["hidden_dim"]) if temporal else GraphConvolutionForecaster(2, cfg["hidden_dim"])
        model = fit_graph_neural(model, scaled_panels, cfg["epochs"], cfg["learning_rate"], temporal)
        prediction = model(torch.tensor((test_x - scaler[0]) / scaler[1], dtype=torch.float32), torch.tensor(test_graph, dtype=torch.float32)).detach().numpy()
    if checkpoint_dir is not None:
        _checkpoint(model, checkpoint_dir, checkpoint_id)
    return np.asarray(prediction, dtype=float)


def _temporal_graph_panels(rows: pd.DataFrame, panel: pd.DataFrame, countries: list[str], qidx: dict[str, int], adjacency: np.ndarray, variant: str, seed: int) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    result = []
    for _, group in rows.groupby("origin_quarter", sort=True):
        group = group.set_index("country").reindex(countries)
        periods = pd.period_range(end=group.macro_feature_quarter.iloc[0], periods=4, freq="Q")
        graph_periods = pd.period_range(end=group.trade_graph_quarter.iloc[0], periods=4, freq="Q")
        if any(str(item) not in qidx for item in graph_periods):
            raise ValueError("temporal graph history lacks a persisted snapshot")
        lookup = panel.set_index(["entity_id", "period"])[["cpi_yoy", "energy_cpi_yoy"]]
        keys = [(country, item) for item in periods for country in countries]
        if not set(keys).issubset(lookup.index) or lookup.loc[keys].isna().any().any():
            raise ValueError("temporal graph input lacks permitted node observations")
        x = lookup.loc[keys].to_numpy(float).reshape(len(periods), len(countries), 2)
        graph = np.stack([build_graph(adjacency[qidx[str(item)]], variant, np.random.default_rng(seed + qidx[str(item)])) for item in graph_periods])
        result.append((x, graph, group.target_cpi_yoy.to_numpy(float)))
    return result


def _provenance(cfg: dict[str, Any]) -> dict[str, str]:
    base = build_execution_provenance(CONFIG_PATH)
    base["configuration_id"] = f"{cfg['configuration_id']}:sha256:{sha256_file(CONFIG_PATH)}"
    return base


def _write_environment_manifest(transaction: Path) -> None:
    packages = ("numpy", "pandas", "scipy", "statsmodels", "scikit-learn", "pyarrow", "torch")
    versions = {package: version(package) for package in packages}
    payload = {"python_version": sys.version, "platform": platform.platform(), "machine": platform.machine(), "torch_cuda_available": torch.cuda.is_available(), "dependency_versions": versions}
    (transaction / "environment_manifest.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _write_manifest(run_id: str, cfg: dict[str, Any], tuning: dict[str, Any]) -> None:
    target = RESULTS / "run_manifest.json"
    existing = json.loads(target.read_text()) if target.exists() else {"engine_version": "2.1", "runs": []}
    if any(item.get("run_id") == run_id for item in existing["runs"]):
        raise RuntimeError("generated run_id already exists")
    existing["runs"].append({"run_id": run_id, "status": "started", "configuration_sha256": sha256_file(CONFIG_PATH), "tuning_manifest_sha256": sha256_file(TUNING_MANIFEST), "models": cfg["models"], "graph_variants": cfg["graph_variants"], "seeds": cfg["seeds"], "horizons": cfg["horizons"], "started_at": datetime.now(timezone.utc).isoformat(), "python_version": sys.version, "platform": platform.platform(), "tuning_selected_parameters": tuning["selected_parameters"]})
    target.write_text(json.dumps(existing, indent=2) + "\n")


def _assert_complete(frame: pd.DataFrame, cfg: dict[str, Any], samples: pd.DataFrame) -> None:
    test = samples[samples.split.eq("test")]
    deterministic = {"persistence", "arima", "var", "ets", "dynamic_factor", "ridge", "gradient_boosting"}
    expected = len(test) * len(deterministic) + len(test) * len(cfg["seeds"]) * (3 + 2 * len(cfg["graph_variants"]))
    if len(frame) != expected or set(frame.model) != set(cfg["models"]):
        raise RuntimeError("staged forecasts do not cover the exact locked v2.1 registry")
    if frame.duplicated(["run_id", "country", "forecast_origin", "horizon", "model", "graph_variant", "seed"]).any():
        raise RuntimeError("staged forecasts contain duplicate run-level keys")
    if not ((frame.lower_95 <= frame.lower_80) & (frame.lower_80 <= frame["mean"]) & (frame["mean"] <= frame.upper_80) & (frame.upper_80 <= frame.upper_95)).all():
        raise RuntimeError("staged predictive intervals are incoherent")


def main() -> int:
    parser = argparse.ArgumentParser(description="Execute the locked transactional v2.1 benchmark.")
    parser.add_argument("--execute", action="store_true", help="Required; no partial registry option exists.")
    args = parser.parse_args()
    if not args.execute:
        raise SystemExit("refusing to fit: pass --execute after tuning and v2.1 preflight validation")
    contract = validation_report()
    if not contract["passed"]:
        raise RuntimeError(f"v2.1 contract validation failed: {contract['errors']}")
    require_validated_raw()
    tuning = require_tuning_manifest()
    cfg, samples, panel, countries, qidx, adjacency = _load_inputs()
    if set(cfg["models"]) != REQUIRED_MODELS or tuple(cfg["graph_variants"]) != tuple(GRAPH_VARIANTS):
        raise RuntimeError("locked v2.1 model or graph registry differs from runtime registry")
    provenance = _provenance(cfg)
    transaction = RESULTS / "staging" / provenance["run_id"]
    if transaction.exists() or any((RESULTS / name).exists() for name in ("forecasts.parquet", "metrics.parquet", "dm_tests.parquet", "checkpoints")):
        raise RuntimeError("refusing to overwrite an existing v2.1 transaction or canonical output")
    transaction.mkdir(parents=True)
    _write_environment_manifest(transaction)
    _write_manifest(provenance["run_id"], cfg, tuning)
    rows: list[dict[str, Any]] = []
    try:
        for horizon in cfg["horizons"]:
            for origin in sorted(samples[(samples.horizon_quarters == horizon) & samples.split.eq("test")].origin_quarter.unique()):
                test = samples[(samples.horizon_quarters == horizon) & (samples.origin_quarter == origin) & samples.split.eq("test")].sort_values("country")
                train = eligible_training(samples, horizon, origin).sort_values(["origin_quarter", "country"])
                for model in ("persistence", "arima", "var", "ets", "dynamic_factor"):
                    point = _classical_predictions(model, test, panel, countries, horizon)
                    _append_rows(rows, provenance, model, "none", "deterministic", horizon, test, point, tuning)
                train_features = sequence_eligible_rows(train, panel).sort_values(["origin_quarter", "country"])
                x_train = tabular_features(train_features, panel, countries, qidx, adjacency)
                x_test = tabular_features(test, panel, countries, qidx, adjacency)
                ridge = np.linalg.solve(standardize(x_train, x_test)[0].T @ standardize(x_train, x_test)[0] + np.eye(x_train.shape[1]), standardize(x_train, x_test)[0].T @ train_features.target_cpi_yoy.to_numpy(float))
                _append_rows(rows, provenance, "ridge", "none", "deterministic", horizon, test, standardize(x_train, x_test)[1] @ ridge, tuning)
                boost = gradient_boosting_regressor().fit(x_train, train_features.target_cpi_yoy.to_numpy(float))
                _append_rows(rows, provenance, "gradient_boosting", "none", "deterministic", horizon, test, boost.predict(x_test), tuning)
                for seed in cfg["seeds"]:
                    for model in ("mlp", "lstm", "tcn"):
                        point = _neural_predictions(model, "none", seed, train, test, panel, countries, qidx, adjacency, cfg, transaction / "checkpoints", f"{model}_h{horizon}_{origin}_s{seed}")
                        _append_rows(rows, provenance, model, "none", str(seed), horizon, test, point, tuning)
                    for variant in cfg["graph_variants"]:
                        for model in ("gcn", "temporal_graph"):
                            point = _neural_predictions(model, variant, seed, train, test, panel, countries, qidx, adjacency, cfg, transaction / "checkpoints", f"{model}_{variant}_h{horizon}_{origin}_s{seed}")
                            _append_rows(rows, provenance, model, variant, str(seed), horizon, test, point, tuning)
        forecasts = pd.DataFrame(rows, columns=FORECAST_COLUMNS)
        _assert_complete(forecasts, cfg, samples)
        write_parquet_exact(transaction / "forecasts.parquet", forecasts, FORECAST_COLUMNS, ["run_id", "country", "forecast_origin", "horizon", "model", "graph_variant", "seed"])
        write_parquet_exact(transaction / "metrics.parquet", score_forecasts(forecasts), METRIC_COLUMNS, ["run_id", "model", "graph_variant", "seed", "horizon", "metric"])
        write_parquet_exact(transaction / "dm_tests.parquet", dm_tests(forecasts, cfg), DM_COLUMNS, ["run_id", "model", "graph_variant", "seed", "horizon", "comparator", "loss"])
        for name in ("forecasts.parquet", "metrics.parquet", "dm_tests.parquet", "checkpoints", "environment_manifest.json"):
            shutil.move(str(transaction / name), str(RESULTS / name))
        transaction.rmdir()
    except BaseException as exc:
        append_failure_record({"run_id": provenance["run_id"], "status": "failed", "transaction_directory": str(transaction.relative_to(ROOT)), "exception_type": type(exc).__name__, "exception_message": str(exc)})
        raise
    print(f"v2.1 benchmark transaction completed: {provenance['run_id']}")
    return 0


def _append_rows(rows: list[dict[str, Any]], provenance: dict[str, str], model: str, graph_variant: str, seed: str, horizon: int, test: pd.DataFrame, point: np.ndarray, tuning: dict[str, Any]) -> None:
    calibration = tuning["calibration"].get(model)
    if not isinstance(calibration, dict):
        raise RuntimeError(f"tuning manifest lacks validation calibration for {model}")
    lower80, upper80, lower95, upper95 = _normal_intervals(np.asarray(point, dtype=float), float(calibration["residual_scale"]))
    for row, mean, lo80, hi80, lo95, hi95 in zip(test.to_dict("records"), point, lower80, upper80, lower95, upper95, strict=True):
        rows.append({**provenance, "country": row["country"], "forecast_origin": str(row["origin_quarter"]), "target_quarter": str(row["target_quarter"]), "horizon": horizon, "model": model, "graph_variant": graph_variant, "seed": seed, "mean": float(mean), "lower_80": float(lo80), "upper_80": float(hi80), "lower_95": float(lo95), "upper_95": float(hi95), "actual": float(row["target_cpi_yoy"])})


if __name__ == "__main__":
    raise SystemExit(main())
