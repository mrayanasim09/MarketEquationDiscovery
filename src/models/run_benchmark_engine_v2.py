"""Submission-grade, append-only rolling-origin benchmark engine for v2.

This module is intentionally separate from ``run_benchmarks.py``, which remains an
archived exploratory runner.  It never changes raw or processed data.
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import cast


import numpy as np
import pandas as pd
import torch
from statsmodels.tsa.api import VAR
from statsmodels.tsa.arima.model import ARIMA
from torch import nn

from src.models.baselines import BayesianShrinkageVAR, gradient_boosting_regressor
from src.models.features import feature_sequence, volatility
from src.models.graphs.factory import GRAPH_VARIANTS as GRAPH_FACTORY_VARIANTS, build as build_graph
from src.models.neural import (
    GraphConvolutionForecaster,
    SequenceLSTM,
    TemporalConvNet,
    TemporalGraphForecaster,
)
from src.models.provenance import RUN_MANIFEST_REQUIRED_FIELDS, build_execution_provenance
from src.models.storage import FORECAST_COLUMNS, append_parquet, write_run_manifest
from src.models.training import eligible_training, quarterly_step_count, seed_everything, training_metadata
from src.transform.common import V2_PROCESSED, require_validated_raw

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "experiments/results/v2"
CONFIG_PATH = RESULTS / "configs/benchmark_engine.json"
NEURAL_MODELS = {"mlp", "lstm", "tcn", "gcn", "temporal_graph"}
GRAPH_VARIANTS = GRAPH_FACTORY_VARIANTS
RUNNER_OUTPUTS = ("run_manifest.json", "forecasts.parquet")


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def quarter(value: object) -> pd.Period:
    return cast(pd.Period, pd.Period(str(value), freq="Q"))


def standardize(train: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = train.mean(axis=0)
    scale = train.std(axis=0)
    scale = np.where(scale > 0, scale, 1.0)
    return (train - mean) / scale, (values - mean) / scale, np.stack([mean, scale])


def ridge_predict(x: np.ndarray, y: np.ndarray, z: np.ndarray, penalty: float = 1.0) -> np.ndarray:
    x_scaled, z_scaled, _ = standardize(x, z)
    coef = np.linalg.solve(x_scaled.T @ x_scaled + penalty * np.eye(x.shape[1]), x_scaled.T @ y)
    return z_scaled @ coef


def dependency_versions() -> dict[str, str]:
    packages = ("numpy", "pandas", "scipy", "statsmodels", "scikit-learn", "pyarrow", "torch")
    values: dict[str, str] = {}
    for package in packages:
        try:
            values[package] = version(package)
        except PackageNotFoundError:
            values[package] = "not-installed"
    return values


def split_period(samples: pd.DataFrame, split: str) -> dict[str, str]:
    subset = samples[samples["split"].eq(split)]
    if subset.empty:
        raise ValueError(f"locked split has no {split} origins")
    return {
        "origin_start": str(subset["origin_quarter"].min()),
        "origin_end": str(subset["origin_quarter"].max()),
        "target_start": str(subset["target_quarter"].min()),
        "target_end": str(subset["target_quarter"].max()),
    }


def build_run_manifest_payload(cfg: dict, samples: pd.DataFrame, provenance: dict[str, str]) -> dict:
    payload = {
        **provenance,
        "engine_version": cfg["engine_version"],
        "hardware_environment": {
            "machine": platform.machine(),
            "processor": platform.processor(),
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        },
        "python_version": sys.version,
        "dependency_versions": dependency_versions(),
        "models": cfg["models"],
        "graph_variants": list(GRAPH_VARIANTS),
        "seeds": cfg["seeds"],
        "epochs": cfg["epochs"],
        "forecast_horizon": cfg["horizons"],
        "train_period": split_period(samples, "train"),
        "validation_period": split_period(samples, "validation"),
        "test_period": split_period(samples, "test"),
        "sequence_length": cfg["sequence_length"],
        "learning_rate": cfg["learning_rate"],
    }
    missing = [field for field in RUN_MANIFEST_REQUIRED_FIELDS if field not in payload]
    if missing:
        raise RuntimeError(f"execution manifest contract is incomplete: {missing}")
    return payload


def load_inputs() -> tuple[dict, pd.DataFrame, pd.DataFrame, list[str], dict[str, int], np.ndarray]:
    cfg = json.loads(CONFIG_PATH.read_text())
    samples = pd.read_csv(V2_PROCESSED / "forecast_samples.csv")
    panel = pd.read_csv(V2_PROCESSED / "quarterly_feature_panel.csv")
    for col in ["origin_quarter", "target_quarter", "macro_feature_quarter", "trade_graph_quarter"]:
        samples[col] = samples[col].map(quarter)
    panel["period"] = panel["quarter"].map(quarter)
    countries = json.loads((V2_PROCESSED / "countries.json").read_text())
    qidx = {q: i for i, q in enumerate(json.loads((V2_PROCESSED / "quarters.json").read_text()))}
    adjacency = np.load(V2_PROCESSED / "adjacency_directed_trade_eur.npy")
    return cfg, samples, panel, countries, qidx, adjacency


def tabular_features(rows: pd.DataFrame, panel: pd.DataFrame, countries: list[str], qidx: dict[str, int], adjacency: np.ndarray) -> np.ndarray:
    """Features available at each sample's persisted macro/trade input quarter only."""
    values = []
    for row in rows.to_dict("records"):
        country = str(row["country"])
        macro_quarter = cast(pd.Period, row["macro_feature_quarter"])
        history = feature_sequence(panel, country, macro_quarter, 4)
        graph = adjacency[qidx[str(row["trade_graph_quarter"])]]
        node = countries.index(country)
        values.append([
            history[-1, 0], history[-1, 1], volatility(panel, country, macro_quarter, 4),
            float(graph[node].sum()), float(graph[:, node].sum()),
        ])
    return np.asarray(values, dtype=float)


def sequence_eligible_rows(rows: pd.DataFrame, panel: pd.DataFrame) -> pd.DataFrame:
    """Exclude labels lacking a published four-quarter input history; never fill it."""
    keep = []
    for index, row in rows.iterrows():
        try:
            feature_sequence(panel, row.country, row.macro_feature_quarter, 4)
            keep.append(index)
        except ValueError:
            continue
    result = rows.loc[keep].copy()
    if result.empty:
        raise ValueError("no training rows have a complete permitted four-quarter feature history")
    return result


def sequence_features(rows: pd.DataFrame, panel: pd.DataFrame) -> np.ndarray:
    return np.stack([
        feature_sequence(panel, str(row["country"]), cast(pd.Period, row["macro_feature_quarter"]), 4)
        for row in rows.to_dict("records")
    ])


def history_matrix(panel: pd.DataFrame, countries: list[str], end: pd.Period) -> np.ndarray:
    subset = panel[panel.period <= end].pivot(index="period", columns="entity_id", values="cpi_yoy").reindex(columns=countries).dropna()
    return subset.to_numpy(dtype=float)


def graph_panels(rows: pd.DataFrame, panel: pd.DataFrame, countries: list[str], qidx: dict[str, int], adjacency: np.ndarray, variant: str, seed: int) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """One complete node panel per historical origin, using only its persisted lagged inputs."""
    output = []
    for origin, group in rows.groupby("origin_quarter", sort=True):
        group = group.set_index("country").reindex(countries)
        if group.target_cpi_yoy.isna().any():
            raise ValueError(f"incomplete country panel at training origin {origin}")
        graph_quarter = group.trade_graph_quarter.iloc[0]
        # Static GCN uses the persisted contemporaneously permitted node inputs.
        x = group[["cpi_yoy_input", "energy_cpi_yoy_input"]].to_numpy(dtype=float)
        a = build_graph(adjacency[qidx[str(graph_quarter)]], variant, np.random.default_rng(seed + qidx[str(graph_quarter)]))
        output.append((x, a, group.target_cpi_yoy.to_numpy(dtype=float)))
    return output


def temporal_graph_panels(rows: pd.DataFrame, panel: pd.DataFrame, countries: list[str], qidx: dict[str, int], adjacency: np.ndarray, variant: str, seed: int) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Build G(t-4)..G(t-1) and matching X(t-4)..X(t-1), never later snapshots."""
    output = []
    for origin, group in rows.groupby("origin_quarter", sort=True):
        group = group.set_index("country").reindex(countries)
        end = group.macro_feature_quarter.iloc[0]
        graph_end = group.trade_graph_quarter.iloc[0]
        periods = pd.period_range(end=end, periods=4, freq="Q")
        graph_periods = pd.period_range(end=graph_end, periods=4, freq="Q")
        if any(str(item) not in qidx for item in graph_periods):
            raise ValueError(f"missing graph history before {graph_end}")
        x = np.stack([np.stack([feature_sequence(panel, country, item, 4)[-1] for country in countries]) for item in periods])
        a = np.stack([build_graph(adjacency[qidx[str(item)]], variant, np.random.default_rng(seed + qidx[str(item)])) for item in graph_periods])
        output.append((x, a, group.target_cpi_yoy.to_numpy(dtype=float)))
    return output


def fit_neural(model: nn.Module, x: np.ndarray, y: np.ndarray, epochs: int, learning_rate: float) -> nn.Module:
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    tx, ty = torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)
    for _ in range(epochs):
        optimizer.zero_grad()
        loss = torch.mean((model(tx) - ty) ** 2)
        loss.backward()
        optimizer.step()
    return model.eval()


def fit_graph_neural(model: nn.Module, panels: list[tuple[np.ndarray, np.ndarray, np.ndarray]], epochs: int, learning_rate: float, temporal: bool) -> nn.Module:
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    for _ in range(epochs):
        optimizer.zero_grad()
        losses = []
        for x, a, y in panels:
            tx, ta, ty = (torch.tensor(item, dtype=torch.float32) for item in (x, a, y))
            prediction = model(tx, ta) if temporal else model(tx, ta)
            losses.append(torch.mean((prediction - ty) ** 2))
        torch.stack(losses).mean().backward()
        optimizer.step()
    return model.eval()


def forecast_rows(provenance: dict[str, str], name: str, variant: str, seed: int | str, horizon: int, test: pd.DataFrame, predictions: np.ndarray, train: pd.DataFrame, feature_set: str, graph_type: str) -> pd.DataFrame:
    meta = training_metadata(train)
    frame = pd.DataFrame({
        **provenance,
        "model_name": name, "model_variant": variant, "graph_variant": graph_type, "seed": str(seed), "horizon": horizon,
        "forecast_origin": test.origin_quarter.astype(str), "country": test.country,
        "target_quarter": test.target_quarter.astype(str), "prediction": np.asarray(predictions, dtype=float),
        "actual": test.target_cpi_yoy.to_numpy(dtype=float), "split": test.split,
        "training_sample_count": meta["training_sample_count"], "earliest_training_quarter": meta["earliest_training_quarter"],
        "latest_training_quarter": meta["latest_training_quarter"], "feature_set": feature_set, "graph_type": graph_type,
    })
    return cast(pd.DataFrame, frame.loc[:, FORECAST_COLUMNS].copy())


def checkpoint(model: nn.Module, run_id: str) -> None:
    target = RESULTS / "checkpoints" / f"{run_id}.pt"
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), target)


def run_origin(cfg: dict, samples: pd.DataFrame, panel: pd.DataFrame, countries: list[str], qidx: dict[str, int], adjacency: np.ndarray, horizon: int, origin: pd.Period, selected_models: set[str], provenance: dict[str, str]) -> None:
    test = cast(pd.DataFrame, samples[(samples.horizon_quarters == horizon) & (samples.origin_quarter == origin) & (samples.split == "test")]).sort_values("country")
    train = eligible_training(samples, horizon, origin).sort_values(["origin_quarter", "country"])
    # Four-quarter feature models exclude only early labels without four published input quarters.
    train_features = sequence_eligible_rows(train, panel).sort_values(["origin_quarter", "country"])
    y = train_features.target_cpi_yoy.to_numpy(dtype=float)
    x_train = tabular_features(train_features, panel, countries, qidx, adjacency)
    x_test = tabular_features(test, panel, countries, qidx, adjacency)
    outputs: list[pd.DataFrame] = []

    if "persistence" in selected_models:
        outputs.append(forecast_rows(provenance, "persistence", "last_cpi_yoy", "deterministic", horizon, test, test.cpi_yoy_input, train, "cpi_only", "none"))
    if "ridge" in selected_models:
        outputs.append(forecast_rows(provenance, "ridge", "macro_trade_exposure", "deterministic", horizon, test, ridge_predict(x_train, y, x_test), train_features, "cpi_energy_volatility_trade_exposure", "none"))
    if "gradient_boosting" in selected_models:
        model = gradient_boosting_regressor().fit(x_train, y)
        outputs.append(forecast_rows(provenance, "gradient_boosting", "hist_gradient_boosting", "deterministic", horizon, test, model.predict(x_test), train_features, "cpi_energy_volatility_trade_exposure", "none"))

    end = test.macro_feature_quarter.iloc[0]
    matrix = history_matrix(panel, countries, end)
    steps = quarterly_step_count(end, test.target_quarter.iloc[0])
    if "var" in selected_models or "bayesian_shrinkage_var" in selected_models:
        try:
            ordinary_var = VAR(matrix).fit(maxlags=1, trend="c").forecast(matrix[-1:], steps)[-1]
        except Exception:
            ordinary_var = test.cpi_yoy_input.to_numpy(dtype=float)
        if "var" in selected_models:
            outputs.append(forecast_rows(provenance, "var", "panel_var_lag1", "deterministic", horizon, test, ordinary_var, train, "cpi_history", "none"))
        if "bayesian_shrinkage_var" in selected_models:
            try:
                bvar = BayesianShrinkageVAR(lags=1, prior_precision=1.0).fit(matrix).forecast(steps)[-1]
            except Exception:
                bvar = test.cpi_yoy_input.to_numpy(dtype=float)
            outputs.append(forecast_rows(provenance, "bayesian_shrinkage_var", "ridge_prior_lag1", "deterministic", horizon, test, bvar, train, "cpi_history", "none"))
    if "arima" in selected_models:
        arima_predictions: list[float] = []
        for row in test.to_dict("records"):
            country = str(row["country"])
            macro_quarter = cast(pd.Period, row["macro_feature_quarter"])
            history = cast(pd.DataFrame, panel[(panel.entity_id == country) & (panel.period <= macro_quarter)])
            values = history.sort_values("period").cpi_yoy.dropna().to_numpy(float)
            try:
                arima_predictions.append(float(ARIMA(values, order=(1, 0, 0)).fit().forecast(steps)[-1]))
            except Exception:
                arima_predictions.append(float(row["cpi_yoy_input"]))
        outputs.append(forecast_rows(provenance, "arima", "country_arima_1_0_0", "deterministic", horizon, test, np.asarray(arima_predictions), train, "cpi_history", "none"))

    seq_train, seq_test = sequence_features(train_features, panel), sequence_features(test, panel)
    for seed in cfg["seeds"]:
        seed_everything(seed)
        # All neural preprocessing is fit only on this outer origin's eligible training observations.
        if "mlp" in selected_models:
            x_scaled, z_scaled, _ = standardize(x_train, x_test)
            mlp = fit_neural(MLP(x_train.shape[1], cfg["hidden_dim"]), x_scaled, y, cfg["epochs"], cfg["learning_rate"])
            pred = mlp(torch.tensor(z_scaled, dtype=torch.float32)).detach().numpy()
            checkpoint_id = f"mlp_h{horizon}_{origin}_s{seed}"
            checkpoint(mlp, f"{provenance['run_id']}__{checkpoint_id}")
            outputs.append(forecast_rows(provenance, "mlp", "macro_trade_exposure", seed, horizon, test, pred, train_features, "cpi_energy_volatility_trade_exposure", "none"))
        for name, cls in (("lstm", SequenceLSTM), ("tcn", TemporalConvNet)):
            if name not in selected_models:
                continue
            flat_train, flat_test = seq_train.reshape(-1, 2), seq_test.reshape(-1, 2)
            scaled_train, scaled_test, _ = standardize(flat_train, flat_test)
            net = fit_neural(cls(2, cfg["hidden_dim"]), scaled_train.reshape(seq_train.shape), y, cfg["epochs"], cfg["learning_rate"])
            pred = net(torch.tensor(scaled_test.reshape(seq_test.shape), dtype=torch.float32)).detach().numpy()
            checkpoint_id = f"{name}_h{horizon}_{origin}_s{seed}"
            checkpoint(net, f"{provenance['run_id']}__{checkpoint_id}")
            outputs.append(forecast_rows(provenance, name, "cpi_energy_history_k4", seed, horizon, test, pred, train_features, "cpi_energy_sequence", "none"))
        for variant in GRAPH_VARIANTS:
            for name, temporal in (("gcn", False), ("temporal_graph", True)):
                if name not in selected_models:
                    continue
                graph_train = train_features if temporal else train
                panels = temporal_graph_panels(graph_train, panel, countries, qidx, adjacency, variant, seed) if temporal else graph_panels(graph_train, panel, countries, qidx, adjacency, variant, seed)
                raw = np.concatenate([item[0].reshape(-1, 2) for item in panels])
                scaled, _, scaler = standardize(raw, raw)
                pos = 0
                scaled_panels = []
                for x, a, target in panels:
                    count = x.shape[0] * (x.shape[1] if temporal else 1)
                    shape = x.shape
                    scaled_panels.append((scaled[pos:pos + count].reshape(shape), a, target))
                    pos += count
                test_panels = temporal_graph_panels(test, panel, countries, qidx, adjacency, variant, seed) if temporal else graph_panels(test, panel, countries, qidx, adjacency, variant, seed)
                test_x, test_a, _ = test_panels[0]
                test_scaled = (test_x - scaler[0]) / scaler[1]
                network = TemporalGraphForecaster(2, cfg["hidden_dim"]) if temporal else GraphConvolutionForecaster(2, cfg["hidden_dim"])
                network = fit_graph_neural(network, scaled_panels, cfg["epochs"], cfg["learning_rate"], temporal)
                pred = network(torch.tensor(test_scaled, dtype=torch.float32), torch.tensor(test_a, dtype=torch.float32)).detach().numpy()
                checkpoint_id = f"{name}_{variant}_h{horizon}_{origin}_s{seed}"
                checkpoint(network, f"{provenance['run_id']}__{checkpoint_id}")
                outputs.append(forecast_rows(provenance, name, variant, seed, horizon, test, pred, graph_train, "cpi_energy_sequence" if temporal else "cpi_energy_input", variant))

    for frame in outputs:
        append_parquet("forecasts.parquet", frame, FORECAST_COLUMNS, ["run_id", "model_name", "model_variant", "seed", "horizon", "forecast_origin", "country"])


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the isolated v2 benchmark engine.")
    parser.add_argument("--execute", action="store_true", help="Required safeguard before any model fitting occurs.")
    parser.add_argument("--models", nargs="*", help="Optional subset of locked model names for a documented dry-run.")
    args = parser.parse_args()
    if not args.execute:
        raise SystemExit("Refusing to train: pass --execute only after the benchmark-engine validation gate passes.")
    require_validated_raw()
    cfg, samples, panel, countries, qidx, adjacency = load_inputs()
    if tuple(cfg["graph_variants"]) != tuple(GRAPH_VARIANTS):
        raise RuntimeError("locked configuration graph_variants do not match the graph factory registry")
    provenance = build_execution_provenance(CONFIG_PATH)
    write_run_manifest(build_run_manifest_payload(cfg, samples, provenance))
    selected = set(args.models or cfg["models"])
    unknown = selected - set(cfg["models"])
    if unknown:
        raise ValueError(f"unknown model(s): {sorted(unknown)}")
    RESULTS.mkdir(parents=True, exist_ok=True)
    for horizon in cfg["horizons"]:
        origins = sorted(samples[(samples.horizon_quarters == horizon) & (samples.split == "test")].origin_quarter.unique())
        for origin in origins:
            run_origin(cfg, samples, panel, countries, qidx, adjacency, horizon, origin, selected, provenance)
    print("V2 benchmark engine run completed; calculate results only with the post-run evaluation module.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
