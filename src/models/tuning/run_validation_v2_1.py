"""Authorized validation-only tuning for the locked v2.1 registry.

This module never selects, loads, or predicts rows in the final-test split. It is
separate from the benchmark runner so that its immutable result can be inspected
and committed before final-test execution is permitted.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

from src.models.baselines import gradient_boosting_regressor
from src.models.provenance import build_execution_provenance, git_commit
from src.models.run_benchmark_engine_v2 import (
    sequence_eligible_rows,
    standardize,
    tabular_features,
)
from src.models.run_benchmark_engine_v2_1 import (
    _classical_predictions,
    _load_inputs,
    _neural_predictions,
)
from src.models.training import eligible_training
from src.models.tuning.candidates import candidate_registry
from src.models.tuning.manifest import CONFIG, TUNING_MANIFEST, write_tuning_manifest
from src.models.tuning.validation import normal_calibration, select_by_validation_mae
from src.models.validate_v2_1_contract import REQUIRED_MODELS, validation_report
from src.transform.common import require_validated_raw


class TestAccessError(RuntimeError):
    """Raised when a tuning path would touch final-test rows."""


def _validate_only(samples: pd.DataFrame) -> pd.DataFrame:
    validation = samples[samples.split.eq("validation")].copy()
    if validation.empty or not validation.split.eq("validation").all():
        raise TestAccessError("tuning input must contain only validation rows")
    if samples.loc[validation.index, "split"].eq("test").any():
        raise TestAccessError("final-test row detected in tuning input")
    return validation


def _ridge_predictions(train: pd.DataFrame, validation: pd.DataFrame, panel: pd.DataFrame, countries: list[str], qidx: dict[str, int], adjacency: np.ndarray) -> np.ndarray:
    train_features = sequence_eligible_rows(train, panel).sort_values(["origin_quarter", "country"])
    x_train = tabular_features(train_features, panel, countries, qidx, adjacency)
    x_validation = tabular_features(validation, panel, countries, qidx, adjacency)
    scaled_train, scaled_validation, _ = standardize(x_train, x_validation)
    y = train_features.target_cpi_yoy.to_numpy(float)
    coefficient = np.linalg.solve(scaled_train.T @ scaled_train + np.eye(scaled_train.shape[1]), scaled_train.T @ y)
    return scaled_validation @ coefficient


def _boosting_predictions(train: pd.DataFrame, validation: pd.DataFrame, panel: pd.DataFrame, countries: list[str], qidx: dict[str, int], adjacency: np.ndarray) -> np.ndarray:
    train_features = sequence_eligible_rows(train, panel).sort_values(["origin_quarter", "country"])
    x_train = tabular_features(train_features, panel, countries, qidx, adjacency)
    x_validation = tabular_features(validation, panel, countries, qidx, adjacency)
    return gradient_boosting_regressor().fit(x_train, train_features.target_cpi_yoy.to_numpy(float)).predict(x_validation)


def _record(rows: list[dict[str, Any]], residuals: list[dict[str, Any]], model: str, variant: str, seed: str, horizon: int, origin: object, validation: pd.DataFrame, prediction: np.ndarray, candidate_id: str, parameters: dict[str, Any]) -> None:
    actual = validation.target_cpi_yoy.to_numpy(float)
    rows.append({
        "model": model,
        "graph_variant": variant,
        "seed": seed,
        "horizon": horizon,
        "origin": str(origin),
        "candidate_id": candidate_id,
        "candidate_parameters": parameters,
        "mae": float(np.mean(np.abs(actual - prediction))),
        "rmse": float(np.sqrt(np.mean((actual - prediction) ** 2))),
    })
    residuals.extend({"model": model, "actual": float(y), "mean": float(mean)} for y, mean in zip(actual, prediction, strict=True))


def main() -> int:
    parser = argparse.ArgumentParser(description="Run validation-only tuning for v2.1; final-test access is prohibited.")
    parser.add_argument("--execute", action="store_true", help="Required acknowledgement before any validation model fitting.")
    args = parser.parse_args()
    if not args.execute:
        raise SystemExit("refusing to tune: pass --execute for the validation-only tuning phase")
    contract = validation_report()
    # A missing tuning manifest is expected before tuning; all other contract errors are fatal.
    unexpected = [error for error in contract["errors"] if "tuning isolation gate failed" not in error]
    if unexpected:
        raise RuntimeError(f"v2.1 contract is not executable: {unexpected}")
    require_validated_raw()
    if TUNING_MANIFEST.exists():
        raise FileExistsError("refusing to overwrite immutable v2.1 tuning manifest")
    cfg, samples, panel, countries, qidx, adjacency = _load_inputs()
    if set(cfg["models"]) != REQUIRED_MODELS:
        raise RuntimeError("configured model registry differs from the locked v2.1 registry")
    validation_rows = _validate_only(samples)
    candidates = candidate_registry(cfg)
    if set(candidates) != set(cfg["models"]):
        raise RuntimeError("candidate registry does not cover every configured model exactly")

    scores: list[dict[str, Any]] = []
    residuals: list[dict[str, Any]] = []
    for horizon in cfg["horizons"]:
        origins = sorted(validation_rows[validation_rows.horizon_quarters.eq(horizon)].origin_quarter.unique())
        for origin in origins:
            validation = validation_rows[(validation_rows.horizon_quarters.eq(horizon)) & (validation_rows.origin_quarter.eq(origin))].sort_values("country")
            if validation.empty or not validation.split.eq("validation").all():
                raise TestAccessError("non-validation forecast target requested during tuning")
            train = eligible_training(samples, horizon, origin).sort_values(["origin_quarter", "country"])
            if train.split.eq("test").any():
                raise TestAccessError("final-test label requested during tuning")
            for model in ("persistence", "arima", "var", "ets", "dynamic_factor"):
                for candidate_index, parameters in enumerate(candidates[model]):
                    prediction = _classical_predictions(model, validation, panel, countries, horizon)
                    _record(scores, residuals, model, "none", "deterministic", horizon, origin, validation, prediction, f"{model}:{candidate_index}", parameters)
            for candidate_index, parameters in enumerate(candidates["ridge"]):
                _record(scores, residuals, "ridge", "none", "deterministic", horizon, origin, validation, _ridge_predictions(train, validation, panel, countries, qidx, adjacency), f"ridge:{candidate_index}", parameters)
            for candidate_index, parameters in enumerate(candidates["gradient_boosting"]):
                _record(scores, residuals, "gradient_boosting", "none", "deterministic", horizon, origin, validation, _boosting_predictions(train, validation, panel, countries, qidx, adjacency), f"gradient_boosting:{candidate_index}", parameters)
            for seed in cfg["seeds"]:
                for model in ("mlp", "lstm", "tcn"):
                    for candidate_index, parameters in enumerate(candidates[model]):
                        prediction = _neural_predictions(model, "none", seed, train, validation, panel, countries, qidx, adjacency, cfg, None, f"validation_only_{model}")
                        _record(scores, residuals, model, "none", str(seed), horizon, origin, validation, prediction, f"{model}:{candidate_index}", parameters)
                for variant in cfg["graph_variants"]:
                    for model in ("gcn", "temporal_graph"):
                        for candidate_index, parameters in enumerate(candidates[model]):
                            prediction = _neural_predictions(model, variant, seed, train, validation, panel, countries, qidx, adjacency, cfg, None, f"validation_only_{model}_{variant}")
                            _record(scores, residuals, model, variant, str(seed), horizon, origin, validation, prediction, f"{model}:{candidate_index}", parameters)

    score_frame = pd.DataFrame(scores)
    selected, selected_scores = select_by_validation_mae(score_frame.groupby(["model", "candidate_id"], as_index=False).agg(mae=("mae", "mean"), rmse=("rmse", "mean")))
    calibration = normal_calibration(pd.DataFrame(residuals))
    selected_parameters = {model: next(parameters for index, parameters in enumerate(candidates[model]) if f"{model}:{index}" == candidate_id) for model, candidate_id in selected.items()}
    records = [{
        "model": model,
        "candidate_parameters": candidates[model],
        "validation_period": ["2015Q1", "2016Q4"],
        "validation_metrics": {"mae": selected_scores[model], "rmse": float(score_frame[score_frame.model.eq(model)].rmse.mean())},
        "selected_parameters": selected_parameters[model],
        "selection_metric": "validation_origin_mean_mae_then_rmse",
        "interval_calibration_method": calibration[model],
        "timestamp": datetime.now(timezone.utc).isoformat(),
    } for model in cfg["models"]]
    provenance = build_execution_provenance(CONFIG)
    manifest = {
        "run_id": f"tuning-{provenance['run_id']}",
        "git_commit": git_commit(),
        "configuration_id": cfg["configuration_id"],
        "candidate_registry": candidates,
        "validation_losses": score_frame.to_dict("records"),
        "selected_parameters": selected_parameters,
        "calibration": calibration,
        "records": records,
        "validation_period": ["2015Q1", "2016Q4"],
    }
    path = write_tuning_manifest(manifest)
    print(f"v2.1 validation-only tuning completed: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
