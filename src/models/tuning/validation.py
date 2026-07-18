"""Validation-only selection primitives for the v2.1 engine.

The runner never calls these on final-test rows.  A future authorized tuning run
must supply validation-origin losses produced from information available at each
validation origin and then persist the selections through ``write_tuning_manifest``.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def select_by_validation_mae(losses: pd.DataFrame) -> tuple[dict[str, Any], dict[str, float]]:
    """Select candidates by validation MAE, then RMSE, then candidate identifier."""
    required = {"model", "candidate_id", "mae", "rmse"}
    missing = required - set(losses.columns)
    if missing:
        raise ValueError(f"validation loss table lacks columns: {sorted(missing)}")
    if losses.empty or losses[["mae", "rmse"]].isna().any().any():
        raise ValueError("validation selection requires non-empty finite MAE and RMSE")
    ordered = losses.sort_values(["model", "mae", "rmse", "candidate_id"], kind="stable")
    winners = ordered.groupby("model", as_index=False).first()
    selected = {str(row.model): str(row.candidate_id) for row in winners.itertuples(index=False)}
    scores = {str(row.model): float(row.mae) for row in winners.itertuples(index=False)}
    return selected, scores


def normal_calibration(residuals: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Estimate per-model validation residual scales for normal predictive intervals."""
    required = {"model", "actual", "mean"}
    missing = required - set(residuals.columns)
    if missing:
        raise ValueError(f"validation residual table lacks columns: {sorted(missing)}")
    output: dict[str, dict[str, float]] = {}
    for model, group in residuals.groupby("model", sort=True):
        scale = float(np.std(group["actual"].to_numpy(float) - group["mean"].to_numpy(float), ddof=1))
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError(f"validation residual scale is invalid for {model}")
        output[str(model)] = {"distribution": "normal", "residual_scale": scale}
    return output
