"""Pre-specified candidate generation; this module never reads test rows."""
from __future__ import annotations

from typing import Any


def candidate_registry(config: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    """Return the finite, documented candidate set evaluated on validation origins only.

    Numerical values are inherited from the locked configuration.  This function
    deliberately exposes no test-data-dependent branching.
    """
    hidden = int(config["hidden_dim"])
    epochs = int(config["epochs"])
    learning_rate = float(config["learning_rate"])
    return {
        "persistence": [{"method": "last_available_cpi_yoy"}],
        "arima": [{"order": [1, 0, 0]}],
        "var": [{"lags": 1}],
        "ets": [{"trend": None, "seasonal": None, "damped_trend": False}],
        "dynamic_factor": [{"k_factors": 1, "factor_order": 1, "error_order": 0}],
        "ridge": [{"penalty": 1.0}],
        "gradient_boosting": [{"max_iter": 100, "learning_rate": 0.05, "l2_regularization": 1.0}],
        "mlp": [{"hidden_dim": hidden, "epochs": epochs, "learning_rate": learning_rate}],
        "lstm": [{"hidden_dim": hidden, "epochs": epochs, "learning_rate": learning_rate}],
        "tcn": [{"hidden_dim": hidden, "epochs": epochs, "learning_rate": learning_rate}],
        "gcn": [{"hidden_dim": hidden, "epochs": epochs, "learning_rate": learning_rate}],
        "temporal_graph": [{"hidden_dim": hidden, "epochs": epochs, "learning_rate": learning_rate}],
    }
