"""Additional non-graph baselines for the reconstructed engine.

BayesianShrinkageVAR: defined for reference. The current benchmark uses
statsmodels VAR(1) as the named 'var' baseline; this class is kept here
for future extension to Bayesian shrinkage priors (Minnesota-type), which
is a direction recommended for future work in the paper.

gradient_boosting_regressor: thin factory for HistGradientBoostingRegressor,
used in the benchmark as the 'gradient_boosting' model family.
"""
from __future__ import annotations

import numpy as np


class BayesianShrinkageVAR:
    """Conjugate ridge/shrinkage VAR approximation.

    Not used in the v2.1 benchmark — retained for future extension. The
    current 'var' baseline is statsmodels VAR(1) without Bayesian priors.
    A Minnesota-prior BVAR is a recommended addition to future work.
    """
    def __init__(self, lags: int = 1, prior_precision: float = 1.0):
        self.lags = lags
        self.prior_precision = prior_precision

    def fit(self, y: np.ndarray):
        if len(y) <= self.lags:
            raise ValueError("insufficient history")
        x = np.array([y[t - self.lags:t].ravel() for t in range(self.lags, len(y))])
        target = y[self.lags:]
        self.coef = np.linalg.solve(
            x.T @ x + self.prior_precision * np.eye(x.shape[1]), x.T @ target
        )
        self.last = y[-self.lags:].copy()
        return self

    def forecast(self, steps: int) -> np.ndarray:
        history = list(self.last.copy())
        out = []
        for _ in range(steps):
            nxt = np.asarray(history[-self.lags:]).ravel() @ self.coef
            out.append(nxt)
            history.append(nxt)
        return np.asarray(out)


def gradient_boosting_regressor():
    """Factory for HistGradientBoostingRegressor used in benchmark.

    Note: random_state=0 is fixed (deterministic); the benchmark runs this
    model once without re-seeding across the 20 neural seeds, since GBM
    has negligible sensitivity to random initialisation relative to neural models.
    """
    try:
        from sklearn.ensemble import HistGradientBoostingRegressor
    except ImportError as exc:
        raise RuntimeError(
            "gradient boosting requires scikit-learn; install it before the final benchmark run"
        ) from exc
    return HistGradientBoostingRegressor(
        max_iter=100, learning_rate=0.05, l2_regularization=1.0, random_state=0
    )
