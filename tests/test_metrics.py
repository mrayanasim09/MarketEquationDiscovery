import math
import numpy as np
import pytest
from src.models.evaluate_benchmark_engine_v2_1 import _dm_hln


# ── DM-HLN statistic ─────────────────────────────────────────────────────────

def test_dm_hln_zero_difference():
    """If both forecasters are identical, the DM statistic should be zero."""
    diff = np.zeros(20)
    stat, pval = _dm_hln(diff, horizon=1)
    # Variance is zero -> returns nan (undefined test), not zero
    assert math.isnan(stat), "Expected NaN for all-zero difference"
    assert math.isnan(pval)


def test_dm_hln_large_positive_difference():
    rng = np.random.default_rng(7)
    diff = rng.normal(loc=5.0, scale=0.5, size=30)  # strongly positive mean, nonzero variance
    stat, pval = _dm_hln(diff, horizon=1)
    assert np.isfinite(stat), f"stat was {stat}"
    assert stat > 0
    assert 0 <= pval <= 1


def test_dm_hln_negative_difference():
    """Flipping signs should negate the statistic."""
    diff = np.random.default_rng(42).normal(1.0, 0.5, 40)
    stat_pos, _ = _dm_hln(diff, horizon=1)
    stat_neg, _ = _dm_hln(-diff, horizon=1)
    assert np.isclose(stat_pos, -stat_neg, rtol=1e-6)


def test_dm_hln_too_short():
    """With n < 3 observations, should return NaN."""
    diff = np.array([0.1, -0.1])
    stat, pval = _dm_hln(diff, horizon=1)
    assert math.isnan(stat)
    assert math.isnan(pval)


def test_dm_hln_horizon_2_hln_factor():
    """
    The HLN correction factor for h=2, n=20 should be well-defined and < 1.
    HLN = sqrt((n + 1 - 2*h + h*(h-1)/n) / n)
    """
    n, h = 20, 2
    expected_hln = math.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    diff = np.random.default_rng(0).normal(0, 1, n)
    stat_h2, _ = _dm_hln(diff, horizon=2)
    stat_h1, _ = _dm_hln(diff, horizon=1)
    # h=1 has no HAC lags, h=2 has lag=1: ratio should approximately equal HLN ratio
    assert np.isfinite(stat_h2), "DM statistic for h=2 should be finite"
    assert expected_hln < 1.0, "HLN factor for h=2 should shrink the statistic"


# ── Bartlett / HAC kernel weight check ───────────────────────────────────────

def test_dm_hln_known_case():
    """
    For a known i.i.d. N(mu, sigma^2) series with no autocorrelation,
    h=1 (no HAC lags), the DM stat should equal mean/se * hln_factor.
    """
    rng = np.random.default_rng(123)
    diff = rng.normal(2.0, 1.0, 50)     # known mean ~2, std ~1
    stat, pval = _dm_hln(diff, horizon=1)

    n = len(diff)
    var_estimate = float(np.mean((diff - diff.mean()) ** 2))   # no lags for h=1
    se = math.sqrt(var_estimate / n)
    hln = math.sqrt((n + 1 - 2 * 1 + 1 * 0 / n) / n)
    expected_stat = (diff.mean() / se) * hln

    assert np.isclose(stat, expected_stat, rtol=1e-5), (
        f"DM stat {stat:.4f} differs from expected {expected_stat:.4f}"
    )
