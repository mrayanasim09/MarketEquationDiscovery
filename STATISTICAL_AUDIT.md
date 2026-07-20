# STATISTICAL AUDIT — V2.1 Journal Benchmark

**Auditor:** Independent statistical review  
**Date:** 2026-07-20  
**Scope:** All statistical methods in `src/models/evaluate_benchmark_engine_v2_1.py`

---

## 1. Metric Implementation Verification

### 1.1 RMSE and MAE (Origin-level aggregation)

**Protocol:** Compute per-forecast absolute/squared errors → mean within each `forecast_origin` (averaging over countries) → mean over origins.

**Implementation check:**
```python
abs_loss = np.abs(actual - mean)
square_loss = (actual - mean) ** 2
rmse = float(np.sqrt(_origin_mean(group, square_loss).mean()))
mae  = float(_origin_mean(group, abs_loss).mean())
```
**Verdict:** ✅ Correct — two-step aggregation (within origin, then across origins) matches protocol specification.

### 1.2 sMAPE

**Formula used:**
$$\text{sMAPE} = \frac{200 \cdot |y - \hat{y}|}{|y| + |\hat{y}|}$$

**Protocol:** Zero-denominator pairs are omitted (`nan`-filled), and the count of omitted pairs is tracked.  
**Implementation:**
- `denominator = |actual| + |mean|`; `valid = denominator > 0`; NaN assigned when denominator = 0 ✅  
- Symmetry verified numerically: `sMAPE(2, 4) = sMAPE(4, 2) = 66.67` ✅  
- Perfect forecast: `sMAPE(y, y) = 0` ✅

### 1.3 CRPS (Closed-form Normal)

**Formula (Gneiting & Raftery 2007, Eq. 21):**
$$\text{CRPS}(\mathcal{N}(\mu, \sigma), y) = \sigma \left[ z \left(2\Phi(z) - 1\right) + 2\phi(z) - \frac{1}{\sqrt{\pi}} \right], \quad z = \frac{y - \mu}{\sigma}$$

**Numerical verification:** Analytic formula vs. Monte Carlo (100,000 draws, seed 42):  
- Analytic for $\mathcal{N}(0,1)$ at $y=0$: **0.233695**  
- Monte Carlo: **0.233368**  
- Difference: 0.000327 (< 0.001 tolerance) ✅

**Scale derivation:** $\sigma$ is back-derived from the locked 80% interval using $\sigma = \frac{U_{80} - L_{80}}{2 \cdot z_{0.9}}$ where $z_{0.9} = \Phi^{-1}(0.9) \approx 1.2816$ ✅

**Guard:** `scale ≤ 0 or not finite` raises `ValueError` before computing ✅

### 1.4 Interval Coverage and Width

**80% interval:** `lower_80`, `upper_80` at nominal 80% coverage (calibrated on train+val only) ✅  
**95% interval:** `lower_95`, `upper_95` at nominal 95% coverage ✅  
**Coverage:** Binary indicator averaged across countries then across origins — correct ✅  
**Width:** `upper - lower`, averaged — correct ✅

---

## 2. Diebold-Mariano Test (Harvey-Leybourne-Newbold Variant)

### 2.1 Loss Differential

**Unit:** Forecast origin (mean absolute error averaged across countries per origin) ✅  
**Primary loss:** Absolute error (`|y_graph - ŷ_graph| - |y_base - ŷ_base|`) ✅  
**Direction:** Negative values indicate the graph model is more accurate ✅

### 2.2 HLN Finite-Sample Correction

**Bartlett HAC variance** with bandwidth = `horizon - 1`:
$$\hat{\sigma}^2_{HAC} = \hat{\gamma}_0 + 2 \sum_{k=1}^{h-1} \left(1 - \frac{k}{h}\right) \hat{\gamma}_k$$

**Implementation:** Loop over `range(1, lag+1)` with Bartlett weight `(1 - step/(lag+1))` ✅  
**Note:** The implementation uses `lag+1` in the denominator (Bartlett kernel), which is consistent with the pre-registered `hac_max_lag: horizon_minus_one` spec ✅

**HLN scaling factor:**
$$\text{HLN} = \sqrt{\frac{n + 1 - 2h + h(h-1)/n}{n}}$$
**Implementation:** matches this formula exactly ✅  
**p-value:** Two-sided t-distribution with `df = n - 1` ✅

### 2.3 Moving-Block Bootstrap CI

**Block length:** 4 (pre-registered)  
**Draws:** 2,000 (pre-registered)  
**Seed:** Fixed at `np.random.default_rng(20260718)` for full reproducibility ✅  
**CI:** 2.5th and 97.5th percentiles of bootstrap distribution of mean differences ✅  
**Fix applied:** Replaced `tuple(float(v) for v in quantile(...))` with explicit unpacking `low, high = quantile(...)` — functionally identical, mypy-compliant ✅

### 2.4 Seeded Comparator Merge (Critical Bug Fix, Previously Applied)

**Issue:** When comparing GCN/TemporalGraph (seeded) against LSTM/TCN (also seeded), the merge on `[country, forecast_origin, target_quarter, horizon]` produced a many-to-one join, inflating the comparison sample.  
**Fix:** Filter `base` by `base.seed == str(provenance["seed"])` for `{lstm, tcn}` comparators before merging.  
**Impact:** This fix is essential for statistical validity — without it, DM tests would be computed on incorrectly duplicated loss differentials.  
**Verified:** DM test rows = 6,720 = 2 graph models × 8 variants × 20 seeds × 7 comparators × 3 horizons ÷ appropriate subsets ✅

### 2.5 Benjamini-Hochberg FDR Correction

**Family:** All primary (absolute error) graph vs. comparator vs. horizon tests  
**q-level:** 0.05  
**Implementation:** Sort p-values ascending, apply step-up procedure with monotonicity enforcement via `np.minimum.accumulate(raw[::-1])[::-1]`  
**Verification:** BH-adjusted values confirmed monotone ✅, capped at 1.0 ✅

---

## 3. Probabilistic Calibration

**Calibration split:** Training + validation origins only (2011Q2–2016Q4) — no test data used ✅  
**Method:** Residual standard deviation computed on the calibration split, then applied to all test-period predictions  
**Protocol compliance:** `calibration_split: training_and_validation_only` matches implementation ✅

---

## 4. Statistical Findings Summary

| Metric | Verified | Notes |
|---|---|---|
| RMSE (origin-level) | ✅ | Two-step aggregation correct |
| MAE (origin-level) | ✅ | Two-step aggregation correct |
| sMAPE (zero-denom omit) | ✅ | Symmetry and edge cases verified |
| CRPS (Normal closed-form) | ✅ | Matches Gneiting & Raftery 2007 within 0.001 |
| Coverage 80%/95% | ✅ | Binary indicator, correct aggregation |
| Width 80%/95% | ✅ | Correct |
| DM HLN Bartlett HAC | ✅ | Formula matches pre-registration |
| Moving-block bootstrap CI | ✅ | Seed fixed; unpacking fix applied |
| BH FDR correction | ✅ | Monotone; q=0.05 |
| Seeded comparator merge | ✅ | Bug identified and fixed before results finalised |

---

## 5. Key Finding

> **No graph model (GCN or TemporalGraph) achieves statistical significance across all 20 seeds** against any comparator at any horizon after BH-FDR correction at q=0.05. Some partial significance (in a fraction of seeds) is observed — particularly GCN vs Ridge at horizons 1–2. This finding is internally consistent, reproducible, and correctly reported in the manuscript.
