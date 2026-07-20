# REPRODUCIBILITY REPORT — V2.1 Journal Benchmark

**Date:** 2026-07-20  
**Run ID:** `v2-a7fef2973468-88a5c7683254-2026-07-19T152835.502880+0000`  
**Git Commit:** `a7fef297346868ddf2ae178e45d47ab27cc55317`  
**Publication Commit:** `b6d26d4` (publication packaging, no scientific changes)

---

## 1. Cryptographic Hash Registry (Immutable)

All hashes were computed with SHA-256. Verification performed and confirmed on 2026-07-20.

| File | Rows | SHA-256 | Status |
|---|---|---|---|
| `experiments/results/v2_1/forecasts.parquet` | 781,740 | `f988dfb1249fd77739ffdecae6323073c8cbd363c48412d4e3248454f98b3798` | ✅ VERIFIED |
| `experiments/results/v2_1/metrics.parquet` | 9,288 | `231b349f6ab2d5bba4cc42cf047c2cedd88300f5836302b5e79e4ff9071abcaa` | ✅ VERIFIED |
| `experiments/results/v2_1/dm_tests.parquet` | 6,720 | `a57490a079580c48458143961f9e8dde6f7cf72e77818a71246d73496c397048` | ✅ VERIFIED |
| `experiments/results/v2_1/configs/benchmark_engine_v2_1.json` | — | `88a5c7683254bc53a67b05e9837b272a80e06576972abc818f5f16b1130377f6` | ✅ VERIFIED |
| `experiments/results/v2_1/tuning/tuning_manifest.json` | — | `bad266128befed33c9478269ccd37ff50fe03fe683b9d98a9e089b6a4dcb5b15` | ✅ VERIFIED |

**Verification command:**
```bash
shasum -a 256 experiments/results/v2_1/forecasts.parquet
shasum -a 256 experiments/results/v2_1/metrics.parquet
shasum -a 256 experiments/results/v2_1/dm_tests.parquet
```

---

## 2. Contract Validation

```
V2.1 CONTRACT VALIDATION PASSED
configuration_id=v2.1-journal-prospective-20260718
configuration_sha256=88a5c7683254bc53a67b05e9837b272a80e06576972abc818f5f16b1130377f6
```

The contract validator checks:
- Engine version exactly `2.1`
- Model registry matches 12 pre-specified models
- Graph variant registry matches 8 variants (exact order)
- Metric registry complete (RMSE, MAE, sMAPE, CRPS, coverage, width)
- Split boundaries locked (train: 2011Q2–2014Q4; val: 2015Q1–2016Q4; test: 2017Q1–2025Q3)
- DM test parameters locked (HLN Bartlett HAC, block bootstrap, BH FDR q=0.05)
- Execution integrity flags (transactional staging, no partial registry)
- Tuning manifest present and valid before benchmark permitted to run

---

## 3. Environment Provenance

| Property | Value |
|---|---|
| Operating System | macOS-26.5.2-arm64 (Apple Silicon) |
| Python Version | 3.14.6 |
| numpy | 2.5.1 |
| pandas | 3.0.3 |
| scipy | 1.18.0 |
| statsmodels | 0.14.6 |
| scikit-learn | 1.9.0 |
| torch | 2.13.0 |
| pyarrow | 25.0.0 |
| matplotlib | 3.11.1 |
| seaborn | 0.13.2 |

---

## 4. Execution Summary

- **Wall-clock time:** 11 hours 33 minutes 25 seconds
- **Start:** `2026-07-19T15:28:35+00:00`
- **End:** `2026-07-20T03:02:01+00:00`
- **Total model fits:** 38,380
- **Hardware:** Apple Silicon M-series, 10 CPU cores, 16 GB RAM, CPU execution (no GPU)
- **PyTorch device:** CPU (MPS available but not used for training)

---

## 5. Reproduction Instructions

To reproduce from the frozen configuration:

```bash
# Step 1 — Clone and set up
git clone https://github.com/mrayanasim09/MarketEquationDiscovery
cd MarketEquationDiscovery
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Step 2 — Validate contract (< 5 seconds)
python -m src.models.validate_v2_1_contract

# Step 3 — Execute benchmark (~12 hours on Apple Silicon)
python -m src.models.run_benchmark_engine_v2_1 --execute \
  2>&1 | tee experiments/results/v2_1/execution_log.txt

# Step 4 — Validate results and hashes
python -m src.models.validate_v2_1_results

# Step 5 — Statistical analysis
python -m src.models.analyze_v2_1_results

# Step 6 — Manuscript tables and figures
python -m src.models.generate_v2_1_manuscript

# Step 7 — Final audit report
python -m src.models.generate_v2_1_report
```

Or run the end-to-end script (skips step 3 to avoid re-running the benchmark):
```bash
./reproduce.sh    # Unix/macOS
.\reproduce.ps1   # Windows PowerShell
```

---

## 6. Reproducibility Risks

| Risk | Severity | Notes |
|---|---|---|
| ARIMA/ETS convergence warnings | Low | `statsmodels` ConvergenceWarnings are non-fatal; fallback to persistence is triggered |
| Floating-point non-determinism | Low | Neural models use fixed seeds (42–61); CPU execution is deterministic per-platform |
| Cross-platform float parity | Medium | Results on non-ARM hardware will differ in last decimal digits; SHA-256 hashes will differ |
| Python version sensitivity | Low | Tested on 3.14; minor differences expected on 3.10–3.13 in rounding |
| `matplotlib` font cache | Trivial | First run builds font cache; subsequent runs are unaffected |

> **Note:** The published SHA-256 hashes correspond to a single execution on Apple Silicon (arm64, macOS). Re-execution on different hardware will produce numerically close but not bit-identical results due to floating-point ordering differences. The statistical conclusions are expected to be identical.

---

## 7. Validation Results (Post-Execution)

```
Validation PASSED successfully.
Output hashes registry written to experiments/results/v2_1/metadata/output_hashes.json
```

All checks passed:
- forecasts.parquet: 781,740 rows ✅
- metrics.parquet: 9,288 rows ✅  
- dm_tests.parquet: 6,720 rows ✅
- All required models present ✅
- All required graph variants present ✅
- All seeds 42–61 present for neural/graph models ✅
- All horizons [1, 2, 4] present ✅
- All metrics present ✅
- Provenance fields non-null ✅
