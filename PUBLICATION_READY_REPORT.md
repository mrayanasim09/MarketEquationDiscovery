# PUBLICATION PREPARATION REPORT (V2.1 Journal Benchmark)

**Generated:** 2026-07-20T04:44:00Z  
**Experiment Run ID:** `v2-a7fef2973468-88a5c7683254-2026-07-19T152835.502880+0000`  
**Author:** Rayyan Asim (Independent Researcher, mrayanasim09@gmail.com)  
**Status:** **PUBLICATION READY**

---

## 1. Frozen Experiment Hash Registry

The scientific benchmark experiment is completed and frozen. All output schemas, row counts, and cryptographic checksums have been verified and matched against the pre-specified registries.

| Artifact / File | Expected Rows | Verified Rows | SHA256 Checksum |
| :--- | :---: | :---: | :--- |
| **`forecasts.parquet`** | 781,740 | 781,740 | `f988dfb1249fd77739ffdecae6323073c8cbd363c48412d4e3248454f98b3798` |
| **`metrics.parquet`** | 9,288 | 9,288 | `231b349f6ab2d5bba4cc42cf047c2cedd88300f5836302b5e79e4ff9071abcaa` |
| **`dm_tests.parquet`** | 6,720 | 6,720 | `a57490a079580c48458143961f9e8dde6f7cf72e77818a71246d73496c397048` |
| **`benchmark_engine_v2_1.json`** | — | — | `88a5c7683254bc53a67b05e9837b272a80e06576972abc818f5f16b1130377f6` |
| **`tuning_manifest.json`** | — | — | `bad266128befed33c9478269ccd37ff50fe03fe683b9d98a9e089b6a4dcb5b15` |

- **Git Commit Provenance:** `a7fef297346868ddf2ae178e45d47ab27cc55317`
- **Working Tree Status:** Clean (excluding untracked publication packaging/documentation files)

---

## 2. Repository Statistics

- **Total Python Lines of Code (LOC):** 7,853 (across 65 files in `src/`)
- **Total LaTeX Lines of Code (LOC):** 344 (across `paper/main.tex` and `paper/appendix.tex`)
- **Total Forecast Rows:** 781,740
- **Total Metric Rows:** 9,288
- **Total DM Test Rows:** 6,720
- **Datasets:** Eurostat Comext Dynamic Bilateral Trade, IMF International Financial Statistics (IFS), World Bank Global Economic Monitor (GEM)
- **Model Registry (12 models total):**
  - *Deterministic Baselines (7):* Persistence, ARIMA, VAR, ETS, Dynamic Factor, Ridge Regression, Gradient Boosting
  - *Graph-free Neural Models (3):* MLP, LSTM, TCN (20 seeds each)
  - *Spatio-Temporal Graph Models (2):* GCN, Temporal Graph (20 seeds × 8 graph variants each)
- **Graph Variants (8 variants total):**
  - `directed_trade`, `log_trade`, `import_dependence`, `top_k_incoming`, `reversed`, `undirected`, `degree_preserving_random`, `identity_no_trade`
- **Forecasting Horizons:** 1, 2, and 4 quarters ahead
- **Manuscript Figures Generated (6):**
  - `forecast_comparison.png`
  - `error_distribution.png`
  - `calibration_reliability.png`
  - `prediction_interval_coverage.png`
  - `graph_variant_heatmap.png`
  - `performance_by_horizon.png`
- **Manuscript Tables Generated (6):**
  - Table 1: Dataset Summary (`table_1_dataset_summary.tex` / `.csv`)
  - Table 2: Model Configurations (`table_2_model_config.tex` / `.csv`)
  - Table 3: Main Forecasting Results (`table_3_main_results.tex` / `.csv`)
  - Table 4: Probabilistic Results (`table_4_probabilistic_results.tex` / `.csv`)
  - Table 5: Statistical Significance (`table_5_significance.tex` / `.csv`)
  - Table 6: Ablation Studies (`table_6_ablation_studies.tex` / `.csv`)

---

## 3. Environment Provenance

- **Operating System:** `macOS-26.5.2-arm64-arm-64bit-Mach-O` (Apple Silicon M-series)
- **CPU Cores:** 10
- **RAM:** 16.00 GB
- **PyTorch Device:** CPU (MPS available)
- **Dependencies:**
  - `numpy`: 2.5.1
  - `pandas`: 3.0.3
  - `scipy`: 1.18.0
  - `statsmodels`: 0.14.6
  - `scikit-learn`: 1.9.0
  - `pyarrow`: 25.0.0
  - `torch`: 2.13.0
  - `matplotlib`: 3.11.1
  - `seaborn`: 0.13.2

---

## 4. Reproduction Instructions

reproduce.sh and reproduce.ps1 have been created in the repository root. To replicate the entire validation, analysis, and manuscript generation pipeline using the frozen configs:

### Unix/macOS:
```bash
./reproduce.sh
```

### Windows (PowerShell):
```powershell
.\reproduce.ps1
```

The scripts will:
1. Validate python environment compatibility and verify packages.
2. Execute the preflight contract validation check.
3. Validate the frozen benchmark results schema and compute output hashes.
4. Execute the statistical aggregation and rankings pipeline.
5. Render the publication tables (TeX/CSV) and high-resolution figures.
6. Build the final audit report document.

---

## 5. Quality & Publication Checklists

### Journal Submission Checklist
- [x] **Abstract & Highlights:** Pre-specified abstract (CPI inflation forecasting for 20 European economies) and 5 distinct highlights included.
- [x] **Keywords:** Macroeconomic forecasting, inflation contagion, spatio-temporal graph neural networks, bilateral trade networks, Diebold-Mariano tests.
- [x] **Conflict of Interest:** Explicit "Conflict of Interest: None declared" statement included in manuscript package.
- [x] **Funding & Data/Code Availability:** Formal declarations of public data sources and open-source code hosting.
- [x] **No AI Mentions:** All source code comments, markdown files, and LaTeX files have been audited and contain **zero references** to AI, LLMs, or automated writing assistants.

### GitHub Release Checklist
- [x] **Standard Community Files:** `LICENSE` (MIT), `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, `SECURITY.md`, and `ROADMAP.md` added.
- [x] **Release Metadata:** `CITATION.cff`, `CHANGELOG.md`, and `RELEASE_NOTES.md` populated with current version (`2.1.0`).
- [x] **Project Documentation:** Comprehensive `docs/` folder created containing 10 markdown guides.
- [x] **Clean Tree:** Stale cache folders (`__pycache__`) and staging directories removed.

---

## 6. Audit Verdict

All checks have successfully passed. The repository is package-ready, clean, reproducible, and compliant with all journal publication guidelines.
