# Reproducibility Guide — v2.1

This document describes how to fully reproduce all experiments, results tables,
and figures reported in the manuscript.

## Requirements

- Python 3.11 (tested; 3.10+ should work)
- ~8 GB RAM; GPU optional but recommended for neural models

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Quick Start (Full Pipeline)

```bash
bash reproduce.sh
```

This runs the following six steps end-to-end:

| Step | Module | Description |
|------|--------|-------------|
| 1 | `validate_v2_1_contract` | Verifies SHA256 hashes of processed inputs |
| 2 | `run_benchmark_engine_v2_1` | Runs all 12 model families × 8 graphs × 3 horizons × 20 seeds |
| 3 | `validate_v2_1_results` | Verifies SHA256 hashes of all output artefacts |
| 4 | `analyze_v2_1_results` | Generates Tables 3–6 and significance summaries |
| 5 | `generate_v2_1_manuscript` | Generates Tables 1–6 (`.tex` + `.csv`) and 6 diagnostic figures |
| 6 | `generate_v2_1_report` | Produces the plain-text summary report |

Estimated runtime: ~12 hours on Apple Silicon M-series; longer on CPU-only hardware.

## Output Paths (v2.1)

All outputs are written to `experiments/results/v2_1/`:

- `forecasts.parquet` — 781,740 forecast rows (all models × countries × origins × seeds)
- `metrics.parquet` — MAE / RMSE / CRPS per (model, variant, horizon, seed)
- `dm_tests.parquet` — Diebold-Mariano test statistics and BH-corrected p-values
- `manuscript/table_*.tex` and `.csv` — LaTeX and CSV tables
- `manuscript/*.png` — Six diagnostic figures

## Data

Inputs are pre-committed to the repository under `data/processed/v2/`. No external
download step is required. All input hashes are pre-registered in
`data/processed/v2/transformation_validation.json`.

## Random Seeds

20 random seeds (42–61) are used for all stochastic models to quantify estimation uncertainty. Deterministic baselines (ARIMA, Ridge, Persistence, etc.) use "deterministic" as their seed value.

## Contact

For reproduction issues: mrayanasim09@gmail.com