# Reproducibility Guide

## Environment Setup
To ensure exact reproduction of the experimental results, we provide a strict dependency specification. All analyses were conducted using Python 3.10+ within an isolated virtual environment.

### Prerequisites
- Python 3.10 or higher
- Git

### Installation Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/rayyanasim/inflation-contagion.git
   cd inflation-contagion
   ```
2. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Execution Protocol
The experimental pipeline is orchestrated through the main execution script. The pipeline is designed to be transactional and resumable.

### Running the Benchmark
```bash
python -m src.benchmark.run_all
```
This command initializes the expanding window evaluation protocol, running through 38,380 model fits.

### Expected Outputs
The pipeline generates the following output directories:
- `data/processed/`: Contains the cleaned and transformed panel data.
- `results/forecasts/`: Parquet files containing 781,740 forecast rows.
- `results/metrics/`: Parquet files with 9,288 metric rows (MAE, RMSE, sMAPE, CRPS).
- `results/statistics/`: Diebold-Mariano test results (6,720 rows).

## Hash Verification
To guarantee data integrity, cryptographic hashes (SHA-256) of the input datasets are verified before pipeline execution. If you are using your own local copy of the Eurostat or IMF data, ensure it matches the hashes specified in `config/data_hashes.json`. 

## Troubleshooting
- **Statsmodels Convergence Warnings**: When running deterministic baselines (e.g., ARIMA, VAR), you may encounter convergence warnings. This is expected due to the short initial training window (2011Q2-2014Q4). The pipeline handles these gracefully via fallback mechanisms.
- **Memory Limitations**: Evaluating 20 stochastic seeds across multiple models is computationally intensive. If you experience out-of-memory errors, reduce the number of parallel workers in `config/benchmark.yaml`.

For more details on result interpretation, refer to `RESULTS_GUIDE.md`.
