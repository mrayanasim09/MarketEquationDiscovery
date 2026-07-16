# Inflation Forecasting with Spatio-Temporal Graph Neural Networks

This repository contains the replication code and data for the paper "Inflation Forecasting with Spatio-Temporal Graph Neural Networks" (SSRN, 2026).

## Overview

We forecast quarterly CPI inflation for 23 major economies using a spatio-temporal graph neural network (ST-GNN) built on dynamic CEPII bilateral trade networks. The model combines graph convolutional networks (GCN) with LSTM sequence modeling and a persistence skip connection.

## Key Features

- **ST-GNN architecture**: GCN + LSTM + persistence skip connection
- **Trade network topology**: Dynamic CEPII BACI bilateral trade flows
- **Rolling expanding-window evaluation**: Pseudo-real-time forecast protocol
- **Explainability**: Integrated Gradients on trade edge weights
- **Robust statistical testing**: Cluster-robust Diebold-Mariano tests and TOST equivalence tests

## Revisions (July 2026)

This version addresses reviewer concerns with the following updates:

- Added apples-to-apples baseline comparison (same country sets for ARIMA, VAR, ST-GNN)
- Implemented cluster-robust Diebold-Mariano tests (clustered by quarter)
- Added TOST equivalence test with pre-specified 10% margin
- Added non-shock quarter (2017Q2) baseline for explainability comparison
- Updated manuscript to present results honestly without misleading "competitive" framing
- Added formal definition of trade-versus-domestic attribution ratio
- Added data/code availability section
- Acknowledged COVID onset in grid search validation window
- Added limitation about single test window covering two unusual regimes
- Stated novelty more confidently in introduction

## Installation

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Data

Data sources:
- **CEPII BACI**: Bilateral trade flows (https://www.cepii.fr/CEPII/en/bdd_modele/presentation.asp?id=1)
- **FRED**: CPI, GDP, exchange rates (https://fred.stlouisfed.org/)
- **World Bank**: Policy rates (https://data.worldbank.org/)
- **Brent crude**: Energy price index

Processed datasets are included in `results/data/`. To reprocess from raw sources, run:
```bash
python -m src.process_data
```

## Usage

### Run baseline forecasts (ARIMA, VAR)
```bash
python -m src.run_baselines
```

### Run ST-GNN evaluation
```bash
python -m src.run_evaluation
```

### Run explainability analysis
```bash
python -m src.run_explainability
```

### Generate LaTeX tables
```bash
python -m src.build_paper_tables
```

### Build the paper PDF
```bash
cd paper
./build.sh
```

## Results

- **Baseline metrics**: `results/baselines/metrics.json`
- **ST-GNN evaluation**: `results/evaluation/evaluation_report.json`
- **Explainability**: `results/explainability/explainability_report.json`
- **LaTeX tables**: `paper/generated/`

## File Structure

```
.
├── src/                    # Source code
│   ├── run_baselines.py   # ARIMA/VAR baseline forecasts
│   ├── run_evaluation.py  # ST-GNN evaluation
│   ├── run_explainability.py  # Integrated Gradients analysis
│   ├── stgnn_train.py     # Training utilities
│   ├── stgnn_data.py      # Data loading
│   ├── stgnn_explain.py   # Explainability functions
│   └── ...
├── paper/                  # LaTeX manuscript
│   ├── main.tex
│   ├── references.bib
│   └── generated/         # Auto-generated tables
├── results/               # Results and outputs
│   ├── baselines/
│   ├── evaluation/
│   ├── explainability/
│   └── data/
└── requirements.txt
```

## Citation

```bibtex
@article{inflation_stgnn_2026,
  title={Inflation Forecasting with Spatio-Temporal Graph Neural Networks},
  author={Author},
  journal={SSRN},
  year={2026},
  url={https://github.com/mrayanasim09/MarketEquationDiscovery}
}
```

## License

MIT License

## Contact

For questions or issues, please open an issue on GitHub or contact the author.
