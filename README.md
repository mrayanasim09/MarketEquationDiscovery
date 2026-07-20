# Predicting Inflation Contagion: A Spatio-Temporal Graph Neural Network Approach to Trade-Linked Economies

![Python](https://img.shields.io/badge/Python-3.14-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Under%20Review-orange)

## Abstract
This repository contains the codebase and data processing pipelines for the research paper "Predicting Inflation Contagion: A Spatio-Temporal Graph Neural Network Approach to Trade-Linked Economies." The study investigates the efficacy of Graph Neural Networks (GNNs) in forecasting quarterly Consumer Price Index (CPI) inflation (Year-over-Year) across 20 European economies. By modeling economies as nodes and dynamic bilateral trade relationships as edges, we evaluate whether explicitly embedding cross-border trade linkages improves predictive accuracy over traditional univariate, multivariate, and non-graph deep learning baselines.

## Research Question and Motivation
Inflation is traditionally modeled as a domestic phenomenon influenced by monetary policy and local macroeconomic indicators. However, in highly integrated regions like Europe, inflation can propagate across borders through trade networks. This research asks: **Does explicitly modeling the dynamic, directed trade relationships between countries using Spatio-Temporal Graph Neural Networks improve inflation forecasting accuracy compared to state-of-the-art non-graph methods?**

## Methodology Summary
The benchmark incorporates:
- **Target Variable**: Quarterly CPI inflation (YoY).
- **Timeframes**: Training (2011Q2-2014Q4), Validation (2015Q1-2016Q4), and an Expanding-window Prospective Test (2017Q1-2025Q3).
- **Horizons**: 1, 2, and 4 quarters ahead ($h=1, 2, 4$).
- **Scale**: 38,380 model fits generating 781,740 forecast rows.
- **Evaluation**: Root Mean Squared Error (RMSE) and Mean Absolute Error (MAE), rigorously tested across 20 random seeds (42-61) for all neural architectures to ensure statistical robustness.

## Key Results
Our comprehensive evaluation reveals that while spatio-temporal graph models demonstrate competitive performance, simpler graph topologies often outperform complex, dense trade networks. Notably, the `temporal_graph` model paired with an `identity_no_trade` graph (effectively isolating domestic temporal dynamics) achieved top rankings at $h=2$ and $h=4$. Statistical testing indicates that no single graph formulation consistently achieved significant superiority across all 20 initialization seeds when compared to strong non-graph baselines.

| Horizon | Top Performing Model | Top Graph Structure |
|---------|----------------------|---------------------|
| $h=1$ | Ensemble / Baseline | N/A |
| $h=2$ | temporal_graph | identity_no_trade |
| $h=4$ | temporal_graph | identity_no_trade |

## Repository Structure
```text
repository/
├── data/                  # Raw and processed datasets
│   ├── raw/               # Eurostat and IMF/World Bank data
│   └── processed/         # Cleaned features and adjacency matrices
├── models/                # Model implementations
│   ├── baselines/         # ARIMA, VAR, Persistence
│   ├── machine_learning/  # Ridge, Gradient Boosting, MLP
│   └── graph_neural/      # GCN, Temporal Graph models
├── scripts/               # Execution scripts
│   ├── validate.py        # Data validation
│   ├── benchmark.py       # Main training and evaluation loop
│   ├── analyze.py         # Results aggregation and significance testing
│   ├── report.py          # Tables and figures generation
│   └── manuscript.py      # Final LaTeX manuscript compilation
├── environment.yml        # Conda environment definition
├── requirements.txt       # Pip dependencies
├── CITATION.cff           # Citation metadata
├── LICENSE                # MIT License
└── README.md              # This file
```

## Installation

### Option 1: Virtual Environment (Pip)
```bash
python3.14 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Option 2: Conda
```bash
conda env create -f environment.yml
conda activate inflation-gnn
```

## Quick Start / Reproduction
The pipeline is designed to be executed sequentially. Run the following commands from the repository root:

1. **Validate Data**:
   ```bash
   python scripts/validate.py
   ```
2. **Run Benchmarks**:
   ```bash
   python scripts/benchmark.py
   ```
3. **Analyze Results**:
   ```bash
   python scripts/analyze.py
   ```
4. **Generate Report/Figures**:
   ```bash
   python scripts/report.py
   ```
5. **Compile Manuscript**:
   ```bash
   python scripts/manuscript.py
   ```

## Dataset Description
- **Macroeconomic Data**: Quarterly indicators including CPI, GDP growth, and interest rates sourced from IMF International Financial Statistics (IFS) and the World Bank.
- **Trade Data**: Quarterly bilateral import/export volumes sourced from Eurostat (Comext), used to construct dynamic weighted adjacency matrices representing the evolving European trade network.

## Models Evaluated

| Category | Models |
|----------|--------|
| **Naïve & Statistical** | `persistence`, `ARIMA`, `VAR`, `ETS`, `dynamic_factor` |
| **Machine Learning** | `ridge`, `gradient_boosting` |
| **Deep Learning (Non-Graph)** | `MLP`, `LSTM`, `TCN` |
| **Graph Neural Networks** | `GCN`, `temporal_graph` |

## Graph Variants (Adjacency Matrix Formulations)

| Variant | Description |
|---------|-------------|
| `directed_trade` | Standard directed trade volumes. |
| `log_trade` | Log-transformed trade volumes to handle scale disparities. |
| `import_dependence` | Trade normalized by the importing country's total imports. |
| `top_k_incoming` | Filtered matrix retaining only the top-k trading partners. |
| `reversed` | Transposed trade network (exports vs imports). |
| `undirected` | Symmetrized trade relationships. |
| `degree_preserving_random` | Null model: randomized edges preserving node degrees. |
| `identity_no_trade` | Identity matrix; eliminates cross-country information flow. |

## Citation
If you use this code or data in your research, please cite our work:

```bibtex
@article{asim2026predicting,
  title={Predicting Inflation Contagion: A Spatio-Temporal Graph Neural Network Approach to Trade-Linked Economies},
  author={Asim, Rayyan},
  journal={SSRN Preprint},
  year={2026},
  url={https://github.com/mrayanasim09/MarketEquationDiscovery}
}
```

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements
We acknowledge the data provision by Eurostat, the IMF, and the World Bank, which made this empirical analysis possible.
