# Benchmark Protocol

## Overview
The evaluation framework employs a strict expanding window protocol to simulate realistic macroeconomic forecasting conditions, preventing forward-looking bias and data leakage. 

## Temporal Splits
The panel dataset of 20 European countries spans from 2011Q2 to 2025Q3. The temporal splits are defined as follows:
- **Initial Training Window**: 2011Q2 - 2014Q4
- **Validation Window**: 2015Q1 - 2016Q4 (used for hyperparameter tuning)
- **Test Window**: 2017Q1 - 2025Q3 (used for final evaluation)

## Expanding Window Mechanics
For each step $t$ in the test window:
1. Models are trained on data from $T_0$ to $t-1$.
2. Forecasts are generated for horizons $h \in \{1, 2, 4\}$.
3. The actual observed values at $t, t+1, t+3$ are recorded for evaluation.
4. The window expands by one quarter, and the process repeats.

## Model Registry
12 distinct models are evaluated:
- **Deterministic Baselines**: Persistence, ARIMA, VAR, ETS, Dynamic Factor, Ridge Regression, Gradient Boosting.
- **Graph-Free Neural Baselines**: Multi-Layer Perceptron (MLP), Long Short-Term Memory (LSTM), Temporal Convolutional Network (TCN).
- **Graph Neural Networks**: Graph Convolutional Network (GCN), Temporal Graph Network.

## Stochastic Evaluation Strategy
To account for initialization variance in neural network architectures, all stochastic models (neural networks and gradient boosting) are trained and evaluated across 20 distinct random seeds (42 to 61). Aggregated performance metrics and significance tests are computed using the ensemble distribution of these runs.

## Leakage Prevention
The benchmark engine strictly enforces temporal boundaries. The data loader for step $t$ only exposes features up to $t-1$. Cross-sectional scaling is computed dynamically at each step using only the available training history.

## Transactional Execution
To manage the computational burden of 38,380 model fits, the benchmark engine uses a transactional storage layer. Intermediate forecasts and metrics are continuously flushed to disk in Parquet format. If execution is interrupted, it can be resumed without recomputing completed windows.
