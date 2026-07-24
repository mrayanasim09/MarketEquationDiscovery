Methods Appendix

Model architectures
- ST-GNN: graph convolution layers (GCN) followed by LSTM; input window: 4 quarters; output: country-level incremental CPI change.
- Baselines: ARIMA(1,0,0), VAR(1), ETS, ridge regression, gradient boosting (sklearn HistGradientBoosting), MLP, TCN.

Training
- Optimizer: Adam, learning rate 0.01, weight decay (L2) as reported in config.
- Loss: masked Huber for point forecasts; CRPS for probabilistic models where applicable.
- Early stopping based on validation MAE with patience of 5 epochs.

Evaluation
- Rolling expanding-window refit with origins aligned to quarterly periods.
- Metrics: RMSE, MAE, sMAPE for point forecasts; CRPS and empirical interval coverage for probabilistic performance.
- Statistical tests: Diebold--Mariano for pairwise predictive accuracy; BH-adjusted p-values for multiple-testing.

Hyperparameters and configurations
- See `experiments/configs/v2_benchmark.json` and `manuscript/submission_docs/reproducibility_README.md` for scripts used to run experiments.

Additional details
- Data preprocessing steps include CPI seasonality checks, missing value imputation using forward-fill within countries, and log-transform where specified.
- Graph variants: directed trade, log-trade, import dependence, degree-preserving randomization, top-k incoming, undirected, identity/no-trade.
Methods Appendix

Model specifications
- ST-GNN architecture: graph convolution layers followed by an LSTM over a 4-quarter input window. Hidden dim = 16, epochs = 30.
- Baselines: ARIMA(1,0,0), VAR(1), ETS, Ridge regression, Gradient boosting (scikit-learn), MLP (2-layer), TCN.

Evaluation protocol
- Rolling expanding-window refit with origins from 2016Q1 to 2024Q1.
- Forecast horizons: 1, 2, 4 quarters.
- Metrics: RMSE, MAE, sMAPE for point forecasts; CRPS and empirical interval coverage for probabilistic forecasts.
- Statistical tests: Diebold--Mariano for pairwise comparisons; BH-adjusted p-values for multiple comparisons; TOST for equivalence.

Hyperparameters
- See `experiments/configs/v2_benchmark.json` for full optimization grids.

Implementation notes
- Code organization: `src/` contains training, evaluation, and explainability scripts. `experiments/` stores configurations and results.
- Reproducible notebook: `paper/generated/analysis.ipynb` demonstrates key plots and IG explainability examples.

Limitations
- Graph definitions are based on trade aggregates and may omit input-output relationships.
- Calibration methods were not exhaustively explored (future work).