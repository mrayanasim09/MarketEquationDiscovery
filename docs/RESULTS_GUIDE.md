# Results Guide

## Interpreting Output Artifacts
The benchmark engine outputs results in highly compressed and structured Parquet format to manage the extensive volume of data (over 781,740 forecast rows).

## Directory Structure
```
results/
├── forecasts/       # Raw point and probabilistic forecasts
├── metrics/         # Aggregated error metrics
└── statistics/      # Diebold-Mariano significance tests
```

## Schema Definitions

### Forecast Columns (`results/forecasts/*.parquet`)
- `timestamp`: The target prediction date (Quarter).
- `country`: ISO-3 country code (e.g., DEU, FRA).
- `horizon`: Forecast horizon ($h \in \{1, 2, 4\}$).
- `model_name`: Identifier from the model registry.
- `graph_variant`: The adjacency matrix formulation used (if applicable).
- `seed`: Random seed used for initialization (42-61).
- `y_true`: Observed inflation rate.
- `y_pred`: Point forecast.
- `y_pred_lower_95`, `y_pred_upper_95`: 95% predictive interval bounds.

### Metric Columns (`results/metrics/*.parquet`)
- `model_name`, `graph_variant`, `horizon`, `seed`: Grouping keys.
- `MAE`: Mean Absolute Error.
- `RMSE`: Root Mean Squared Error.
- `sMAPE`: Symmetric Mean Absolute Percentage Error.
- `CRPS`: Continuous Ranked Probability Score (for probabilistic evaluation).
- `interval_coverage`: Empirical coverage of the 95% interval.
- `interval_width`: Average width of the 95% interval.

### Diebold-Mariano Columns (`results/statistics/*.parquet`)
- `baseline_model`: The reference model for the test.
- `competitor_model`: The model being evaluated against the baseline.
- `horizon`: Forecast horizon.
- `dm_stat`: Harvey-Leybourne-Newbold modified DM test statistic.
- `p_value_raw`: Uncorrected p-value.
- `p_value_fdr`: Benjamini-Hochberg corrected p-value.
- `significant_05`: Boolean flag indicating significance at $\alpha = 0.05$.

## Analysis using Pandas
```python
import pandas as pd
metrics_df = pd.read_parquet('results/metrics/summary_metrics.parquet')
best_models = metrics_df.groupby(['horizon', 'model_name'])['MAE'].mean().unstack()
```
