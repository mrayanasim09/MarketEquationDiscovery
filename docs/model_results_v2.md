# V2 Benchmark Evaluation Results

**Status:** Exploratory benchmark run complete; this is **not submission-grade final evidence** because neural/graph models have been run for one seed only. No manuscript claim should be generated from these results until the configured 20-seed sweep and validation-only tuning procedure are completed.

## Hypothesis

The test is predictive, not causal: whether lagged directed Eurostat Comext export-network information improves CPI-inflation forecasts beyond macro-only and conventional time-series baselines for the validated 20-country European panel.

No result is interpreted as contagion, a transmission mechanism, or causal shock propagation.

## Data and evaluation contract

- Data: `data/processed/v2/`, derived from the validated 2010Q1–2025Q4 Eurostat HICP/Comext raw package.
- Countries: 20 aligned European economies.
- Targets: country-level `cpi_yoy` at 1-, 2-, and 4-quarter horizons.
- Inputs at forecast origin `t`: macro features and directed trade graph from `t-1`.
- Training cutoff: for horizon `h`, only samples with `target_quarter <= t` are used at evaluation origin `t`.
- Evaluation: fixed rolling-origin test split, with 35, 34, and 32 origins at 1, 2, and 4 quarters respectively.
- Loss and inference unit: countries are averaged within origin; origin-level losses are used for metrics and DM comparisons.
- Run: seed `42`, 10 neural epochs, fixed predeclared settings recorded in `experiments/configs/v2_benchmark.json` and `experiments/results/v2_run_metadata.json`.

This is pseudo-out-of-sample evaluation on revised source snapshots, not historical-vintage real-time evaluation.

## Models evaluated

| Family | Implemented model |
|---|---|
| Persistence | Last available lagged CPI inflation |
| Classical | Country ARIMA(1,0,0); country bivariate VAR(1) on CPI and energy CPI |
| Macro-only | Global ridge; MLP; LSTM baseline |
| Directed graph | Directed GCN; directed temporal graph-convolution variant |
| Graph ablations | Zero edge, shuffled edge, fully connected, undirected, and no-energy variants |

The zero-edge and fully connected variants are ablations, not evidence of trade-network value.

## Test metrics (selected models)

Values are origin-averaged test metrics. Full results—including RMSE, MAE, MASE, directional accuracy, origin standard deviations, and approximate normal CIs—are in `experiments/results/v2_metrics.csv`.

| Horizon | Best non-graph by MAE | MAE | Directed GCN MAE | Directed temporal graph MAE |
|---|---|---:|---:|---:|
| 1 quarter | ARIMA | 1.751 | 2.130 | 2.141 |
| 2 quarters | ARIMA | 2.433 | 2.662 | 2.922 |
| 4 quarters | LSTM | 3.053 | 3.972 | 4.917 |

In this run, none of the actual directed-edge graph models improves on the best non-graph comparator.

## Trade-information ablations

The ablations do not support a meaningful incremental value from the supplied directed trade structure:

- At one quarter, the **zero-edge** temporal graph variant has lower MAE (1.647) than directed graph variants.
- At two quarters, the **shuffled-edge** variant (MAE 2.594) outperforms the directed temporal graph variant (2.922).
- At four quarters, the no-energy and several non-directed/ablated variants are lower-error than the directed temporal graph variant.

These are diagnostic comparisons in a single-seed run; they do not identify mechanisms or prove that trade is irrelevant in all specifications.

## Diebold–Mariano comparisons

The DM comparison is restricted to actual directed-edge models (`gcn_directed` or `tgcn_directed`) against the best non-graph model at each horizon. The differential is origin-level MAE loss: positive values favor the non-graph model. HAC bandwidth is at least `h-1`.

| Horizon | Directed trade model | Non-trade comparator | Origins | Trade − non-trade MAE | DM statistic | p-value |
|---|---|---|---:|---:|---:|---:|
| 1 | Directed GCN | ARIMA | 35 | 0.379 | 2.066 | 0.047 |
| 2 | Directed GCN | ARIMA | 34 | 0.229 | 0.599 | 0.553 |
| 4 | Directed GCN | LSTM | 32 | 0.919 | 2.311 | 0.028 |

The one- and four-quarter tests reject equal loss at conventional levels **against** the directed GCN. The two-quarter comparison is inconclusive. These unadjusted p-values are descriptive because model/horizon comparisons are multiple and neural uncertainty has not been aggregated across the configured seed set.

## Non-causal structural importance

`experiments/results/v2_graph_importance.csv` reports descriptive mean graph strength only. The largest mean outgoing Comext exporters are Germany, Netherlands, Belgium, France, and Italy. This is a graph descriptive, not attention attribution, partner importance, shock propagation evidence, or causal result.

## Reproducibility outputs

```text
experiments/
├── configs/v2_benchmark.json
└── results/
    ├── v2_forecasts.csv
    ├── v2_metrics.csv
    ├── v2_dm_tests.csv
    ├── v2_graph_importance.csv
    └── v2_run_metadata.json
```

Run the exploratory evaluation with:

```bash
.venv/bin/python -m src.models.run_benchmarks --neural-seeds 1 --epochs 10
```

The configured submission-grade seed list contains 20 seeds. Before any paper-facing interpretation, run the full list with the locked epoch budget (or a validation-only tuning protocol), aggregate neural forecasts across seeds, and repeat metrics, ablations, bootstrap intervals, and DM comparisons on the common origin support.

## Limitations

1. One neural seed and ten epochs are a functional benchmark run, not a stable neural-model estimate.
2. ARIMA emits occasional convergence/start-parameter warnings; fallback events are not yet separately counted and must be logged before final reporting.
3. The LSTM baseline is deliberately compact and must receive the same validation-only architecture/epoch selection as graph models in a full sweep.
4. The panel is regional, trade is directed exports in current euros, and raw source snapshots are revised rather than historical vintages.
5. The raw period is sufficient for the required 32 four-quarter test origins but leaves a compact training/validation span.

## Interim conclusion

The present exploratory evidence is a negative result for incremental directed-trade-graph forecasting value. The appropriate next action is not a causal or spillover interpretation; it is a locked 20-seed robustness run with full origin-level uncertainty reporting. No paper has been drafted.
