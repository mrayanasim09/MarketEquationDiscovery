# Trade-Network Forecasting Robustness and Diagnosis (v2)

**Status:** Submission-grade diagnosis is blocked by the current benchmark implementation and result artifacts.  
**Conclusion:** **No A/B/C conclusion is warranted yet.** The existing one-seed negative result is a functional smoke-test result, not evidence that trade networks provide little predictive value.

## Research question

Does lagged trade-network information improve inflation forecasting beyond macro-only and classical non-graph baselines in the validated 20-country European panel?

The question is predictive. No graph result is interpreted as causal transmission, contagion, or policy evidence.

## Preflight finding

`src.models.robustness_preflight` fails closed against the current artifacts for four reasons:

1. The locked experiment configuration specifies 20 neural seeds, but the current run contains only seed 42.
2. `v2_forecasts.csv` has no `seed` field, so results cannot be averaged across seeds and seed-level DM tests cannot be computed.
3. The functional run used 10 epochs, while the locked configuration specifies 30 epochs.
4. Required stronger baselines—Bayesian VAR, temporal convolution network, and gradient boosting—are not present.

The full error record is written to `experiments/results/v2_robustness_preflight.json`.

## Why the current graph result is not yet diagnostic

The one-seed exploratory run finds that directed graph models do not outperform the best non-graph models. It also finds that zero-edge or shuffled-edge ablations can be as good as or better than directed graphs. This is a useful warning, but it does not isolate the cause because the current implementation has unresolved representation limitations:

- The LSTM path repeats the current macro vector rather than consuming a country inflation-history sequence.
- The temporal graph path does not yet consume a sequence of graph snapshots and node features; it is not a temporal graph model in the required sense.
- The graph estimator trains on only the latest eligible panel rather than the full history of eligible origin panels.
- The current graph variants do not yet implement import-dependence, top-k sparse, or degree-preserving random graphs.
- The evaluation outputs do not retain per-seed forecasts.

These are implementation/design limitations, not evidence about the economic content of trade networks.

## Exploratory result retained as a smoke-test finding

The completed single-seed run is retained in `docs/model_results_v2.md` and `experiments/results/` as a reproducibility record. It must be labeled exploratory.

| Horizon | Directed-edge model | Best non-graph model | Direction of exploratory MAE difference |
|---|---|---|---|
| 1 | GCN | ARIMA | Directed graph worse |
| 2 | GCN | ARIMA | Directed graph worse, inconclusive DM |
| 4 | GCN | LSTM | Directed graph worse |

This is not an A/B/C diagnosis because it cannot distinguish inadequate neural robustness or graph representation from limited predictive information in the network.

## Required redesign before a robustness run

The benchmark runner must be revised before spending compute on 20 seeds:

1. **Seed-safe outputs** — store a `seed` column for every neural/graph forecast; deterministic baselines may be replicated or recorded with `seed=deterministic`.
2. **Locked training budget** — use the configured 20 seeds and 30 epochs, or revise the configuration before execution using validation-only evidence.
3. **Actual temporal inputs** — LSTM and TCN must use historical country feature sequences ending at the allowed `t-1` macro quarter. The temporal graph model must use a pre-origin sequence of lagged feature/graph snapshots.
4. **Full eligible training history** — each graph model must train on all horizon-specific samples whose targets were available by the outer origin, not only the most recent eligible panel.
5. **Alternative graph formulations** — add, without changing raw data: import-dependence weights; `log1p` trade; top-k incoming exposures; reversed directed graph; degree-preserving random graph; and undirected graph. Importer-GDP normalization must remain absent unless a release-valid GDP input is separately acquired and validated.
6. **Feature diagnostics** — add only lagged, origin-safe CPI volatility and trade-exposure summaries; do not alter the processed dataset or add unavailable macro series.
7. **Stronger baselines** — add Bayesian/shrinkage VAR if feasible, TCN, and gradient boosting with all preprocessing fit inside the outer training window.
8. **Inference** — compute seed-specific origin-aggregated loss differentials, moving-block bootstrap confidence intervals over origins, HAC/HLN DM tests per seed, and a pre-specified across-seed aggregation. Adjust or clearly qualify multiple comparison results.
9. **Interpretability only after predictive performance** — run edge deletion and permutation tests across seeds; compare against shuffled graphs; report sensitivity only, never causal importance.

## Current diagnosis

**Unresolved.** The present evidence is compatible with each possibility below:

- insufficient neural robustness;
- inadequate graph representation;
- insufficient but potentially augmentable feature space;
- genuinely limited predictive value of the directed export network.

The correct action is to repair and validate the experiment runner, then run the locked 20-seed diagnosis. It would be scientifically invalid to choose conclusion A, B, or C now.

## Data and scope limitations that remain fixed

- The panel is 20 European economies, not a global network.
- Trade is directed Comext total-goods exports in current euros.
- HICP CP045 is electricity, gas, and other fuels; it is not a global energy-price series.
- The raw inputs are revised current-release snapshots, not historical data vintages.
- Forecast origins remain fixed and must not be changed for robustness work.
