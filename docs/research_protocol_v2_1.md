# Research Protocol v2.1: Journal-Ready Trade-Network Inflation Forecasting

**Status:** v2.1 forecasts fully generated and SHA256-validated; results archived under `experiments/results/v2_1/`.  
**Supersedes for submission:** v2.0 experiment contract only.  
**Preserves:** v2 raw/processed artifacts, configuration, failed manifests, checkpoints, and all prior commits.

## 1. Scope and scientific boundary

The estimand is incremental **predictive** value: whether lagged, observed bilateral trade-network structure improves country-level CPI inflation forecasts beyond pre-specified non-graph comparators. This protocol does not identify causal transmission, contagion, policy effects, or structural spillovers.

The v2.0 exploratory/partial records remain archived and are not evidence for a v2.1 conclusion. V2.1 uses the existing validated 20-country European panel, revised-source pseudo-real-time design, and fixed one-quarter macro and trade availability lags. It is not a historical-vintage real-time study.

## 2. Locked data, timing, and splits

- **Raw and processed inputs:** the committed `data/raw/v2/` and `data/processed/v2/` artifacts only, gated by their committed validation hashes.
- **Countries:** the registered 20-country node set in `data/processed/v2/countries.json`.
- **Target:** country CPI annual inflation (`cpi_yoy`) at 1-, 2-, and 4-quarter horizons.
- **Information set at origin t:** only inputs with `macro_feature_quarter < t` and `trade_graph_quarter < t`; labels satisfy `target_quarter > t`.
- **Training origins:** 2011Q2–2014Q4.
- **Validation origins:** 2015Q1–2016Q4.
- **Final test origins:** 2017Q1–2025Q3, yielding 35, 34, and 32 origins for horizons 1, 2, and 4 respectively.

No transformation, country inclusion choice, graph variant, hyperparameter, or comparator may be selected using final-test losses.

## 3. Pre-specified model registry

Every v2.1 execution must include every configured member of the following registry:

1. Persistence/random-walk reference;
2. Country ARIMA;
3. Panel VAR;
4. Exponential smoothing (ETS);
5. Dynamic factor model (DFM);
6. Ridge regression;
7. Histogram gradient boosting;
8. Graph-free MLP, LSTM, and TCN;
9. Static GCN and temporal graph forecaster under every locked graph variant.

The non-graph neural models must use the same release-valid CPI/energy information available to their graph counterparts. The graph registry is fixed as `directed_trade`, `log_trade`, `import_dependence`, `top_k_incoming`, `reversed`, `undirected`, `degree_preserving_random`, and `identity_no_trade`.

## 4. Hyperparameter selection and tuning isolation

All candidate choices are selected exclusively using training and validation origins. The final test split is inaccessible to tuning code.

- Fit candidates only on expanding samples whose targets are available at the validation origin.
- Select one configuration per model family/horizon using mean validation-origin MAE; ties are resolved by lower validation RMSE, then the simpler documented candidate.
- Record the candidate grid, validation losses, selected configuration, seed list, and selection timestamp in an immutable v2.1 tuning manifest before final-test execution.
- The selected model/horizon settings and graph registry are then frozen in the v2.1 configuration. A configuration hash identifies the execution.
- No re-tuning, early-stopping choice, calibration choice, or comparator selection may inspect test outcomes.

## 5. Forecast outputs and deterministic evaluation

Every forecast must carry run, Git, dataset, configuration, environment, model, graph, seed, horizon, origin, country, target-quarter, prediction, actual, and training-extent provenance.

At each horizon, compute country-averaged origin losses and report:

- RMSE;
- MAE;
- sMAPE, defined as `200 * mean(|y - yhat| / (|y| + |yhat|))`, omitting only zero denominators and reporting their count.

Country-quarter rows are retained for transparency, but forecast origins are the independent time dimension for aggregate uncertainty and testing.

## 6. Probabilistic evaluation

Each model must emit a predictive distribution or a documented, validation-calibrated interval procedure. For each horizon, report:

- CRPS (or an equivalent proper score only if formally documented before execution);
- nominal 80% and 95% prediction-interval coverage;
- mean interval width at each nominal level.

Distributional/calibration parameters are fit on training/validation information only. Point-only models are not eligible for a submission-grade v2.1 benchmark until a locked probabilistic forecast construction is implemented.

## 7. Statistical inference

The primary loss is origin-level MAE: for each origin, average absolute loss across countries, preserving contemporaneous cross-country dependence. Secondary inference repeats the comparison for squared error and sMAPE only as explicitly labelled secondary analyses.

For each graph model/variant, compare against each pre-specified non-graph comparator in the configuration: ARIMA, ETS, DFM, ridge, gradient boosting, LSTM, and TCN. Comparisons use their common country-origin support.

- Apply the Harvey–Leybourne–Newbold small-sample-adjusted Diebold–Mariano statistic with a Bartlett HAC variance and maximum lag `horizon - 1`.
- Report the signed mean loss differential (`graph - comparator`), two-sided p-value, and a moving-block-bootstrap 95% confidence interval over forecast origins (block length 4).
- Correct the complete family of primary graph-versus-comparator/horizon tests using Benjamini–Hochberg false-discovery-rate control at `q = 0.05`, with the monotone step-up adjusted p-values.
- A graph-model improvement may be described only when its signed loss differential favors the graph model, its confidence interval excludes zero, and its BH-adjusted primary-test p-value is below 0.05.

## 8. Execution integrity and retention

A submission run accepts no model subset, seed subset, horizon subset, or graph subset. It writes to a unique transaction directory under `experiments/results/v2_1/staging/`. Canonical `forecasts.parquet`, checkpoints, metrics, and tests are published only after the complete registry and schema are validated. A failure retains its transaction directory and an append-only failure record; it never overwrites prior manifests or canonical output.

## 9. Reporting and limitations

Report all models, graph variants, seeds, countries, horizons, intervals, failed transactions, and negative findings. Interpret graph results as predictive associations only. Discuss revised-data pseudo-real-time limitations, regional scope, short early training windows, and the absence of causal identification.
