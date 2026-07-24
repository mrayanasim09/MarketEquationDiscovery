# Benchmark Engine v2

## Status and scope

This document specifies the reconstructed evaluation engine for the v2 study. It
is an isolated replacement for the exploratory benchmark runner; it does not
modify the validated raw layer, processed forecasting samples, rolling-origin
splits, horizons, or timing rules. No submission-grade experiment has been run
with this engine yet.

The implementation is in `src/models/run_benchmark_engine_v2.py`. It is
protected by an explicit `--execute` flag, and is to be invoked only after the
pre-training validation gate passes.

## Reproducible result storage

All new output is append-only under `experiments/results/v2/`:

- `forecasts.parquet`: one row per prediction;
- `metrics.parquet`: post-run seed-level scoring estimates;
- `dm_tests.parquet`: per-seed HAC/HLN Diebold--Mariano comparisons;
- `configs/benchmark_engine.json`: locked engine configuration;
- `checkpoints/`: network parameters keyed by run identifier;
- `metadata/validation.json`: benchmark-engine preflight report;
- `run_manifest.json`: append-only run records.

Each forecast has `model_name`, `model_variant`, `seed`, `horizon`,
`forecast_origin`, `country`, `target_quarter`, `prediction`, `actual`,
`split`, its training sample count and extent, feature set, and graph type.
The storage layer rejects duplicate forecast keys; a previous seed can never be
silently overwritten.

Every run record contains Python, NumPy, and Torch seed values, model and
optimizer settings, epochs, sequence length, horizon, forecast-origin cutoff,
feature and graph choices, training extent, and runtime versions.

## Origin-safe training protocol

For each test forecast origin `t` and horizon `h`, the engine selects training
samples with:

```text
target_quarter <= t
```

This is stricter than simply selecting rows by split label. Neural and graph
models are initialized and fitted from scratch at every outer origin and seed.
Feature scalers are fit only to that origin's eligible training observations.
The final stored prediction is for the fixed test sample at that origin, never
for a random holdout.

The engine uses the existing one-quarter macro and trade availability lags in
`forecast_samples.csv`; it does not manufacture new timing fields. It does not
write to `data/processed/v2/`.

## Models

The locked model set contains:

- persistence (`cpi_yoy` at the persisted input quarter);
- country ARIMA(1,0,0), with a persistence fallback when a fit fails;
- panel VAR(1);
- ridge regression;
- histogram gradient boosting;
- a separately labelled ridge-prior Bayesian/shrinkage VAR approximation;
- MLP;
- LSTM;
- temporal convolution network (TCN);
- static graph convolution forecaster (GCN);
- temporal graph forecaster.

The Bayesian/shrinkage implementation is not described as an ordinary VAR.
The gradient-boosting baseline requires `scikit-learn`; Parquet result storage
requires `pyarrow`.

### Sequences

LSTM and TCN use four ordered, lagged quarterly observations:

```text
[cpi_yoy, energy_cpi_yoy] at t-4, t-3, t-2, t-1
```

where `t-1` is the persisted allowed macro input quarter. The historical values
are fetched directly from the validated quarterly feature panel. Repeating a
single current row as a surrogate sequence is prohibited.

The temporal graph model receives matching histories:

```text
X(t-4) ... X(t-1)
G(t-4) ... G(t-1)
```

It graph-convolves each node feature snapshot with its matching directed trade
snapshot and uses an LSTM across those representations. It never receives a
graph at or after the forecast origin.

## Features

The non-neural tabular feature set comprises the most recent permitted CPI and
energy inflation, four-quarter CPI volatility, and lagged in/out trade exposure
sums. The sequence feature set contains CPI and energy CPI histories. All
feature construction is in memory (`src/models/features.py`) and rejects
incomplete histories; it does not interpolate or impute values. Early eligible
labels that do not have four published quarterly CPI observations are excluded
only from models requiring that history, and their reduced training extent is
persisted with each forecast. Static GCNs use the persisted latest permitted
CPI and energy inputs and therefore do not require the four-quarter sequence.

## Graph variants and ablations

The graph factory only transforms persisted Comext adjacency snapshots. Every
variant retains the fixed country node ordering and existing quarter indexing:

1. `directed_trade`: exporter to importer trade values;
2. `log_trade`: `log(1 + trade)`;
3. `import_dependence`: trade divided by importer total imports;
4. `top_k_incoming`: five strongest observed incoming edges per importer;
5. `reversed`: edge orientation reversed;
6. `undirected`: bilateral sum;
7. `degree_preserving_random`: weights shuffled over the observed directed support;
8. `identity_no_trade`: self-only message passing, with no cross-country trade edge.

Except for the self-only ablation, outgoing row normalization is applied after
the named graph transformation. GDP-normalized trade exposure is not included:
no separately release-valid GDP panel has been acquired or validated.

## Pre-training validation

Run, before any model fit:

```bash
.venv/bin/python -m src.models.validate_experiment
```

The validator checks the locked seed and epoch counts, sample uniqueness,
strictly lagged macro and graph timestamps, target timing, non-missing labels,
graph-quarter coverage, and graph tensor shape. If result files later exist, it
also rejects duplicate result keys and incomplete neural seed coverage.

## Post-run inference

Only after all configured forecasts exist, run:

```bash
.venv/bin/python -m src.models.validate_experiment
.venv/bin/python -m src.models.evaluate_benchmark_engine_v2 --execute
```

The evaluator groups losses by forecast origin before block bootstrapping, thus
retaining cross-country dependence. It reports seed-level RMSE, MAE, MASE, and
directional accuracy; aggregates seed estimates with mean, standard deviation,
and normal 95% intervals; and computes per-seed HAC/HLN Diebold--Mariano tests.
Benjamini--Hochberg adjusted p-values are written to the companion adjusted DM
table. Deterministic baselines remain explicitly `seed=deterministic`; they are
not replicated to create artificial seed variation.

## Known limitations

- Data are revised official releases rather than real-time vintage snapshots.
- The panel is regional (20 European countries), not global.
- Pre-test historical training windows are compact at early rolling origins.
- The graph is trade-value based and is not GDP-normalized.
- The engine evaluates predictive association only. It cannot support causal,
  contagion, transmission, or policy claims.
