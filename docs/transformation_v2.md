# V2 Transformation and Forecasting-Dataset Design

**Status:** Processed v2 dataset generated and validated; no model has been trained.  
**Input boundary:** validated files under `data/raw/v2/` only.  
**Output boundary:** `data/processed/v2/` only.  
**Legacy boundary:** v1 data and results are not read or modified by this pipeline.

## Pipeline

```text
validated monthly HICP / Comext raw observations
  -> complete-quarter aggregation
  -> quarterly inflation features and directed graph snapshots
  -> lagged rolling-origin forecast samples
```

Each transform starts by verifying that raw macro, trade, and source-registry hashes still match `data/raw/v2/validation_manifest.json`. A change to any raw input requires raw validation to be rerun before processing.

## Quarterly macro aggregation

For country `i` and quarter `q`, each HICP index is the arithmetic mean of the three native monthly index values:

```text
HICP(i, q) = mean(HICP(i, m) for the three months m in q)
```

A quarter is emitted only when all three monthly observations are present. Missing months are never interpolated, carried forward, or otherwise replaced.

The two raw HICP components remain distinct:

- `HICP_CP00_INDEX`: all-items HICP;
- `HICP_CP045_ENERGY_INDEX`: electricity, gas, and other fuels.

Quarterly annual inflation features are then calculated only when both complete quarterly averages exist exactly four quarters apart:

```text
cpi_yoy(i, q) = 100 * (HICP_CP00(i, q) / HICP_CP00(i, q - 4) - 1)
energy_cpi_yoy(i, q) = 100 * (HICP_CP045(i, q) / HICP_CP045(i, q - 4) - 1)
```

The CP045 definition must be retained downstream; it is not relabeled as a generic global energy-price index.

## Directed trade graphs

The graph node set is the verified macro–trade intersection of 20 countries in `countries.json`. For directed edge `i -> j` in quarter `q`:

```text
trade_eur(i, j, q) = sum(monthly Comext export VALUE_IN_EUROS from i to j)
```

The sum is calculated only when all three monthly directed exports are observed. An incomplete pair-quarter is not imputed as zero.

Outputs:

- `quarterly_trade_edges_directed.csv`: complete and incomplete source pair-quarter records, monthly-count and provenance fields included;
- `adjacency_directed_trade_eur.npy`: raw directed quarterly euro weights;
- `adjacency_directed_observed_mask.npy`: boolean indicator that distinguishes observed edges from unobserved pair-quarters;
- `graph_manifest.json`: node order, graph definition, and missing-edge counts.

`trade_log1p` is included in the edge table as a transparent alternative scale. No adjacency normalization occurs in the transformation layer. Directed/undirected conversion, self-loops, and normalization are pre-specified model-variant choices for the later benchmark/ablation milestone.

## Forecast timing and leakage prevention

This is a pseudo-real-time reference-period-lag design using revised source snapshots, not a historical-vintage study.

At forecast origin `t` for target horizon `h`:

- target: `cpi_yoy(i, t + h)`;
- macro input: `cpi_yoy(i, t - 1)` and `energy_cpi_yoy(i, t - 1)`;
- graph input: directed trade snapshot for `t - 1`.

The one-quarter macro lag is conservative: it prevents same-quarter macro data from entering an end-of-quarter forecast origin. The one-quarter trade lag implements the v2 protocol’s availability rule. `forecast_samples.csv` records every origin, input-quarter, graph-quarter, target-quarter, and split explicitly.

## Rolling-origin splits

No random split is used. All split boundaries are fixed before model evaluation:

| Split | Origins |
|---|---|
| Train-design origins | 2011Q2–2014Q4 |
| Validation origins | 2015Q1–2016Q4 |
| Test origins | 2017Q1 onward, subject to horizon-specific target availability |

At every origin, a fit may use only preceding origins in an expanding window; `expanding_train_end` is persisted in `forecast_origins.csv`.

Generated origin counts are:

| Horizon | Train | Validation | Test |
|---|---:|---:|---:|
| 1 quarter | 15 | 8 | 35 |
| 2 quarters | 15 | 8 | 34 |
| 4 quarters | 15 | 8 | 32 |

Thus each horizon meets the protocol minimum of 32 test origins. The 4-quarter test window is the binding evaluation design.

## Processed outputs

```text
data/processed/v2/
├── countries.json
├── quarterly_hicp_panel.csv
├── quarterly_feature_panel.csv
├── quarterly_trade_edges_directed.csv
├── adjacency_directed_trade_eur.npy
├── adjacency_directed_observed_mask.npy
├── quarters.json
├── graph_manifest.json
├── forecast_origins.csv
├── forecast_samples.csv
├── split_report.json
└── transformation_validation.json
```

## Validation

Run after all transforms:

```bash
.venv/bin/python -m src.transform.validate
```

The gate verifies raw-hash lineage, country-order consistency, complete country coverage at each forecast origin, strict macro and graph lags, future-only targets, existing graph references, and at least 32 test origins per required horizon.

## Limitations to retain in later analysis

- The panel is regional: 20 European economies, not a global system.
- Eurostat responses are archived current-release snapshots; they do not recreate historical data vintages.
- The graph is directed exports in current euros; it is not a measure of imports, total bilateral trade, or causal exposure.
- The final-test requirement is met, but the 2010–2025 history leaves a relatively compact pre-test training/validation span. Model tuning must remain strictly confined to the eight validation origins and all model comparisons must retain the common origin structure.
