# Research Protocol: Quarterly Inflation Forecasting with Trade Graphs

**Version:** 2.0  
**Status:** Rebuild in progress — no v2 results yet  
**Controlling plan:** [`ijf_rebuild_plan.md`](ijf_rebuild_plan.md)  
**Machine-readable specification:** [`../config/protocol_v2.yaml`](../config/protocol_v2.yaml)

## Scope and legacy boundary

This protocol supersedes the v1 study for all submission-facing work. The v1 data, results, manuscript, and claims are retained only as a reproducibility record and are **legacy / not for submission**. In particular, annual CEPII BACI values must not be apportioned across quarters or described as dynamic quarterly trade networks, and annual World Bank variables must not be repeated as quarterly node features.

## Forecasting objective

Forecast country-level CPI inflation at one-, two-, and four-quarter horizons. CPI inflation is the annual percentage change in the quarterly mean of a native monthly CPI index (or a native quarterly CPI series where its release schedule is documented).

A forecast origin at the end of quarter `t` uses only release-valid information available then. The primary trade graph is built from the newest fully available monthly bilateral-trade quarter no later than `t-1`; that lag must be applied consistently to every graph model and graph ablation.

This is a pseudo-real-time design using revised observations with explicit availability lags unless historical vintages are supplied. It is not a real-time-vintage study otherwise.

## Admissible inputs

| Component | Requirement |
|---|---|
| Trade edges | Monthly bilateral merchandise trade from IMF IMTS/DOTS or UN Comtrade, aggregated to quarter |
| CPI target | Native monthly or quarterly CPI with documented release timing |
| Required node features | `cpi_yoy`, `energy_idx` |
| Optional node features | Policy rate, exchange rate, industrial production, or GDP nowcast only where native frequency and release lag are documented |
| Acquisition record | Each download has source URL/API query, retrieval timestamp, source release/vintage date, file hash, and license/access note |

Countries are selected after data validation: retain only those with complete, release-valid CPI and trade inputs over the modeled sample. No target panel size is privileged over valid coverage.

## Input contract

Place source observations—not derived variables—in the v2 raw package:

- `macro/macro_observations.csv`: long-form monthly/quarterly macro observations and provenance
- `trade/trade_observations.csv`: directed monthly/quarterly bilateral trade observations and provenance
- `metadata/source_registry.csv`: source, release-lag, coverage, and license metadata
- `raw_manifest.json`: non-empty acquisition records, hashes, official URLs, release dates, and retrieval timestamps

The machine-readable schemas live under `data/raw/v2/metadata/`; see [`data_acquisition_v2.md`](data_acquisition_v2.md) for the complete contract and acquisition instructions. The raw layer accepts only observed monthly/quarterly values. It does not contain CPI inflation, energy changes, quarterly aggregation, imputation, or graph weights. `source_release_date` must be parseable for every row. The acquisition manifest is an input and is never overwritten; validation writes a separate `validation_manifest.json` with input hashes.

Run the hard gate before any v2 data construction:

```bash
python -m src.validate_v2_inputs
```

## Pre-specified evaluation

The final test period must contain at least 32 forecast origins where coverage permits. All hyperparameters, transformations, and model choices are selected using training and validation data only before final test evaluation.

The required model comparison includes:

1. Persistence/random walk.
2. ETS and auto-ARIMA or transparently tuned ARIMA/ARX.
3. Dynamic-factor and regularized panel/global models.
4. A graph-free neural model using identical node features.
5. Lagged directed and undirected graph models, plus static, shuffled-edge, and zero-edge variants.

Report RMSE, MAE, CRPS or log score, interval coverage, and interval width at every horizon. Forecast-origin blocks—not country-quarter observations—are the independent time dimension for inferential comparisons. Use a moving-block bootstrap or equivalent method that preserves cross-country dependence, report confidence intervals for loss differentials, and pre-specify any equivalence margin.

## Interpretation boundary

Integrated gradients are sensitivity diagnostics, not evidence of causal transmission or contagion. Any partner-importance interpretation requires stability across seeds, comparison with trade strength and centrality, shuffled-graph null checks, and edge-deletion counterfactuals. Causal policy claims require a separate identification or structural-validation design.

## Submission package requirements

The eventual IJF package must include a blinded manuscript and separate title page, a public repository with acquisition instructions, hashes, environment lockfile, seeds and hardware/software information, a one-command reproduction path, and an online appendix covering all countries, horizons, settings, forecasts, and failed or negative checks.
