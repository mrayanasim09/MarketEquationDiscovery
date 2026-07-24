# IJF Rebuild Plan (v2)

## Purpose

This document replaces the v1 design.  It exists because v1 expanded annual CEPII
BACI trade flows to quarters and repeated annual World Bank controls within each
quarter.  Those transformations are not valid evidence for a dynamic quarterly
trade-graph forecasting study.

No result, table, figure, or claim from v1 may be used in a v2 manuscript unless
it is regenerated from the v2 pipeline.

## Step 1 — Freeze and label the legacy study

- Retain `dataset_v1` and all current results as a reproducibility record.
- Label them **legacy / not for submission** in the README.
- Do not call BACI-derived within-year graphs dynamic quarterly networks.

## Step 2 — Construct a defensible information set

- Target: next-quarter CPI inflation, computed from monthly CPI indices.
- Trade graph: monthly bilateral trade from IMF IMTS (formerly DOTS) or UN
  Comtrade, aggregated to quarter.
- Availability rule: a forecast made at the end of quarter `t` may use the latest
  fully available trade quarter no later than `t-1`.  The release/vintage date of
  every raw download must be stored.
- Node variables: use only series that are monthly or quarterly and whose release
  lag is documented.  Do not upsample annual observations and present them as
  quarterly inputs.
- Exclude a country if a complete, release-valid CPI and trade series cannot be
  supplied.  Coverage is more important than retaining 23 countries.

## Step 3 — Pre-specify the forecasting exercise

- Use a long test period with at least 32 forecast origins where data permit.
- Tune every model on validation data only; lock all choices before final test
  evaluation.
- Report one-, two-, and four-quarter-ahead forecasts.
- Use point and probabilistic forecast evaluation: RMSE, MAE, CRPS/log score,
  interval coverage, and sharpness.

## Step 4 — Benchmark and ablation suite

Required comparators:

1. Persistence / random walk.
2. ETS and auto-ARIMA or a transparently tuned ARIMA/ARX.
3. Dynamic-factor model and a regularized panel/global model.
4. A graph-free LSTM or MLP using the identical node features.
5. Static, lagged, shuffled-edge, zero-edge, and directed-graph variants.

The ST-GNN may be presented as useful only if it has incremental forecast value
relative to this suite.  If it does not, the paper should be reframed as a
well-documented negative result, not as evidence of contagion.

## Step 5 — Valid inference and interpretation

- Treat forecast origins, not country-quarter rows, as the primary independent
  time dimension.  Use a moving-block/bootstrap or an equivalent panel forecast
  comparison procedure that preserves cross-country dependence.
- Pre-specify any equivalence margin and report confidence intervals for loss
  differentials.
- Treat integrated gradients as model sensitivity only.  Validate any claimed
  partner importance against trade strength/centrality, multiple seeds, shuffled
  graphs, and edge-deletion counterfactuals.
- Do not use “contagion”, “transmission channel”, or causal policy language without
  an identification design or structural validation.

## Step 6 — IJF submission package

- Blinded manuscript and separate title page.
- Public repository with raw-data acquisition instructions, hashes, environments,
  seeds, software/hardware versions, and a one-command reproduction path.
- Full online appendix containing all countries, all horizons, all model settings,
  all forecasts, and all failed/negative checks.
