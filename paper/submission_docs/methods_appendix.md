# Methods Appendix — v2.1

## Model Architectures

**Graph Convolutional Network (GCN):** A single-layer spatial graph convolution
over node features: $\mathbf{H} = \mathrm{ReLU}(\tilde{\mathbf{A}}\mathbf{X}\mathbf{W}_{\text{node}})$,
where $\tilde{\mathbf{A}}$ is the row-normalised adjacency matrix, followed by a
linear readout $\hat{\mathbf{y}} = \mathbf{H}\mathbf{W}_{\text{head}}$.
Hidden dimension $d = 16$.

**Temporal Graph:** Extends GCN by computing spatial node embeddings at each
lagged time step $t' \in \{t-K, \ldots, t-1\}$ (lookback window $K = 4$ quarters),
then feeding the sequence $\{\mathbf{h}_i^{(t')}\}$ through an LSTM temporal
encoder. The LSTM final hidden state is passed to a linear readout.

**Classical baselines:** ARIMA (AIC-selected $p \in \{1,\ldots,4\}$, automatic
differencing), VAR ($p$ selected by AIC up to 2), ETS (automatic model selection).

**Neural baselines:** MLP (2 hidden layers, hidden dim 16), LSTM (hidden dim 16,
lookback 4 quarters), TCN (dilated causal convolutions, hidden dim 16).

**Regularised regression:** Ridge (tuned on validation set, penalty $= 1.0$).

All five neural architectures (MLP, LSTM, TCN, GCN, Temporal Graph) share
identical hyperparameters — hidden dimension 16, learning rate 0.01, 30 training
epochs — to enable a controlled architecture-level comparison independent of
tuning effort.

## Graph Construction Strategies (8 variants)

| Variant | Description |
|---------|-------------|
| `identity_no_trade` | Identity matrix — no cross-country edges |
| `directed_trade` | Row-normalised bilateral export flow |
| `log_trade` | Log-transformed export flow |
| `import_dependence` | Import share of importer's total imports |
| `top_k_incoming` | Retain only top-$k=5$ import partners per country |
| `reversed` | Transpose of directed_trade (import perspective) |
| `undirected` | Symmetrised: (A + A') / 2 |
| `degree_preserving_random` | Random rewiring preserving degree sequence |

## Training Protocol

- **Optimizer:** Adam, learning rate 0.01, no weight decay
- **Loss:** MSE (point forecasts); CRPS (probabilistic outputs)
- **Epochs:** 30 (fixed; no early stopping — by design for controlled comparison)
- **Seeds:** 20 independent random seeds (42–61)

## Evaluation Protocol

- **Window type:** Expanding; origins from 2017Q1 to 2024Q4 (35 test origins)
- **Train period:** 2011Q2–2014Q4 (initial fit); expanding from 2015Q1 onwards
- **Validation period:** 2015Q1–2016Q4 (used for Ridge alpha selection only)
- **Test period:** 2017Q1–2025Q3 (strictly out-of-sample)
- **Horizons:** $h = 1, 2, 4$ quarters

## Statistical Testing

- **Diebold-Mariano (HLN):** Harvey, Leybourne and Newbold (1997) small-sample
  finite-horizon correction applied. Comparators: ARIMA, Ridge, MLP, LSTM, TCN.
- **HAC weighting:** Bartlett kernel to account for serial correlation
- **Bootstrap:** Moving-block bootstrap, 2,000 resamples, block length 4 quarters
- **FDR correction:** Benjamini-Hochberg at $q = 0.05$ across all simultaneous
  comparisons within each (model, variant, horizon) group

## Configuration Reference

All experiment parameters are locked in:
`experiments/results/v2_1/configs/benchmark_engine_v2_1.json`

All input data hashes are locked in:
`data/processed/v2/transformation_validation.json`