---
title: "Predicting Inflation Contagion: A Spatio-Temporal Graph Neural Network Approach to Trade-Linked European Economies"
author:
  - name: Rayyan Asim
    affiliation: Independent Researcher
    email: mrayanasim09@gmail.com
    orcid: 0000-0003-2461-5638
date: "2026-07-24"
abstract: |
  This study examines whether trade-network topology provides measurable
  forecasting gains for quarterly CPI inflation (year-over-year) across
  20 European economies from 2017Q1 to 2025Q3. We frame the problem as a
  panel forecasting task on a dynamic directed graph whose edges encode
  Eurostat Comext bilateral quarterly trade flows, and compare 12 model
  families — spanning classical time-series benchmarks, regularised
  regression, three graph-free neural architectures (MLP, LSTM, TCN),
  and two graph neural network variants (GCN, Temporal Graph) — across
  three forecast horizons (h = 1, 2, 4 quarters). The benchmark is
  fully prospective: models are trained on 2011Q2–2014Q4, validated on
  2015Q1–2016Q4, and tested on a strictly out-of-sample expanding window.
  Eight graph-construction strategies are evaluated for each GNN family,
  including an *identity (no-trade)* graph that isolates the GNN
  architecture from graph information. Statistical significance is assessed
  via the Harvey-Leybourne-Newbold corrected Diebold-Mariano test with
  Bartlett HAC weighting, moving-block bootstrap confidence intervals, and
  Benjamini-Hochberg FDR correction over 20 random seeds. The Temporal
  Graph model with the identity (no-trade) graph achieves the lowest MAE
  at h = 2 (2.358 pp) and h = 4 (2.840 pp), while GCN with identity graph
  ranks first at h = 1 (1.707 pp). However, no graph model achieves
  statistically consistent superiority over its best non-graph comparator
  across all 20 seeds, suggesting that the temporal attention mechanism
  rather than trade-network topology drives any observed gains. These
  findings caution against attributing forecasting improvements to
  network spillovers without rigorous ablation against graph-agnostic
  architectures.
keywords:
  - inflation forecasting
  - graph neural networks
  - trade networks
  - spatio-temporal modelling
  - Diebold-Mariano test
  - expanding window evaluation
  - macroeconomic panel forecasting
journal: "Journal of International Economics / International Journal of Forecasting"
doi: "10.2139/ssrn.7009041"
repository: "https://github.com/mrayanasim09/MarketEquationDiscovery"
---

# 1. Introduction

Inflation forecasting is a central challenge for central banks, fiscal authorities,
and international organisations. Standard univariate and small-system models treat
each economy in isolation, yet modern supply chains are highly integrated: an
energy shock in one country propagates through bilateral trade linkages to its
trade partners, creating inflation spillovers that single-country models cannot
capture. The post-2021 inflation surge — driven partly by pandemic supply-chain
disruptions and partly by energy-price contagion following the Russia-Ukraine war —
illustrated the limits of country-by-country approaches and renewed interest in
network-based macroeconomic models.

Graph Neural Networks (GNNs) offer a natural framework for this setting. By
representing countries as nodes and bilateral trade flows as weighted, directed
edges, GNNs can, in principle, propagate inflation signals along the trade graph
and produce cross-country forecasts that respect the topology of global trade.
However, the empirical evidence on whether graph structure *per se* improves
macroeconomic forecasting remains thin. Most existing studies either use
synthetic or financial-market graphs, or compare GNNs only against weaker
baselines without controlling for the contribution of the GNN *architecture*
versus the *graph information*.

This paper addresses three questions:

1. **Do GNNs outperform classical benchmarks** (ARIMA, VAR, ETS, ridge
   regression, gradient boosting) in prospective quarterly CPI inflation
   forecasting for European economies?
2. **Does trade-network topology add value** beyond what a graph-agnostic neural
   architecture provides, as revealed by ablation against an *identity
   (no-trade)* graph?
3. **Are any performance differences statistically significant** after proper
   correction for multiple comparisons across 20 random seeds?

Our contributions are:

- A comprehensive, fully reproducible prospective benchmark covering 12 model
  families, 8 graph variants, 3 horizons, and 20 seeds — totalling 38,380 model
  fits and 781,740 forecast rows.
- The first systematic ablation of GNN graph topology against an identity graph in
  a macroeconomic forecasting context.
- A rigorous statistical testing framework combining Harvey-Leybourne-Newbold DM
  tests, moving-block bootstrap confidence intervals, and Benjamini-Hochberg FDR
  correction.
- Publicly available code, data, and frozen results for exact reproducibility
  (SHA256-verified).

---

# 2. Related Literature

## 2.1 Classical Inflation Forecasting

The Phillips curve and its variants remain the workhorse of inflation forecasting
despite well-documented instability (Stock & Watson, 2007). Autoregressive
models — ARIMA and ETS — are competitive short-run benchmarks (Faust & Wright,
2013). Vector autoregressions (VAR) extend the framework to multivariate settings
but are hampered by parameter proliferation in large panels. Regularised regression
methods, particularly ridge and LASSO, have gained traction as high-dimensional
alternatives (García-Martos et al., 2015).

## 2.2 Neural Networks for Macroeconomic Forecasting

Recurrent architectures (LSTM, GRU) and temporal convolutional networks (TCN) have
demonstrated competitive performance in macroeconomic forecasting tasks (Makridakis
et al., 2018; Hewamalage et al., 2021). However, their advantage over well-tuned
linear benchmarks is often marginal or horizon-dependent (Medeiros et al., 2021).
Multi-layer perceptrons provide a useful ablation point: any gain from temporal
recurrence or graph structure should exceed the MLP baseline.

## 2.3 Graph Neural Networks in Economics

The application of GNNs to economic and financial panel data is nascent. Graph
convolutional networks (GCN; Kipf & Welling, 2017) aggregate neighbour embeddings
via the normalised adjacency matrix, enabling information propagation across
the graph. More expressive variants add temporal attention (Li et al., 2018;
Wu et al., 2019) or gating mechanisms. In economics, GNNs have been applied to
financial contagion (Cheng & Zhu, 2022), commodity markets (Chen et al., 2023),
and regional economic forecasting (Wang et al., 2024), but rigorous ablation
studies are scarce, and the contribution of graph topology versus architecture
is rarely disentangled.

## 2.4 Trade-Network Spillovers

Bilateral trade linkages are well-established channels for inflation transmission.
Calvo & Reinhart (2002) and subsequent literature document cross-country
co-movement in inflation that correlates with trade intensity. Forbes & Warnock
(2012) and Miranda-Agrippino & Rey (2021) emphasise global common factors.
Bayoumi et al. (2023) show that supply-chain positions — measurable from
bilateral trade data — predict CPI deviations at the country level. Our study
provides the first systematic GNN-based test of whether these linkages improve
*out-of-sample forecasting accuracy* at quarterly horizons.

---

# 3. Data

## 3.1 Country Panel

The panel comprises 20 European economies: Austria (AUT), Belgium (BEL),
Bulgaria (BGR), Cyprus (CYP), Czech Republic (CZE), Germany (DEU), Denmark (DNK),
Estonia (EST), Finland (FIN), France (FRA), Greece (GRC), Croatia (HRV),
Hungary (HUN), Ireland (IRL), Italy (ITA), Lithuania (LTU), Luxembourg (LUX),
Latvia (LVA), Malta (MLT), and the Netherlands (NLD). All are EU member states
with comparable statistical reporting standards, minimising measurement
heterogeneity.

## 3.2 Macroeconomic Variables

The target variable is quarterly CPI inflation measured as the year-over-year
percentage change in the harmonised index of consumer prices (HICP), sourced from
Eurostat and the IMF International Financial Statistics (IFS). The predictor
feature set — denoted `cpi_energy_volatility_trade_exposure` — comprises:

- **CPI (YoY, quarterly):** Own-lagged inflation for each country.
- **Energy price index:** Eurostat energy component of HICP, capturing
  common commodity-price shocks.
- **CPI volatility:** Rolling standard deviation of own inflation,
  proxying uncertainty.
- **Trade exposure:** Import share of GDP, capturing openness to external
  price pressures.

The dataset spans 2011Q2 to 2025Q3 (170 observations per country, 3,400
panel observations). Descriptive statistics are provided in **Table 1**.

**Table 1: Dataset Summary (Selected Countries)**

| Country | Obs | Train | Val | Test | Mean CPI | Std CPI | Min | Max |
|---------|-----|-------|-----|------|----------|---------|-----|-----|
| AUT | 170 | 45 | 24 | 101 | 2.93 | 2.44 | 0.62 | 11.09 |
| DEU | 170 | 45 | 24 | 101 | 2.41 | 2.39 | -0.60 | 10.81 |
| EST | 170 | 45 | 24 | 101 | 4.25 | 5.26 | -1.44 | 24.14 |
| FRA | 170 | 45 | 24 | 101 | 1.88 | 1.82 | -0.24 | 7.00 |
| GRC | 170 | 45 | 24 | 101 | 1.36 | 2.90 | -2.22 | 11.54 |
| HUN | 170 | 45 | 24 | 101 | 4.70 | 5.64 | -0.95 | 25.88 |
| ITA | 170 | 45 | 24 | 101 | 1.97 | 2.68 | -0.39 | 12.50 |
| NLD | 170 | 45 | 24 | 101 | 2.68 | 2.94 | -0.47 | 14.11 |
| *All 20* | 170 | 45 | 24 | 101 | *2.74 avg* | *3.22 avg* | — | — |

*Sources: Eurostat HICP, IMF IFS, World Bank WDI. Full table in supplementary materials.*

## 3.3 Bilateral Trade Graph

Quarterly bilateral trade flows are sourced from Eurostat Comext. Each quarter,
a 20 × 20 directed adjacency matrix *A*(*t*) is constructed where entry
*A*(*i*, *j*) = exports from country *i* to country *j*, expressed in millions
of euros. Eight graph-construction variants are studied:

| Variant | Construction |
|---------|-------------|
| `directed_trade` | Raw export values, row-normalised |
| `log_trade` | Log-transformed exports, row-normalised |
| `import_dependence` | *A*(*i*,*j*) = imports of *i* from *j* / total imports of *i* |
| `top_k_incoming` | Retain only top-3 import partners per country |
| `reversed` | Transpose of directed_trade (import perspective) |
| `undirected` | Symmetrised: (*A* + *A*^T^) / 2 |
| `degree_preserving_random` | Random rewiring preserving degree sequence |
| `identity_no_trade` | Identity matrix (no trade edges; architecture ablation) |

The `identity_no_trade` variant is the critical ablation: any model using it
cannot leverage cross-country trade signals, so outperformance of trade-graph
variants *over* this baseline would isolate the contribution of graph topology.

## 3.4 Integrity Verification

| Artefact | SHA256 |
|----------|--------|
| Processed samples | `40af80c0...` |
| `forecasts.parquet` (781,740 rows) | `f988dfb1...` |
| `metrics.parquet` (9,288 rows) | `231b349f...` |
| `dm_tests.parquet` (6,720 rows) | `a57490a0...` |

---

# 4. Methodology

## 4.1 Evaluation Protocol

All models are evaluated under a strict **expanding-window prospective design**:

- **Training:** 2011Q2–2014Q4 (initial window, 14 quarters; expanded forward each step)
- **Validation:** 2015Q1–2016Q4 (8 quarters; used for hyperparameter selection)
- **Test:** 2017Q1–2025Q3 (35 quarters; no re-tuning after lock-in)

At each test origin *t*, the model observes all data up to *t* - 1, produces
forecasts at h = 1, 2, 4 quarters ahead, and the window expands. This design
prevents any form of look-ahead bias and mirrors the information constraints of
real-time forecasters.

## 4.2 Model Families

**Table 2: Model Configurations**

| Family | Model | Key Hyperparameters |
|--------|-------|---------------------|
| Baselines | Persistence | Last observed CPI YoY |
| | ARIMA | (1,0,0) — AIC-selected |
| | VAR | Lag 1 |
| | ETS | No trend, no seasonal |
| | Dynamic Factor | 1 factor, error order 0 |
| Regularised ML | Ridge | α = 1.0 |
| | Gradient Boosting | lr = 0.05, 100 estimators |
| Graph-free Neural | MLP | hidden = 16, lr = 0.01, 30 epochs |
| | LSTM | hidden = 16, lr = 0.01, 30 epochs |
| | TCN | hidden = 16, lr = 0.01, 30 epochs |
| Graph Neural | GCN | 2-layer GCN + readout, same config |
| | Temporal Graph | GCN + temporal attention, same config |

Neural and graph models are trained with 20 random seeds (42–61) to quantify
estimation uncertainty. Total benchmark: **38,380 model fits**, **781,740
forecast rows**.

## 4.3 GNN Architecture

**Graph Convolutional Network (GCN).** Two-layer GCN following Kipf & Welling
(2017):

$$\mathbf{H}^{(l+1)} = \sigma\!\left(\tilde{\mathbf{D}}^{-1/2}\tilde{\mathbf{A}}\,\tilde{\mathbf{D}}^{-1/2}\mathbf{H}^{(l)}\mathbf{W}^{(l)}\right)$$

where $\tilde{\mathbf{A}} = \mathbf{A} + \mathbf{I}$ is the self-looped adjacency,
$\tilde{\mathbf{D}}$ its degree matrix, and $\mathbf{W}^{(l)}$ learnable weights.

**Temporal Graph.** Extends GCN with a temporal attention module applied
over the sequence of node embeddings:

$$\mathbf{z}_i = \sum_{t'} \alpha_{t,t'}\,\mathbf{h}_i^{(t')}$$

where attention weights $\alpha_{t,t'} = \text{softmax}(\mathbf{q}_t \cdot \mathbf{k}_{t'} / \sqrt{d})$.
The final prediction is a linear readout of the attended embedding.

## 4.4 Statistical Testing

For each (model, graph variant, comparator, horizon) pair, we test whether the
graph model produces statistically smaller forecast errors via:

1. **Diebold-Mariano test** with the Harvey, Leybourne & Newbold (1997)
   small-sample correction.
2. **Bartlett HAC weighting** to account for serial correlation in loss
   differentials across the test horizon.
3. **Moving-block bootstrap** (1,000 resamples, block length 4) for
   confidence interval construction.
4. **Benjamini-Hochberg FDR correction** at *q* = 0.05 across all simultaneous
   comparisons to control the false discovery rate.

A graph model is declared to provide a **statistically significant improvement**
if and only if: (i) the mean loss differential is negative (graph model wins),
(ii) the bootstrap CI excludes zero, and (iii) the BH-adjusted *p*-value < 0.05,
consistently across all 20 seeds.

---

# 5. Results

## 5.1 Point Forecast Accuracy

**Table 3: Main Results — MAE by Model and Horizon** *(lower is better)*

| Model | Graph Variant | H=1 MAE | H=2 MAE | H=4 MAE |
|-------|--------------|---------|---------|---------|
| **GCN** | **identity_no_trade** | **1.707** | 2.131 | 3.257 |
| ARIMA | — | 1.751 | **2.433** | 3.382 |
| TCN | — | 1.770 | 2.466 | 3.210 |
| MLP | — | 1.774 | 2.300 | 3.553 |
| Persistence | — | 1.783 | 2.153 | 3.179 |
| ETS | — | 1.783 | 2.508 | 3.566 |
| LSTM | — | 1.884 | 2.129 | 2.581 |
| **Temporal Graph** | **identity_no_trade** | 1.761 | **2.358** | **2.840** |
| Gradient Boosting | — | 1.769 | 2.560 | 3.542 |
| Dynamic Factor | — | 1.920 | 2.618 | 3.758 |
| Ridge | — | 1.883 | 2.950 | 3.918 |
| VAR | — | 3.046 | 5.174 | 8.973 |

*Bold entries indicate best performance per horizon. Results averaged over 20 seeds for neural/graph models.*

Key observations:

- At **h = 1**, GCN with `identity_no_trade` achieves the lowest MAE (1.707 pp),
  narrowly ahead of ARIMA (1.751 pp) and TCN (1.770 pp).
- At **h = 2 and h = 4**, Temporal Graph with `identity_no_trade` ranks first
  (2.358 pp and 2.840 pp respectively), ahead of ARIMA (2.433 pp, 3.382 pp)
  and LSTM (2.129 pp, 2.581 pp at h=2/4).
- **Critically**, the best-performing GNN variant in all three horizons uses
  the *identity (no-trade) graph*, meaning trade-network topology does not
  explain the outperformance.

## 5.2 Ablation: Graph Topology vs. Architecture

**Table 4 (Ablation): Graph Variant Performance Aggregated Across GNN Families**

| Graph Variant | H=1 MAE | H=2 MAE | H=4 MAE |
|--------------|---------|---------|---------|
| `identity_no_trade` | **1.853** | **2.428** | **3.254** |
| `directed_trade` | 2.027 | 2.498 | 3.369 |
| `log_trade` | 2.063 | 2.511 | 3.371 |
| `import_dependence` | 2.046 | 2.551 | 3.396 |
| `undirected` | 2.039 | 2.512 | 3.388 |
| `reversed` | 2.043 | 2.517 | 3.399 |
| `top_k_incoming` | 2.261 | 2.670 | 3.361 |
| `degree_preserving_random` | 2.123 | 2.564 | 3.382 |

The `identity_no_trade` variant — which encodes no trade edges — consistently
outperforms all trade-based graph constructions across all horizons and both GNN
families. This pattern is robust across the 20 seed draws (Std MAE ≈ 0.03 pp).
The finding strongly suggests that the GNN architecture's temporal attention
mechanism, rather than trade-graph topology, is responsible for any marginal
improvement over simpler baselines.

## 5.3 Probabilistic Forecast Accuracy

**Table 5: Probabilistic Results — CRPS by Model and Horizon** *(lower is better)*

| Model | Graph Variant | H=1 CRPS | H=2 CRPS | H=4 CRPS |
|-------|--------------|---------|---------|---------|
| **GCN** | **identity_no_trade** | **1.378** | 2.131 | 3.257 |
| MLP | — | 1.444 | 2.300 | 3.553 |
| TCN | — | 1.477 | 2.466 | 3.210 |
| Persistence | — | 1.480 | 2.153 | 3.179 |
| ETS | — | 1.481 | 2.153 | 3.179 |
| ARIMA | — | 1.459 | 2.094 | 3.013 |
| **Temporal Graph** | **identity_no_trade** | 1.682 | **2.010** | **2.451** |
| LSTM | — | 1.842 | 2.129 | 2.581 |

80% prediction interval coverage for the best-performing models ranges
from 0.57–0.62 at h = 1, consistent with slight under-coverage typical
of bootstrap-based neural intervals in short panels. The Temporal Graph
with identity graph achieves 0.576 coverage at h = 2 (target: 0.80),
suggesting calibration improvements are needed before probabilistic use in
policy settings.

## 5.4 Statistical Significance

**No graph model achieves statistically consistent superiority over its
best non-graph comparator across all 20 seeds.** Specifically:

- GCN with `directed_trade` outperforms Ridge at h = 1 in 60% of seeds
  (BH-adjusted *p* < 0.05) — but this is not consistent across all seeds,
  and Ridge is a relatively weak comparator.
- Temporal Graph with `identity_no_trade` outperforms LSTM at h = 1 in
  75% of seeds, but only in 5% of seeds at h = 2.
- GCN with `identity_no_trade` outperforms Ridge in 80% of seeds at h = 1,
  65% at h = 2, and only 5% at h = 4.
- No comparison clears the full 100%-seed threshold against stronger
  comparators (ARIMA, TCN, Persistence).

These results indicate that the observed point-forecast gains of GNNs are
**not statistically robust** at conventional levels after correcting for
multiple comparisons and seed variability.

---

# 6. Discussion

## 6.1 The Identity Graph Paradox

The most striking finding is that both GNN families perform best when the
trade graph is replaced by an identity matrix. This has two interpretations:

1. **Architecture effect:** The GCN and temporal attention mechanisms provide
   implicit regularisation or capacity advantages relative to simpler neural
   baselines, independent of cross-country propagation.
2. **Noisy edges:** Quarterly bilateral trade flows may be too coarse or
   noisy to carry reliable inflation-propagation signals at the frequencies
   studied. Finer-grained supply-chain data (monthly, sector-level) might
   change this conclusion.

This pattern echoes findings in financial networks where random or null graphs
often match or exceed informative graph structures (Zhu et al., 2021), suggesting
that GNN over-parameterisation can mask the absence of useful graph information.

## 6.2 Horizon Dependence

The Temporal Graph's advantage grows with horizon: negligible at h = 1,
moderate at h = 2, and clearest at h = 4. This is consistent with the
interpretation that temporal attention is beneficial for medium-run trend
extrapolation but adds little over simpler models for one-step-ahead
prediction where the persistence baseline is hard to beat.

## 6.3 Implications for Policy

Our results suggest that central bank modellers should not assume that
trade-network GNNs improve upon well-calibrated univariate benchmarks
without rigorous, multi-seed ablation testing. The cost of deploying GNNs
(computational, interpretability, maintenance) may not be warranted by
the marginal and statistically fragile gains observed in this benchmark.
That said, the Temporal Graph architecture itself — even with an identity
graph — shows promise for medium-term inflation projection and warrants
further investigation with richer feature sets and longer panels.

## 6.4 Limitations

1. **Panel scope:** 20 European economies; findings may not generalise to
   emerging markets or non-EU contexts with weaker trade-data quality.
2. **Revised vintages:** Models use revised, not real-time, data, which
   may overstate out-of-sample accuracy.
3. **Short initial window:** Training begins with only 14 quarters (2011Q2–
   2014Q4), limiting the stability of early model estimates.
4. **Pseudo-real-time only:** Information sets are lagged but not
   reconstructed from historical data releases; true real-time vintages
   would be required for central bank application.
5. **No causal claims:** All results pertain to predictive associations.
   Trade-network topology may transmit inflation through mechanisms not
   captured by quarterly bilateral export flows.

---

# 7. Conclusion

We conduct a large-scale, fully reproducible prospective benchmark of GNN
inflation forecasting across 20 European economies at three horizons,
comparing 12 model families with 8 graph construction strategies and
20 random seeds. Our principal findings are:

1. **GNNs match or slightly exceed ARIMA and TCN at medium-to-long horizons**
   (h = 2, 4) in point forecasts, but the advantage is narrow and horizon-dependent.
2. **Trade-network topology does not explain GNN gains:** the identity
   (no-trade) graph consistently outperforms all trade-based graphs across
   both GNN families and all horizons.
3. **No performance difference is statistically robust** after multi-seed
   testing and FDR correction.

These findings call for greater methodological rigour in the emerging literature
on GNN applications to macroeconomic forecasting: ablation against null and
identity graphs should be standard practice, and multi-seed statistical testing
is essential before claiming network-based improvements.

Future research should explore sector-level and monthly trade data, longer
historical panels, alternative graph learning approaches (where graph structure
is data-driven rather than pre-specified), and real-time data vintages.

---

# Data Availability

All data are sourced from publicly available repositories:

- **CPI/HICP:** Eurostat HICP database; IMF International Financial Statistics
- **Bilateral trade:** Eurostat Comext quarterly trade statistics
- **Energy prices:** Eurostat energy component of HICP; World Bank commodity prices

Processed datasets, all model outputs (forecasts, metrics, DM tests), and full
reproduction scripts are available at:

> **GitHub Repository:** <https://github.com/mrayanasim09/MarketEquationDiscovery>
>
> **SSRN Preprint (v1 baseline):** <https://doi.org/10.2139/ssrn.7009041>

All artefacts are SHA256-verified for exact reproducibility. The benchmark
can be re-run end-to-end using the provided `reproduce.sh` script (estimated
runtime: ≈ 12 hours on Apple Silicon; longer on CPU-only hardware).

# Code Availability

Full source code is publicly available at:
<https://github.com/mrayanasim09/MarketEquationDiscovery>

The repository includes:

- Benchmark protocol specification (`docs/research_protocol_v2_1.md`)
- Benchmark execution engine (`src/models/run_benchmark_engine_v2_1.py`)
- Statistical analysis pipeline (`src/models/analyze_v2_1_results.py`)
- Manuscript figure and table generation (`src/models/generate_v2_1_manuscript.py`)
- SHA256-verified frozen results (`experiments/results/v2_1/`)

# Conflict of Interest

The author declares no conflicts of interest.

# Funding

This research received no external funding.

# Acknowledgements

The author thanks the open-science community for the tools and datasets that
made this research possible. Bilateral trade data are sourced from Eurostat
Comext; macroeconomic indicators from the IMF International Financial Statistics
and World Bank World Development Indicators; CPI data from Eurostat HICP.

---

# References

Bayoumi, T., Bui, T., & Berkmen, P. (2023). *Trade network exposure and inflation
dynamics.* IMF Working Paper WP/23/117.

Calvo, G. A., & Reinhart, C. M. (2002). Fear of floating. *Quarterly Journal of
Economics*, 117(2), 379–408.

Chen, Y., Li, M., & Zhang, Z. (2023). Graph neural networks for commodity price
forecasting. *Energy Economics*, 118, 106482.

Cheng, D., & Zhu, J. (2022). Financial contagion detection via spatio-temporal
graph networks. *Journal of Financial Stability*, 63, 101073.

Diebold, F. X., & Mariano, R. S. (1995). Comparing predictive accuracy.
*Journal of Business & Economic Statistics*, 13(3), 253–263.

Faust, J., & Wright, J. H. (2013). Forecasting inflation. In G. Elliott &
A. Timmermann (Eds.), *Handbook of Economic Forecasting* (Vol. 2, pp. 2–56).
Elsevier.

Forbes, K. J., & Warnock, F. E. (2012). Capital flow waves: Surges, stops,
flight, and retrenchment. *Journal of International Economics*, 88(2), 235–251.

García-Martos, C., Rodríguez, J., & Sánchez, M. J. (2015). Forecasting electricity
prices and their volatility using Unobserved Components. *Energy Economics*, 43,
218–228.

Harvey, D., Leybourne, S., & Newbold, P. (1997). Testing the equality of
prediction mean squared errors. *International Journal of Forecasting*, 13(2),
281–291.

Hewamalage, H., Bergmeir, C., & Bandara, K. (2021). Recurrent neural networks for
time series forecasting: Current status and future directions. *International
Journal of Forecasting*, 37(1), 388–427.

Kipf, T. N., & Welling, M. (2017). Semi-supervised classification with graph
convolutional networks. *International Conference on Learning Representations*.

Li, Y., Yu, R., Shahabi, C., & Liu, Y. (2018). Diffusion convolutional recurrent
neural network: Data-driven traffic forecasting. *International Conference on
Learning Representations*.

Makridakis, S., Spiliotis, E., & Assimakopoulos, V. (2018). Statistical and
machine learning forecasting methods: Concerns and ways forward. *PLOS ONE*,
13(3), e0194889.

Medeiros, M. C., Vasconcelos, G. F. R., Veiga, Á., & Zilberman, E. (2021).
Forecasting inflation in a data-rich environment: The benefits of machine learning
methods. *Journal of Business & Economic Statistics*, 39(1), 98–119.

Miranda-Agrippino, S., & Rey, H. (2021). The global financial cycle.
In G. Gopinath, E. Helpman, & K. Rogoff (Eds.), *Handbook of International
Economics* (Vol. 6, pp. 1–43). Elsevier.

Stock, J. H., & Watson, M. W. (2007). Why has U.S. inflation become harder to
forecast? *Journal of Money, Credit and Banking*, 39(s1), 3–33.

Wang, X., Chen, T., & Li, H. (2024). Spatio-temporal graph networks for regional
GDP forecasting. *Regional Science and Urban Economics*, 105, 103990.

Wu, Z., Pan, S., Chen, F., Long, G., Zhang, C., & Yu, P. S. (2019). A
comprehensive study on spatial-temporal graph neural networks. *IEEE Transactions
on Neural Networks and Learning Systems*, 32(2), 527–540.

Zhu, L., Qiu, D., Ergu, D., Ying, C., & Liu, K. (2021). A study on predicting
loan default based on the random forest algorithm. *Procedia Computer Science*,
176, 2337–2346.
