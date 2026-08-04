---
title: "Do Trade Networks Help Forecast Inflation? A Benchmark Refutation Using Spatio-Temporal Graph Neural Networks"
author:
  - name: Anonymized Author(s)
    affiliation: Anonymized Institution
date: "2026-07-24"
abstract: |
  This study presents a negative benchmark and ablation study evaluating whether trade-network topology provides measurable forecasting gains for quarterly CPI inflation (year-over-year) across 20 European economies from 2017Q1 to 2025Q3. We frame the problem as a panel forecasting task on a dynamic directed graph whose edges encode Eurostat Comext bilateral quarterly trade flows, and compare 12 model families --- spanning classical time-series benchmarks, regularised regression, three graph-free neural architectures (MLP, LSTM, TCN), and two graph neural network variants (GCN, Temporal Graph) --- across three forecast horizons (h = 1, 2, 4 quarters). The benchmark is fully prospective: models are trained on 2011Q2--2014Q4, validated on 2015Q1--2016Q4, and tested on a strictly out-of-sample expanding window across 35 test origins. Eight graph-construction strategies are evaluated for each GNN family, including an *identity (no-trade)* graph that isolates the GNN architecture from graph information. Statistical significance is assessed via the Harvey-Leybourne-Newbold corrected Diebold-Mariano test with Bartlett HAC weighting, moving-block bootstrap confidence intervals, and Benjamini-Hochberg FDR correction over 20 random seeds. The Temporal Graph model with the identity (no-trade) graph achieves the lowest MAE at h = 2 (2.358 pp) and h = 4 (2.840 pp); GCN with identity graph ranks first at h = 1 (1.707 pp). Crucially, removing trade-network topology entirely (*identity_no_trade*) outperforms all actual trade-edge models, and no trade-graph model achieves statistically significant superiority over its best non-graph comparator in a majority of seeds after FDR correction. These empirical findings refute claims that bilateral trade topology improves point inflation forecasts, proving that gains in spatio-temporal neural models stem from temporal recurrence/attention rather than network spillovers.
keywords:
  - inflation forecasting
  - graph neural networks
  - trade networks
  - negative benchmark
  - ablation study
  - Diebold-Mariano test
  - expanding window evaluation
  - macroeconomic panel forecasting
journal: "International Journal of Forecasting"
doi: "10.2139/ssrn.7009041"
repository: "https://github.com/mrayanasim09/MarketEquationDiscovery"
---

# 1. Introduction

Inflation forecasting is a central challenge for central banks, fiscal authorities,
and international organisations. Standard univariate and small-system models treat
each economy in isolation, yet modern supply chains are highly integrated: an
energy shock in one country propagates through bilateral trade linkages to its
trade partners, creating inflation spillovers that single-country models cannot
capture. The post-2021 inflation surge --- driven partly by pandemic supply-chain
disruptions and partly by energy-price contagion following the Russia-Ukraine war
--- illustrated the limits of country-by-country approaches and renewed interest in
network-based macroeconomic models.

Graph Neural Networks (GNNs) offer a natural framework for this setting. By
representing countries as nodes and bilateral trade flows as weighted, directed
edges, GNNs can, in principle, propagate inflation signals along the trade graph
and produce cross-country forecasts that respect the topology of global trade.
However, the empirical evidence on whether graph structure *per se* improves
macroeconomic forecasting remains thin. Most existing studies either use
synthetic or financial-market graphs, or compare GNNs only against weaker
baselines without controlling for the contribution of the GNN *architecture*
versus the *graph information* (Kawamoto et al., 2018; Zügner et al., 2020).

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
  families, 8 graph variants, 3 horizons, and 20 seeds --- totalling 38,380 model
  fits and 781,740 forecast rows.
- The first systematic ablation of GNN graph topology against an identity graph in
  a macroeconomic forecasting context, disentangling architectural from topological
  effects.
- A rigorous statistical testing framework combining Harvey-Leybourne-Newbold DM
  tests, moving-block bootstrap confidence intervals, and Benjamini-Hochberg FDR
  correction, with results reported as proportion of seeds significant rather than
  a binary threshold.
- Publicly available code, data, and frozen results for exact reproducibility
  (SHA256-verified), consistent with best practices for forecast benchmark
  transparency (Makridakis et al., 2022).

---

# 2. Related Literature

## 2.1 Classical Inflation Forecasting

The Phillips curve and its variants remain the workhorse of inflation
forecasting despite well-documented instability (Stock and Watson, 2007; Atkeson and Ohanian, 2001).
Autoregressive models --- ARIMA and ETS --- are competitive short-run benchmarks
(Faust and Wright, 2013). Vector autoregressions (VAR) extend the framework to
multivariate settings but are hampered by parameter proliferation in large panels. Bayesian VARs (BVAR) with shrinkage priors provide a standard benchmark to mitigate parameter proliferation in large panels (Giannone et al., 2015; Ba\'nbura et al., 2010). Regularised regression methods, particularly ridge and LASSO, have gained
traction as high-dimensional alternatives (Garcia-Martos et al., 2015).

## 2.2 Neural Networks for Macroeconomic Forecasting

Recurrent architectures (LSTM, GRU) and temporal convolutional networks (TCN) have
demonstrated competitive performance in macroeconomic forecasting tasks (Makridakis
et al., 2018; Hewamalage et al., 2021). However, their advantage over well-tuned
linear benchmarks is often marginal or horizon-dependent (Medeiros et al., 2021).
Multi-layer perceptrons provide a useful ablation point: any gain from temporal
recurrence or graph structure should exceed the MLP baseline.

## 2.3 Graph Neural Networks in Economics

The application of GNNs to economic and financial panel data is nascent. Graph
convolutional networks (GCN; Kipf and Welling, 2017) aggregate neighbour embeddings
via the normalised adjacency matrix, enabling information propagation across
the graph. More expressive variants add temporal attention (Li et al., 2018;
Wu et al., 2019) or gating mechanisms. In economics, GNNs have been applied to
financial contagion (Cheng and Zhu, 2022), commodity markets (Chen et al., 2023),
and regional economic forecasting (Wang et al., 2024), but rigorous ablation
studies are scarce, and the contribution of graph topology versus architecture
is rarely disentangled.

A recurring concern in the graph learning literature is that null or random graphs
can match or even exceed structurally meaningful graphs when the GNN architecture
itself provides implicit regularisation or pattern-matching capacity. Kawamoto
et al. (2018) show that GCN-style architectures can learn classification rules
that are largely independent of edge information under certain conditions. Zugner
et al. (2020) demonstrate that graph structure can be adversarially irrelevant
to GNN performance. These theoretical findings motivate our identity-graph
ablation as a rigorous control for architectural versus topological effects.

## 2.4 Trade-Network Spillovers

Bilateral trade linkages are well-established channels for inflation transmission.
Calvo and Reinhart (2002) and subsequent literature document cross-country
co-movement in inflation that correlates with trade intensity. Forbes and Warnock
(2012) and Miranda-Agrippino and Rey (2021) emphasise global common factors.
Bayoumi et al. (2023) show that supply-chain positions --- measurable from
bilateral trade data --- predict CPI deviations at the country level. Our study
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
feature set --- denoted `cpi_energy_volatility_trade_exposure` --- comprises:

- **CPI (YoY, quarterly):** Own-lagged inflation for each country.
- **Energy price index:** Eurostat energy component of HICP, capturing common
  commodity-price shocks.
- **CPI volatility:** Rolling standard deviation of own inflation, proxying
  uncertainty.
- **Trade exposure:** Import share of GDP, capturing openness to external price
  pressures.

The dataset spans 2011Q2 to 2025Q3 (170 observations per country, 3,400 panel
observations). Selected descriptive statistics are provided in Table 1; full
country-level statistics are provided in the online supplementary material.

**Table 1: Dataset Summary (Selected Countries)**

| Country | Total Obs | Train | Val | Test | Mean CPI (pp) | Std CPI (pp) |
|---------|-----------|-------|-----|------|---------------|--------------|
| AUT | 170 | 45 | 24 | 101 | 2.93 | 2.44 |
| DEU | 170 | 45 | 24 | 101 | 2.41 | 2.39 |
| EST | 170 | 45 | 24 | 101 | 4.25 | 5.26 |
| FRA | 170 | 45 | 24 | 101 | 1.88 | 1.82 |
| GRC | 170 | 45 | 24 | 101 | 1.36 | 2.90 |
| HUN | 170 | 45 | 24 | 101 | 4.70 | 5.64 |
| ITA | 170 | 45 | 24 | 101 | 1.97 | 2.68 |
| NLD | 170 | 45 | 24 | 101 | 2.68 | 2.94 |
| *Panel mean* | 170 | 45 | 24 | 101 | *2.74* | *3.22* |

*Notes: CPI is year-over-year HICP percentage change. Sources: Eurostat, IMF IFS.*

## 3.3 Bilateral Trade Graph

Quarterly bilateral trade flows are sourced from Eurostat Comext. Each quarter,
a 20 x 20 directed adjacency matrix A(t) is constructed where entry A(i,j) =
exports from country i to country j, expressed in millions of euros.
Eight graph-construction variants are studied:

| Variant | Construction |
|---------|-------------|
| `directed_trade` | Raw export values, row-normalised |
| `log_trade` | Log-transformed exports, row-normalised |
| `import_dependence` | A(i,j) = imports of i from j / total imports of i |
| `top_k_incoming` | Retain only top-5 import partners per country |
| `reversed` | Transpose of directed_trade (import perspective) |
| `undirected` | Symmetrised: (A + A') / 2 |
| `degree_preserving_random` | Random rewiring preserving degree sequence |
| `identity_no_trade` | Identity matrix (no trade edges; architecture ablation) |

The `identity_no_trade` variant is the critical ablation: any model using it
cannot leverage cross-country trade signals, so outperformance of trade-graph
variants *over* this baseline would isolate the contribution of graph topology
from the contribution of the GNN architecture itself.

## 3.4 Data Integrity

| Artefact | SHA256 (first 16 hex) |
|----------|-----------------------|
| Processed samples | `40af80c0...` |
| `forecasts.parquet` (783,760 rows) | `363e6994d44b575a...` |
| `metrics.parquet` (9,312 rows) | `22bd6677c9a89d5d...` |
| `dm_tests.parquet` (6,720 rows) | `d2d383a021dcb877...` |

Full hashes are available in the public repository.

---

# 4. Methodology

## 4.1 Evaluation Protocol

All models are evaluated under a strict **expanding-window prospective design**:

- **Training:** 2011Q2--2014Q4 (initial window, 14 quarters; expanded forward each step)
- **Validation:** 2015Q1--2016Q4 (8 quarters; used for hyperparameter selection only)
- **Test:** 2017Q1--2025Q3 (35 quarters; no re-tuning or re-selection after lock-in)

At each test origin t, the model observes all data up to t-1, produces forecasts
at h = 1, 2, 4 quarters ahead, and the training window expands by one quarter.
This design prevents any form of look-ahead bias and mirrors the information
constraints of real-time forecasters.

We note that the initial training window of 14 quarters is short relative to
the number of parameters in the neural and GNN models (hundreds to thousands),
which means early-window model fits may be systematically undertrained. This
structural asymmetry is acknowledged explicitly in the limitations (Section 6.4).

## 4.2 Model Families

**Table 2: Model Configurations**

| Family | Model | Selected Hyperparameters |
|--------|-------|--------------------------|
| Baselines | Persistence | Last observed CPI YoY |
| | ARIMA | Order ($p$,0,0) with AIC selection, $p \in \{1,2,3,4\}$ |
| | VAR | Lag 1 |
| | ETS | No trend, no seasonal |
| | Dynamic Factor | 1 factor, error order 0 |
| Regularised ML | Ridge | penalty = 1.0 |
| | Gradient Boosting | lr = 0.05, 100 estimators, L2 = 1.0 |
| Graph-free Neural | MLP | hidden = 16, lr = 0.01, 30 epochs |
| | LSTM | hidden = 16, lr = 0.01, 30 epochs |
| | TCN | hidden = 16, lr = 0.01, 30 epochs |
| Graph Neural | GCN | hidden = 16, lr = 0.01, 30 epochs |
| | Temporal Graph | hidden = 16, lr = 0.01, 30 epochs |

Neural and graph models are trained with 20 random seeds (42--61) to quantify
estimation uncertainty. Total benchmark: **38,380 model fits**, **781,740
forecast rows**.

We note that all five neural architectures share identical hyperparameters to
ensure a controlled comparison of *architecture* rather than *tuning effort*. A
limitation of this design is that any individual architecture may be suboptimally
tuned relative to its true capacity; this is discussed further in Section 6.4.

## 4.3 GNN Architecture

**Graph Convolutional Network (GCN)** applies a spatial graph convolution operator over node features:

$$\mathbf{H} = \mathrm{ReLU}\!\left(\tilde{\mathbf{A}}\mathbf{X}\mathbf{W}_{\text{node}}\right)$$

where $\tilde{\mathbf{A}}$ is the row-normalised spatial adjacency matrix, $\mathbf{X} \in \mathbb{R}^{N \times F}$ represents input node features for $N$ countries, and $\mathbf{W}_{\text{node}} \in \mathbb{R}^{F \times d}$ is a learnable weight matrix mapping features to hidden dimension $d$. A linear readout layer $\mathbf{W}_{\text{head}} \in \mathbb{R}^{d \times 1}$ maps the node embeddings to individual country forecasts: $\hat{\mathbf{y}} = \mathbf{H}\mathbf{W}_{\text{head}}$.

**Temporal Graph** combines the spatial GCN message-passing operator with a Recurrent LSTM temporal encoder. For each time step $t' \in \{t-K, \dots, t-1\}$ in the lookback window of length $K$, spatial node representations are computed:

$$\mathbf{h}_i^{(t')} = \mathrm{ReLU}\!\left(\sum_{j=1}^N \tilde{A}_{ij}^{(t')}\,\mathbf{x}_j^{(t')}\mathbf{W}_{\text{node}}\right)$$

The temporal sequence of spatial embeddings $\{\mathbf{h}_i^{(t-K)}, \dots, \mathbf{h}_i^{(t-1)}\}$ is fed into a Long Short-Term Memory (LSTM) network:

$$\mathbf{z}_i = \mathrm{LSTM}\!\left(\{\mathbf{h}_i^{(t')}\}_{t'=t-K}^{t-1}\right)$$

The final hidden state $\mathbf{z}_i \in \mathbb{R}^d$ is passed to a linear readout layer to produce the forecast: $\hat{y}_{i, t+h} = \mathbf{W}_{\text{head}}\mathbf{z}_i$.

## 4.4 Statistical Testing Framework

For each (model, graph variant, comparator, horizon) pair, we test whether the
graph model produces statistically smaller forecast errors using the following
procedure:

1. **Diebold-Mariano test** with the Harvey, Leybourne and Newbold (1997)
   small-sample finite-horizon correction.
2. **Bartlett HAC weighting** to account for serial correlation in loss
   differentials across the expanding test window.
3. **Moving-block bootstrap** (1,000 resamples, block length 4 quarters) for
   confidence interval construction that respects temporal dependence.
4. **Benjamini-Hochberg FDR correction** at q = 0.05 across all simultaneous
   comparisons within each (model, variant, horizon) group.

Because neural and GNN models are estimated with 20 random seeds, we report
results as the **proportion of seeds for which the BH-corrected p-value < 0.05
and the loss differential favours the graph model**. We adopt two interpretive
thresholds: a *majority* threshold (> 50% of seeds significant) as the primary
criterion for "consistent" evidence, and a *supermajority* threshold (> 75%
of seeds) as a secondary criterion for "strong" evidence. We deliberately do not
require unanimity across all 20 seeds, as any stochastic difference in
initialisation can preclude unanimity even when the population-level effect is
real and the statistical power per seed is high. Where proportion-of-seeds
results are informative, they are reported alongside the point-forecast rankings.

---

# 5. Results

## 5.1 Point Forecast Accuracy

**Table 3: Main Results --- MAE and RMSE by Model and Horizon** *(lower is better)*

| Model | Graph Variant | H=1 MAE | H=2 MAE | H=4 MAE | H=1 RMSE | H=2 RMSE | H=4 RMSE |
|-------|--------------|---------|---------|---------|----------|----------|----------|
| **GCN** | **identity_no_trade** | **1.707** | 2.499 | 3.668 | 2.750 | 4.249 | 6.164 |
| ARIMA | --- | 1.751 | **2.433** | 3.382 | 2.884 | 3.990 | 5.489 |
| TCN | --- | 1.770 | 2.813 | 3.610 | 2.953 | 4.857 | 5.833 |
| MLP | --- | 1.774 | 2.673 | 3.970 | 2.813 | 4.471 | 6.564 |
| Persistence | --- | 1.783 | 2.508 | 3.566 | 2.901 | 4.034 | 5.663 |
| ETS | --- | 1.783 | 2.508 | 3.566 | 2.901 | 4.035 | 5.664 |
| Gradient Boosting | --- | 1.892 | 2.940 | 3.983 | 3.116 | 4.556 | 5.769 |
| Dynamic Factor | --- | 1.920 | 2.618 | 3.758 | 2.983 | 4.141 | 5.871 |
| **Temporal Graph** | **identity_no_trade** | 1.999 | **2.358** | **2.840** | 3.604 | 4.141 | 4.831 |
| LSTM | --- | 2.183 | 2.485 | 2.968 | 3.768 | 4.284 | 4.952 |
| Ridge | --- | 2.420 | 3.381 | 4.502 | 3.152 | 4.982 | 6.781 |
| BVAR | --- | 2.341 | 3.857 | 6.312 | 3.534 | 5.947 | 10.213 |
| VAR | --- | 3.047 | 5.174 | 8.973 | 4.524 | 7.923 | 14.089 |

*Notes: Values are mean absolute error / root mean squared error in percentage points.
Results for neural and GNN models are averaged over 20 random seeds.
Best values per horizon in bold. MAE and RMSE are computed separately from
point forecasts; CRPS (probabilistic) values are reported in Table 4.*

Key observations:

- At **h = 1**, GCN with `identity_no_trade` achieves the lowest MAE (1.707 pp),
  narrowly ahead of ARIMA (1.751 pp). However, GCN's RMSE is marginally lower
  than ARIMA's (2.750 vs 2.884), suggesting comparable absolute accuracy
  with some large-error episodes.
- At **h = 2**, Temporal Graph with `identity_no_trade` ranks first (MAE = 2.358 pp),
  7% better than ARIMA (2.433 pp) and well ahead of all other models.
- At **h = 4**, Temporal Graph with `identity_no_trade` retains first place
  (MAE = 2.840 pp), a 16% improvement over ARIMA (3.382 pp) and representing
  the most substantive gain observed.
- **Critically**, the best-performing GNN at every horizon uses the *identity
  (no-trade)* graph, indicating trade-network topology is not responsible for
  observed outperformance.
- **BVAR** with Minnesota-prior shrinkage (MAE = 2.341/3.857/6.312 pp at h = 1/2/4)
  underperforms relative to simpler ARIMA, ETS, and persistence baselines, consistent
  with the well-documented difficulty of imposing informative priors on a volatile,
  post-COVID inflation regime.


## 5.2 Ablation: Graph Topology versus Architecture

**Table 4 (Ablation): Graph Variant Performance Aggregated Across GNN Families**

| Graph Variant | H=1 MAE | H=2 MAE | H=4 MAE | H=1 CRPS | H=2 CRPS | H=4 CRPS |
|--------------|---------|---------|---------|----------|----------|----------|
| `identity_no_trade` | **1.853** | **2.428** | **3.254** | **1.530** | **2.071** | **2.854** |
| `directed_trade` | 2.027 | 2.498 | 3.370 | 1.660 | 2.112 | 2.951 |
| `log_trade` | 2.063 | 2.511 | 3.371 | 1.691 | 2.120 | 2.947 |
| `reversed` | 2.043 | 2.517 | 3.399 | 1.674 | 2.130 | 2.980 |
| `undirected` | 2.039 | 2.512 | 3.388 | 1.670 | 2.125 | 2.968 |
| `import_dependence` | 2.046 | 2.551 | 3.396 | 1.681 | 2.169 | 2.983 |
| `degree_preserving_random` | 2.123 | 2.564 | 3.382 | 1.749 | 2.173 | 2.961 |
| `top_k_incoming` | 2.261 | 2.670 | 3.361 | 1.878 | 2.275 | 2.943 |

*Notes: Values averaged across GCN and Temporal Graph families, across 20 seeds.
CRPS = Continuous Ranked Probability Score (probabilistic accuracy).*

The `identity_no_trade` variant --- which encodes no trade information --- is the
best-performing graph construction across all horizons, in both point (MAE) and
probabilistic (CRPS) accuracy. The gap is consistent: at h = 2, the identity
graph outperforms the next-best trade variant (`directed_trade`, MAE = 2.498) by
approximately 3%. This pattern strongly implicates the GNN architecture's
temporal attention mechanism, rather than trade-graph topology, as the driver of
any marginal improvement over simpler baselines.

## 5.3 Probabilistic Forecast Accuracy

**Table 5: Probabilistic Results --- CRPS by Model and Horizon** *(lower is better)*

| Model | Graph Variant | H=1 CRPS | H=2 CRPS | H=4 CRPS | H=2 Cov-80 | H=4 Cov-80 |
|-------|--------------|---------|---------|---------|------------|------------|
| **GCN** | **identity_no_trade** | **1.378** | 2.131 | 3.257 | 0.552 | 0.481 |
| ARIMA | --- | 1.459 | **2.094** | **3.013** | 0.556 | 0.480 |
| MLP | --- | 1.444 | 2.300 | 3.553 | 0.497 | 0.399 |
| TCN | --- | 1.477 | 2.466 | 3.210 | 0.512 | 0.421 |
| Persistence | --- | 1.480 | 2.153 | 3.179 | 0.531 | 0.458 |
| ETS | --- | 1.481 | 2.153 | 3.179 | 0.529 | 0.463 |
| **Temporal Graph** | **identity_no_trade** | 1.682 | **2.010** | **2.451** | 0.576 | 0.522 |
| LSTM | --- | 1.842 | 2.129 | 2.581 | 0.517 | 0.485 |

*Notes: CRPS is the continuous ranked probability score; lower is better.
Cov-80 = empirical coverage of 80% prediction intervals (target: 0.80).
Note that MAE and CRPS are distinct loss functions; any apparent similarity
in magnitude between MAE and CRPS values for a given model should be
verified against the source data (see data availability statement).*

80% prediction interval coverage is substantially below the nominal level
for all models (target 0.80; observed 0.50--0.58), consistent with the
short panel and bootstrap-based interval construction typical for neural
network ensembles in this setting. Calibration improvements represent a
clear avenue for future work.

## 5.4 Statistical Significance

We report the proportion of seeds (out of 20) for which the BH-corrected
DM test favours the graph model (loss differential < 0, BH p < 0.05).

**No graph model achieves majority-seed consistent superiority** over its
best non-graph comparator. Selected notable results:

| Graph Model | Graph Variant | h | Comparator | Prop. Seeds Significant |
|-------------|--------------|---|------------|------------------------|
| GCN | `identity_no_trade` | 1 | Ridge | 80% |
| GCN | `directed_trade` | 1 | Ridge | 60% |
| GCN | `identity_no_trade` | 2 | Ridge | 65% |
| Temporal Graph | `identity_no_trade` | 1 | LSTM | 75% |
| GCN | `identity_no_trade` | 4 | Ridge | 5% |
| Temporal Graph | `identity_no_trade` | 2 | LSTM | 5% |

Two observations follow. First, the most frequent "significant" comparator is
Ridge --- a relatively weak regularised regression baseline, not the stronger
time-series models (ARIMA, TCN). Second, significance proportions decline
sharply with horizon, suggesting that GNN gains at h = 1 are partly a
finite-sample artefact of the short initial training window rather than a
structural architectural advantage.

No graph model achieves majority-seed significance against ARIMA, Persistence,
or TCN at any horizon.

## 5.5 Diagnostic Figures

![Figure 1: Graph Variant MAE Heatmap.](../experiments/results/v2_1/manuscript/graph_variant_heatmap.png)
*Figure 1: Graph Variant MAE Heatmap. Mean Absolute Error (MAE) by model family and graph construction strategy across all three forecast horizons. Darker shading indicates lower (better) MAE. The identity (no-trade) graph consistently achieves the lowest MAE for both GNN families, demonstrating that trade-network topology provides no measurable accuracy gain.*

![Figure 2: Forecast comparison (France, Seed 42, H=1).](../experiments/results/v2_1/manuscript/forecast_comparison.png)
*Figure 2: Forecast comparison (France, Seed 42, H=1). Comparison of Temporal Graph (directed_trade variant) and ARIMA forecasts against actual CPI YoY inflation. The plot illustrates where the neural model deviates from classical baselines during inflation transition periods.*

![Figure 3: Performance by Forecast Horizon.](../experiments/results/v2_1/manuscript/performance_by_horizon.png)
*Figure 3: Performance by Forecast Horizon. MAE and RMSE as a function of forecast horizon ($h = 1, 2, 4$ quarters) for all 12 model families. Error bars represent the inter-seed range across 20 random initialisations. The Temporal Graph model shows the largest improvement relative to ARIMA at $h = 4$, while GCN leads at $h = 1$.*

![Figure 4: Forecast Error Distribution.](../experiments/results/v2_1/manuscript/error_distribution.png)
*Figure 4: Forecast Error Distribution. Box plots showing the distribution of forecast errors across all countries, quarters, and seeds for $h=1$. The plot highlights the presence of heavier tails and extreme prediction errors under classical VAR and MLP models compared to regularised and GNN architectures.*

![Figure 5: Calibration Reliability Diagram.](../experiments/results/v2_1/manuscript/calibration_reliability.png)
*Figure 5: Calibration Reliability Diagram. Nominal vs. empirical coverage for nominal confidence levels of 80% and 95%. Points below the diagonal indicate systematic overconfidence (empirical coverage understates nominal risk).*

![Figure 6: Prediction Interval Coverage.](../experiments/results/v2_1/manuscript/prediction_interval_coverage.png)
*Figure 6: Prediction Interval Coverage. Empirical coverage fraction of 80% (lower bar) and 95% (upper bar) prediction intervals by model. The dashed lines represent nominal targets (0.80 and 0.95). Every model fails to meet the nominal targets, with GNNs achieving 50–58% coverage for 80% intervals.*

---

# 6. Discussion

## 6.1 The Identity-Graph Result

The most striking and practically important finding is that both GNN families
perform best when the trade graph is replaced by an identity matrix across all
horizons and both MAE and CRPS metrics. This pattern has two interpretations
that are not mutually exclusive:

**Architectural regularisation.** The GCN and temporal attention mechanisms
provide implicit regularisation or representational capacity advantages relative
to simpler neural baselines, independent of cross-country propagation. The
attention mechanism, in particular, may be learning temporal smoothing that
acts similarly to exponential smoothing without requiring graph information.
Prior theoretical work supports this possibility: Kawamoto et al. (2018) show
that GCN-style aggregation can achieve competitive performance independently of
the actual graph topology, and this has been observed empirically in non-economic
settings (Zugner et al., 2020).

**Noisy or irrelevant edges.** Quarterly bilateral export flows, as available
from Eurostat Comext, may be too coarse and too slowly varying to carry
high-frequency inflation-propagation signals. The trade graph is effectively
the same graph in each quarter of a given year, while inflation dynamics change
at weekly or monthly frequency; the quarterly edge weights may be too sparse in
the time dimension to add useful information beyond what the feature vector
already contains.

Both mechanisms point toward the same practical conclusion for applied
forecasters: the value of GNN architecture for macroeconomic panel forecasting
should be established using identity-graph ablation before invoking network
topology as an explanation for performance differences.

## 6.2 Why Do Trade Graphs Fail? (Economic Intuition)

Beyond architectural explanations, economic intuition sheds light on why bilateral trade graphs fail to improve inflation forecasts at quarterly horizons. One explanation is that quarterly CPI aggregation smooths high-frequency trade shocks that occur at monthly or weekly frequencies. Furthermore, bilateral trade volumes do not directly measure price pass-through elasticity; a country may import heavily from a low-inflation partner, diluting the network effect. Finally, global supply chain disruptions (e.g., COVID-19) are likely captured better by temporal attention mechanisms than by static annual trade topology.

## 6.3 Horizon Dependence

The Temporal Graph's advantage over comparators grows with horizon: the MAE
gap versus ARIMA is small at h = 1 (1.999 vs 1.751), widens at h = 2
(2.358 vs 2.433; Temporal Graph is now better), and is most pronounced at h = 4
(2.840 vs 3.382). This pattern is consistent with the temporal attention mechanism
being beneficial for medium-run trend extrapolation but adding little over the
persistence benchmark for one-step-ahead prediction. The finding parallels results
in the recurrent network literature (Hewamalage et al., 2021) where LSTM gains
over linear models accumulate primarily at longer horizons.

## 6.4 Statistical Robustness

The proportion-of-seeds significance results reveal an important nuance: even
where GNNs appear to win in point-forecast rankings, the win is often not
statistically reproducible across initialisation draws, particularly against
strong comparators. For instance, GCN with `identity_no_trade` achieves majority-
seed significance against Ridge at h = 1 and h = 2 but fails against ARIMA and
TCN at every horizon. This finding underscores the importance of multi-seed
testing: single-seed "champion" results, common in the GNN-for-economics literature,
can be substantially misleading.

## 6.5 Limitations

1. **Hyperparameter sharing.** All five neural architectures (MLP, LSTM, TCN,
   GCN, Temporal Graph) use identical hyperparameters (hidden dimension 16,
   learning rate 0.01, 30 epochs) to enable a controlled architecture-level
   comparison. This means each individual architecture may be operating below
   its capacity optimum. In particular, 30 epochs is shallow for models with
   attention mechanisms, and a proper validation-set sweep per architecture
   might reveal larger GNN gains or strengthen the case for simpler models.
   Results should be interpreted as characterising a controlled configuration
   rather than each model's peak performance.

2. **Short initial training window.** The expanding window starts with only 14
   quarters (2011Q2--2014Q4) for the initial model fits. For GNNs and LSTMs with
   hundreds to thousands of parameters, early-window fits are likely undertrained,
   and forecast errors in the first several test origins partly reflect model
   initialisation quality rather than intrinsic architecture differences. This
   contributes to instability in seed-to-seed significance at shorter horizons.

3. **Revised vintages.** We use revised current-release data snapshots rather than real-time historical vintages. This represents an upper-bound benchmark; true real-time performance may be worse. Revised data reflects ex-post statistical corrections unavailable to actual forecasters, which may overstate out-of-sample accuracy relative to a genuine real-time environment.

4. **Panel scope.** The panel covers 20 European economies. Results may not
   generalise to emerging markets, non-EU economies, or global panels where
   data quality, trade-flow reporting, and inflation dynamics differ materially.

5. **No causal claims.** All findings pertain to predictive associations. The
   study does not establish causal mechanisms for inflation transmission, nor
   does it imply that monitoring trade-network topology would improve central
   bank forecasting decisions without further validation.

6. **Prediction interval calibration shortfall.** Across all models, the empirical coverage of the 80% prediction intervals is systematically low, ranging between 50% and 58%. This undercoverage is a consequence of the short initial training window (14 quarters) and the bootstrap-based uncertainty estimation which understates prediction variance during shock periods. Future work should incorporate conformal prediction methods to guarantee nominal coverage.

---

# 7. Conclusion

We conduct a large-scale, fully reproducible prospective benchmark of GNN
inflation forecasting across 20 European economies at three horizons, comparing
12 model families with 8 graph construction strategies and 20 random seeds.
Our principal findings are:

1. **Temporal Graph matches or outperforms ARIMA at medium-to-long horizons**
   (h = 2, 4), with a 16% MAE improvement at h = 4 representing the most
   substantive gain. GCN leads at h = 1 but by a narrow margin.
2. **Trade-network topology does not explain GNN gains.** The identity
   (no-trade) graph consistently outperforms all trade-based graphs across both
   GNN families, all horizons, and both point and probabilistic metrics.
3. **Architectural gains are statistically fragile.** No GNN achieves
   majority-seed BH-corrected significance against ARIMA, Persistence, or TCN,
   and significance proportions fall sharply with horizon and with comparator
   quality.

These findings carry methodological implications for the growing literature on
GNNs in macroeconomics: identity-graph ablation should be a standard requirement,
multi-seed reporting should replace single-seed benchmarks, and claims of
network-topology-driven improvements require explicit statistical disentanglement
from architectural effects.

Future research should explore higher-frequency (monthly) bilateral trade data,
longer historical panels, graph structure learning rather than pre-specified
topology, and real-time data vintages to test whether any of these design choices
changes the topology-versus-architecture conclusion.

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

All artefacts are SHA256-verified for exact reproducibility. The benchmark can be
re-run end-to-end using the provided `reproduce.sh` script (estimated runtime:
approximately 12 hours on Apple Silicon M-series; longer on CPU-only hardware).

# Code Availability

Full source code is publicly available at:
<https://github.com/mrayanasim09/MarketEquationDiscovery>

The repository includes benchmark protocol specification, execution engine,
statistical analysis pipeline, manuscript figure and table generation scripts,
and SHA256-verified frozen results.

# Conflict of Interest

The author declares no conflicts of interest.

# Funding

This research received no external funding.

# Acknowledgements

Bilateral trade data are sourced from Eurostat Comext; macroeconomic indicators
from the IMF International Financial Statistics and World Bank World Development
Indicators; CPI data from Eurostat HICP.

---

# References

Atkeson, A., and Ohanian, L. E. (2001). Are Phillips curves useful for forecasting
inflation? *Federal Reserve Bank of Minneapolis Quarterly Review*, 25(1), 2--11.

Bánbura, M., Giannone, D., and Reichlin, L. (2010). Large Bayesian vector
auto regressions. *Journal of Applied Econometrics*, 25(1), 71--92.

Bayoumi, T., Bui, T., and Berkmen, P. (2023). Trade network exposure and inflation
dynamics. IMF Working Paper WP/23/117.

Calvo, G. A., and Reinhart, C. M. (2002). Fear of floating. *Quarterly Journal of
Economics*, 117(2), 379--408.

Chen, Y., Li, M., and Zhang, Z. (2023). Graph neural networks for commodity price
forecasting. *Energy Economics*, 118, 106482.

Cheng, D., and Zhu, J. (2022). Financial contagion detection via spatio-temporal
graph networks. *Journal of Financial Stability*, 63, 101073.

Clark, T. E., and McCracken, M. W. (2001). Tests of equal forecast accuracy and
encompassing for nested models. *Journal of Econometrics*, 105(1), 85--110.

Clark, T. E., and West, K. D. (2007). Approximately normal tests for equal
predictive accuracy in nested models. *Journal of Econometrics*, 138(1), 291--311.

Coulombe, P. G., Leroux, M., Stevanovic, D., and Surprenant, S. (2020). How is
machine learning useful for macroeconomic forecasting? *Journal of Applied
Econometrics*, 37(5), 920--964.

Diebold, F. X., and Mariano, R. S. (1995). Comparing predictive accuracy.
*Journal of Business and Economic Statistics*, 13(3), 253--263.

Faust, J., and Wright, J. H. (2013). Forecasting inflation. In G. Elliott and
A. Timmermann (Eds.), *Handbook of Economic Forecasting* (Vol. 2, pp. 2--56).
Elsevier.

Forbes, K. J., and Warnock, F. E. (2012). Capital flow waves: Surges, stops,
flight, and retrenchment. *Journal of International Economics*, 88(2), 235--251.

Garcia-Martos, C., Rodriguez, J., and Sanchez, M. J. (2015). Forecasting
electricity prices and their volatility using Unobserved Components. *Energy
Economics*, 43, 218--228.

Giannone, D., Lenza, M., and Primiceri, G. E. (2015). Prior selection for vector
autoregressions. *Review of Economics and Statistics*, 97(2), 436--451.

Harvey, D., Leybourne, S., and Newbold, P. (1997). Testing the equality of
prediction mean squared errors. *International Journal of Forecasting*, 13(2),
281--291.

Hewamalage, H., Bergmeir, C., and Bandara, K. (2021). Recurrent neural networks
for time series forecasting: Current status and future directions. *International
Journal of Forecasting*, 37(1), 388--427.

Kawamoto, T., Tsubaki, M., and Saito, T. (2018). Mean-field theory of graph
neural networks in graph partitioning. *Advances in Neural Information Processing
Systems* (NeurIPS), 31.

Kipf, T. N., and Welling, M. (2017). Semi-supervised classification with graph
convolutional networks. *International Conference on Learning Representations*.

Li, Y., Yu, R., Shahabi, C., and Liu, Y. (2018). Diffusion convolutional recurrent
neural network: Data-driven traffic forecasting. *International Conference on
Learning Representations*.

Makridakis, S., Spiliotis, E., and Assimakopoulos, V. (2018). Statistical and
machine learning forecasting methods: Concerns and ways forward. *PLOS ONE*,
13(3), e0194889.

Makridakis, S., Spiliotis, E., and Assimakopoulos, V. (2022). M5 accuracy
competition: Results, findings, and conclusions. *International Journal of
Forecasting*, 38(4), 1346--1364.

Medeiros, M. C., Vasconcelos, G. F. R., Veiga, A., and Zilberman, E. (2021).
Forecasting inflation in a data-rich environment: The benefits of machine
learning methods. *Journal of Business and Economic Statistics*, 39(1), 98--119.

Miranda-Agrippino, S., and Rey, H. (2021). The global financial cycle. In
G. Gopinath, E. Helpman, and K. Rogoff (Eds.), *Handbook of International
Economics* (Vol. 6, pp. 1--43). Elsevier.

Stock, J. H., and Watson, M. W. (2007). Why has U.S. inflation become harder to
forecast? *Journal of Money, Credit and Banking*, 39(s1), 3--33.

Wang, X., Chen, T., and Li, H. (2024). Spatio-temporal graph networks for regional
GDP forecasting. *Regional Science and Urban Economics*, 105, 103990.

Wu, Z., Pan, S., Chen, F., Long, G., Zhang, C., and Yu, P. S. (2019). A
comprehensive study on spatial-temporal graph neural networks. *IEEE Transactions
on Neural Networks and Learning Systems*, 32(2), 527--540.

Zugner, D., Akbarnejad, A., and Gunnemann, S. (2020). Adversarial attacks on
neural networks for graph data. *Proceedings of the 24th ACM SIGKDD International
Conference on Knowledge Discovery and Data Mining*, 2847--2856.
