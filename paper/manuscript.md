---
title: "Temporal Architecture or Trade Topology? A Prospective Ablation Study of Spatio-Temporal Graph Neural Networks for Quarterly CPI Inflation Forecasting"
author:
  - name: Anonymized Author(s)
    affiliation: Anonymized Institution
date: "2026-07-24"
abstract: |
  This study presents a controlled, prospective ablation benchmark evaluating whether bilateral trade-network topology provides incremental out-of-sample forecast accuracy for quarterly CPI inflation (year-over-year) across 20 European Union economies over the period 2017Q1–2025Q3. The scope is intentionally bounded: a quarterly panel frequency, a restricted covariate set of four publicly available, real-time-computable variables, and a post-2011 sample reflecting modern Eurostat data standards. We compare 12 model families — spanning classical time-series methods, regularised regression, three graph-free neural architectures (MLP, LSTM, TCN), and two graph neural network (GNN) families (GCN, Temporal Graph) — at horizons $h = 1, 2, 4$ quarters, under a strictly prospective expanding-window design. The key ablation contrasts eight trade-graph constructions against an *identity (no-trade)* graph: a control that isolates the GNN architecture from any cross-country topological signal. Statistical significance is assessed via Harvey–Leybourne–Newbold corrected Diebold–Mariano tests with Bartlett HAC weighting, moving-block bootstrap confidence intervals, and Benjamini–Hochberg FDR correction across 20 random seeds; results are reported as the proportion of seeds achieving BH-corrected significance. Within this scope, the identity (no-trade) graph consistently achieves lower point-forecast error than all trade-based graph variants across both GNN families and all three horizons. No trade-graph model achieves majority-seed BH-corrected Diebold–Mariano superiority over its best formal parametric comparator. These findings indicate that, under the conditions studied, measurable forecast accuracy gains from GNN architectures originate from temporal recurrence rather than cross-country trade-network topology. Whether this conclusion generalises to monthly frequencies, longer panels, richer covariate sets, or non-European economies is an open empirical question explicitly outside the scope of this study.
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

Inflation forecasting is a central challenge for central banks, fiscal authorities, and international organisations. Standard univariate and small-system models treat each economy in isolation, yet modern supply chains are highly integrated: commodity price changes and supply shocks propagate through bilateral trade linkages to trading partners, potentially generating inflation spillovers that single-country models cannot represent explicitly. The post-2021 inflation surge — driven partly by pandemic-era supply disruptions and subsequent energy-price co-movement across European Union economies — intensified interest in panel-based, network-aware forecasting frameworks.

Graph Neural Networks (GNNs) offer a natural representation for this setting. By encoding countries as nodes and bilateral trade flows as directed, weighted edges, GNNs are architecturally capable of propagating inflation signals along the trade network and producing cross-country forecasts that reflect trade-weighted exposures. However, the empirical evidence on whether graph topology *per se* improves out-of-sample forecast accuracy remains sparse. Most existing studies either apply GNNs to financial or synthetic graphs, or benchmark GNN architectures against weaker baselines without isolating the contribution of the *architecture* from the contribution of the *graph information* (Kawamoto et al., 2018; Zügner et al., 2020).

**The present study is deliberately narrow in scope.** We evaluate a specific question: *within a quarterly European panel setting using four publicly available covariates, does the addition of trade-network topology to a GNN architecture improve out-of-sample CPI forecast accuracy relative to the same architecture applied to an identity (no-trade) graph?* We do not claim to evaluate GNNs in general, trade networks in general, or inflation forecasting for non-EU or high-frequency settings.

This paper addresses three questions:

1. **Do GNNs produce lower forecast errors than classical benchmarks** (ARIMA, ETS, BVAR, ridge regression, gradient boosting) in prospective quarterly CPI forecasting for EU economies, under identical evaluation conditions?
2. **Does trade-network topology provide incremental predictive value** beyond what a graph-agnostic GNN architecture provides, as revealed by comparison against an identity (no-trade) graph baseline?
3. **Are any observed performance differences statistically reproducible** across 20 random seeds after Benjamini–Hochberg FDR correction for multiple comparisons?

**Boundary conditions** that qualify every finding in this paper: (i) 20 EU member states; (ii) quarterly frequency; (iii) post-2011Q2 sample; (iv) four-variable covariate set (own CPI, energy price index, CPI volatility, trade exposure); (v) initial training window of 14 quarters. Findings under materially different design choices may differ.

Our contributions are:

- A fully reproducible, prospective expanding-window benchmark covering 12 model families, 8 graph variants, 3 horizons, and 20 seeds — totalling 39,188 model evaluations and 783,760 forecast rows.
- The first systematic ablation of GNN trade-graph topology against an identity-graph control in a macroeconomic panel forecasting context, disentangling architectural from topological sources of forecast accuracy.
- A BVAR baseline with Minnesota-prior shrinkage, situating GNN performance relative to the standard Bayesian multivariate benchmark used in institutional forecasting (Bánbura et al., 2010; Giannone et al., 2015).
- A rigorous multi-seed statistical testing framework combining Harvey–Leybourne–Newbold DM tests, Bartlett HAC weighting, moving-block bootstrap confidence intervals, and Benjamini–Hochberg FDR correction.
- Publicly available code, data, and SHA256-verified frozen results (Makridakis et al., 2022).

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

## 4.0 Scope of the Feature Set and Sample Boundary

The predictor set used in this study — comprising own-lagged CPI (YoY), an energy price index, a rolling CPI volatility measure, and import-share trade exposure — is intentionally parsimonious. This choice reflects three design constraints.

*First, real-time availability.* At quarterly frequency, many standard macroeconomic drivers of inflation — output gaps, unit labour costs, wage growth series, and commodity price sub-indices — are subject to substantial real-time revisions and publication lags. Including them from revised releases would introduce look-ahead bias by giving models access to information unavailable to a genuine real-time forecaster at the time of each forecast origin (Stark and Croushore, 2002). The four variables retained are published with minimal revision and are available from Eurostat within one quarter of the reference period.

*Second, the ablation objective.* The paper's primary research question is whether trade-network topology provides incremental predictive value beyond what the GNN architecture provides when applied to the same feature set. This topological ablation is cleanest when the feature set is held constant across all model families: any additional covariates that interact differently with GNN aggregation versus linear models would confound the architectural comparison. The identity-graph control isolates topology from architecture only when the feature set is common to all models.

*Third, cross-country harmonisation.* The panel covers 20 EU member states with heterogeneous national statistical systems. Variables that are comparable and consistently defined across all 20 economies at quarterly frequency are substantially fewer than those available for major economies individually. Restricting to Eurostat-sourced variables ensures measurement homogeneity, which is particularly important for the GNN models whose graph convolution aggregates cross-country feature vectors.

*The post-2011Q2 sample boundary* reflects the availability of sufficiently complete and harmonised bilateral trade flows from Eurostat Comext at quarterly frequency. Prior to 2011, missing values and structural breaks in the Comext series for several smaller EU member states would require non-trivial imputation that could itself introduce bias. We therefore adopt 2011Q2 as a conservative, data-driven starting point.

We acknowledge that this restricted feature set means models lack access to drivers that are potentially important for inflation dynamics: output gaps, wage growth, import prices disaggregated by commodity class, and monetary policy variables. The findings — particularly the failure of trade-graph topology to improve forecast accuracy — are conditional on this specific feature set, and models with richer covariate sets may yield different topology-versus-architecture rankings.

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

## 5.4 Statistical Significance of Forecast Accuracy Differences

The analysis in this section distinguishes two conceptually separate exercises that must not be conflated.

**Part A — Heuristic point-forecast rankings (Table 3).** Table 3 ranks all 12 model families by MAE and RMSE. This ranking includes heuristic benchmarks such as *Persistence* (the naïve "no-change" forecast) alongside formally estimated models. These rankings are purely descriptive: they characterise the relative ordering of out-of-sample forecast errors on the test set. *Persistence is included here as a sanity check and descriptive comparator only.* Because Persistence is a degenerate, parameter-free rule, it does not admit a meaningful Diebold–Mariano test against any estimated parametric model in the sense of Diebold and Mariano (1995): the EPA test assumes both competing forecasts are produced by estimated models with finite-dimensional parameter vectors, a condition not met when one series is a fixed, non-estimated rule. Accordingly, *Persistence is explicitly excluded from all formal DM testing in Part B*.

**Part B — Formal DM tests (Table 6).** For each (GNN model, graph variant, horizon $h$) triple, we conduct pairwise Harvey–Leybourne–Newbold corrected DM tests against the following *formally estimated, parametric comparators*: ARIMA, ETS, BVAR, Ridge, TCN, LSTM, and MLP. The null hypothesis is equal predictive accuracy (EPA). For each seed $s \in \{1, \ldots, 20\}$ and comparator $c$, we compute the loss differential sequence:

$$d_t^{(s,c)} = |e_t^{\text{GNN},s}| - |e_t^{c}|, \quad t = 1, \ldots, T_{\text{test}}$$

The HLN-corrected DM statistic is:

$$\text{DM}^*_{s,c} = \frac{\bar{d}^{(s,c)}}{\sqrt{\hat{V}(\bar{d}^{(s,c)})}} \cdot \sqrt{\frac{T + 1 - 2h + h(h-1)/T}{T}}$$

where $\hat{V}(\bar{d})$ is Bartlett HAC-estimated with truncation lag $\lfloor T^{1/3} \rfloor$. Significance is assessed after Benjamini–Hochberg FDR correction at $q = 0.05$ across all simultaneous comparisons within each (model, variant, horizon) stratum.

Results are reported as the **proportion of seeds for which the BH-corrected $p$-value is below 0.05 and $\bar{d}^{(s,c)} < 0$** (i.e., the GNN produces lower expected loss). We adopt two interpretive thresholds: *majority* (> 50% of seeds) as the primary criterion for consistent evidence; *supermajority* (> 75% of seeds) as strong evidence.

**Table 6 — Selected DM Test Results** *(proportion of 20 seeds with BH-corrected p < 0.05 favouring GNN; Persistence excluded)*

| GNN Model | Graph Variant | $h$ | Comparator | Prop. Seeds Sig. | Inference |
|---|---|---|---|---|---|
| GCN | `identity_no_trade` | 1 | Ridge | 80% | **Strong** (> 75%) |
| Temporal Graph | `identity_no_trade` | 1 | LSTM | 75% | **Strong** (> 75%) |
| GCN | `identity_no_trade` | 2 | Ridge | 65% | Consistent (> 50%) |
| GCN | `directed_trade` | 1 | Ridge | 60% | Consistent (> 50%) |
| GCN | `identity_no_trade` | 1 | **ARIMA** | **35%** | **Not consistent** |
| GCN | `identity_no_trade` | 2 | **ARIMA** | **20%** | **Not consistent** |
| Temporal Graph | `identity_no_trade` | 4 | **ARIMA** | **15%** | **Not consistent** |
| GCN | `identity_no_trade` | 4 | Ridge | 5% | Not consistent |
| Temporal Graph | `identity_no_trade` | 2 | LSTM | 5% | Not consistent |

Two findings follow from Part B. First, the only comparator against which GNNs achieve consistent or strong significance is Ridge regression — a relatively weak regularised baseline. Against the stronger time-series models (ARIMA, BVAR, TCN), no GNN achieves majority-seed significance at any horizon. Second, DM significance rates decline sharply with horizon: all $h = 4$ results fall below the majority threshold, consistent with increasing finite-sample instability in expanding-window estimates at longer horizons.

The pattern in Part A (point rankings) and Part B (DM tests) is coherent: Temporal Graph achieves a lower point MAE than ARIMA at $h = 2$ and $h = 4$, but this difference is not statistically reproducible across the majority of seed draws — itself an informative finding about the precision of the estimated advantage.

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

## 6.1 The Identity-Graph Result: Two Hypotheses

The most striking within-scope finding is that both GNN families achieve lower out-of-sample MAE and CRPS with the identity (no-trade) graph than with any trade-graph construction — consistently across all horizons. This pattern is consistent with at least two distinct mechanistic hypotheses, **neither of which is directly tested in the present study**.

**Hypothesis 1 — Architectural regularisation.** Under this hypothesis, the GCN spatial aggregation operator and the Temporal Graph LSTM module provide implicit regularisation or representational capacity that improves single-country forecasts independently of cross-country propagation. When applied to the identity graph, the GNN reduces to a country-specific neural network; its forecast accuracy advantage would then reflect the temporal architecture alone. *This hypothesis is supported by analogy with theoretical results from Kawamoto et al. (2018), who show that GCN-style aggregation can achieve competitive performance independently of edge information under certain regularity conditions, and with empirical findings from Zügner et al. (2020). However, neither result directly implies the same mechanism operates in our macroeconomic setting.* Testing Hypothesis 1 rigorously would require architecturally equivalent models that differ only in the presence or absence of the spatial aggregation operator — a design not implemented here.

**Hypothesis 2 — Temporally coarse and informationally redundant trade edges.** Under this hypothesis, quarterly bilateral export volumes carry insufficient high-frequency variation to add predictive signal beyond what own-country lagged variables already contain. The trade graph changes slowly within a year, while inflation dynamics relevant to $h = 1$ forecasts operate at monthly or weekly frequency. Furthermore, bilateral trade volumes measure trade *quantities*, not price *pass-through elasticities*. *This hypothesis generates a testable prediction: repeating the ablation with monthly bilateral trade data should reduce the performance gap between identity and trade-graph variants if temporal coarseness is the operative mechanism.* This prediction is not tested here and is listed as a priority in §6.6.

Both hypotheses are consistent with the observed data. They are not mutually exclusive, and the present study cannot adjudicate between them. We report them as study-generated hypotheses to guide future work.

## 6.2 Why Do Trade Graphs Fail? (Economic Intuition)

Beyond architectural explanations, economic intuition suggests several hypotheses for why bilateral trade graphs fail to improve inflation forecasts at quarterly horizons in our panel. First, quarterly aggregation of CPI inflation smooths out higher-frequency shocks that might propagate more rapidly through supply chains. Second, bilateral trade volumes do not capture the elasticity of price pass-through, which is highly dependent on contract currencies, market structure, and input substitution possibilities (e.g., countries importing from a low-inflation partner may not experience significant spillovers). Third, global common factors (such as energy price shocks) may dominate local network-propagated shocks in determining headline inflation co-movement during major transition periods like the post-COVID regime. These economic interpretations are hypotheses for future research to evaluate rather than established mechanisms.

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

## 6.6 Future Research Directions

The hypotheses generated by this study point to five concrete research designs that could distinguish the proposed mechanisms:

1. **Monthly frequency replication.** The most direct test of Hypothesis 2 is to repeat the identical ablation using monthly bilateral trade flows and monthly CPI data. Eurostat provides monthly Comext data with approximately a two-month publication lag. If quarterly aggregation is the binding constraint, the performance gap between `identity_no_trade` and trade-graph variants should narrow materially at monthly frequency.
2. **Real-time data vintages.** The current study uses revised data releases. Replicating the benchmark with real-time vintages (ECB Real-Time Database; OECD MEI real-time releases) would (a) produce accuracy estimates that better reflect genuine forecaster information sets and (b) test whether the topology-versus-architecture ranking changes under data uncertainty.
3. **Graph structure learning.** Rather than pre-specifying eight graph constructions, one could learn the graph jointly with the forecasting model (Wu et al., 2020; Shang et al., 2021). If a learned graph substantially outperforms both trade-based and identity constructions, it would suggest that the *form* of the trade graph — rather than its presence or absence — is the binding constraint, lending support to a refined version of Hypothesis 2.
4. **Richer covariate sets.** Including output gaps, unit labour costs, and import price indices — with appropriate treatment of publication lags — would test whether the topology-versus-architecture conclusion is robust to a richer feature set or is specific to the minimal four-variable specification used here.
5. **Architecture-specific hyperparameter tuning.** The present study uses identical hyperparameters across all neural models for controlled comparability. A study that tunes each architecture independently — with proper validation-set search and longer training — would test whether the GNN advantage strengthens under optimal configuration.

## 6.7 Claim-to-Evidence Mapping

The following table maps every major claim made in this study to the exact empirical evidence presented in the paper, alongside relevant boundary conditions and caveats.

| Core Claim | Supporting Evidence in Paper | Boundary Conditions & Caveats |
|---|---|---|
| **1.** Within the studied scope, trade-network topology does not provide incremental out-of-sample forecast accuracy for EU CPI inflation relative to the same GNN architecture applied to an identity (no-trade) graph. | **Table 4** (ablation): `identity_no_trade` achieves the lowest MAE at all horizons for both GCN and Temporal Graph. Confirmed in both point (MAE, RMSE) and probabilistic (CRPS) accuracy. **Figure 1** (heatmap). | 20 EU economies; quarterly frequency; 4-variable covariate set; post-2011Q2; GCN and Temporal Graph architectures only. Does not generalise to monthly frequency, richer features, or non-EU panels. |
| **2.** No trade-graph GNN achieves statistically reproducible superiority — defined as majority-seed BH-corrected DM significance — over its best formal parametric comparator. | **Table 6** (DM tests): All comparisons against ARIMA, BVAR, ETS, TCN fail the > 50% seed threshold. Ridge comparisons achieve 60–80% at h=1–2 only. | Persistence is *excluded* from formal DM testing (degenerate rule, not an estimated model). Claims apply to MAE loss; RMSE-based DM may differ. ETS is a borderline case: it is estimated but closely related to Persistence in this dataset. |
| **3.** Temporal Graph achieves the lowest out-of-sample point MAE of any model at h=2 and h=4, using the identity (no-trade) graph, within this experimental setting. | **Table 3**: Temporal Graph (identity): MAE = 2.358 pp (h=2), 2.840 pp (h=4). Next best: ARIMA (2.433; 3.382 pp). **Figure 3** (horizon-MAE plot). | Statistical reproducibility is limited: majority-seed DM significance against ARIMA is not achieved at any horizon. Rankings reflect means over 20 seeds; per-seed rankings vary. |
| **4.** GNN accuracy gains, when present, are associated with the temporal architecture rather than cross-country trade-network propagation, based on the identity-graph ablation. | **Table 4**: `identity_no_trade` dominates all 7 trade-graph variants at every horizon. **Figure 1** (heatmap): pattern consistent across GCN and Temporal Graph. | This is an *associational* finding, not a causal identification. Two alternative hypotheses are proposed (architectural regularisation; temporally coarse edges) but not tested within this study. |
| **5.** BVAR with Minnesota-prior shrinkage does not improve out-of-sample forecast accuracy over ARIMA or ETS in this quarterly EU panel setting. | **Table 3**: BVAR MAE = 2.341/3.857/6.312 pp at h=1/2/4 vs ARIMA (1.751/2.433/3.382 pp). BVAR ranks 10th out of 12 overall. | Specific to Minnesota-prior BVAR, lag-1, on a 4-variable system. Prior not exhaustively tuned. Consistent with documented prior-selection challenges in volatile post-COVID regimes (Bánbura et al., 2010). Alternative priors or larger variable sets may yield different results. |

---

# 7. Conclusion

We conduct a large-scale, fully reproducible prospective evaluation of GNN-based inflation forecasting across 20 European economies at three horizons, comparing 12 model families under 8 trade-graph constructions and 20 random seeds. Every finding is conditional on the scope defined in Section 1 and the methodology described in Section 4.

Our principal findings are:

1. **Temporal Graph shows the largest absolute forecast accuracy gain at medium-to-long horizons.** Temporal Graph with the identity graph achieves MAE of 2.358 pp at $h = 2$ (ARIMA: 2.433 pp) and 2.840 pp at $h = 4$ (ARIMA: 3.382 pp), suggesting that temporal recurrence in the GNN architecture provides predictive value at longer horizons. GCN leads at $h = 1$ but by a narrow margin.
2. **Temporal architecture, not trade topology, is the consistent predictor of within-GNN rankings.** The identity (no-trade) graph achieves lower out-of-sample MAE than every trade-based graph variant at every horizon for both GNN families, in both point (MAE, RMSE) and probabilistic (CRPS) accuracy.
3. **Statistical reproducibility is limited, particularly against strong baselines.** Under BH-corrected DM testing across 20 seeds, no GNN achieves majority-seed significance against ARIMA, ETS, or TCN at any horizon. The most consistent GNN advantage is against Ridge regression. Significance rates decline sharply with horizon.

These findings carry methodological implications: identity-graph ablation should be a standard requirement in GNN macroeconomic forecasting papers; multi-seed reporting should replace single-seed benchmarks; and claims of topology-driven forecast improvement require explicit statistical disentanglement from architectural effects.

What this study does not establish is whether trade-network topology improves forecast accuracy at monthly frequency, with richer covariate sets, over longer historical samples, or for non-EU economies. We encourage future work to vary these design dimensions systematically.

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
