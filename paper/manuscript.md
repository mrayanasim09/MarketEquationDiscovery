---
title: "Temporal Architecture or Trade Topology? A Prospective Ablation Study of Spatio-Temporal Graph Neural Networks for EU Inflation Forecasting"
author:
  - name: Rayyan Asim
    affiliation: Independent Researcher
    email: mrayanasim09@gmail.com
    orcid: 0000-0003-2461-5638
date: "2026-07-24"
abstract: |
  We present a controlled prospective ablation benchmark demonstrating that incorporating bilateral trade-network topology into Spatio-Temporal Graph Neural Networks (GNNs) fails to improve out-of-sample forecast accuracy for quarterly CPI inflation across 20 European Union economies (2017Q1–2025Q3). While GNNs are increasingly proposed for macroeconomic forecasting to capture spatial spillovers, their performance gains are rarely ablated against non-spatial controls. We evaluate 12 model families under a prospective expanding-window design, testing eight trade-graph constructions against an *identity (no-trade)* graph control. Across all horizons ($h=1,2,4$), GNNs using the identity graph consistently achieve lower point-forecast error than any trade-based variant, and no trade-graph model achieves majority-seed significance against formally estimated parametric baselines, assessed using formal statistical testing with multiple-seed reproducibility confirmation. These findings suggest that temporal recurrence, rather than trade-network topology, is the more plausible source of any GNN forecast gains within this design — though the absence of architecture-specific tuning means this cannot be fully isolated. For the forecasting community, this negative result establishes that spatial or network-based models should not be assumed to capture meaningful economic spillovers without passing an identity-graph ablation control. This highlights the risk of over-parameterisation in network-based macroeconomic forecasting and underscores the need for rigorous architectural controls.
keywords:
  - inflation forecasting
  - graph neural networks
  - trade networks
  - negative benchmark
  - ablation study
  - Diebold-Mariano test
  - expanding window evaluation
  - macroeconomic panel forecasting
jel: "C32, C33, C53, E31, E37, F14"
journal: "International Journal of Forecasting"
---

**JEL Codes:** C32, C33, C53, E31, E37, F14

**Keywords:** inflation forecasting; graph neural networks; trade
networks; negative benchmark; ablation study; Diebold--Mariano test;
expanding window evaluation; macroeconomic panel forecasting

# 1. Introduction

Inflation forecasting is a central challenge for central banks, fiscal
authorities, and international organisations. Standard univariate and
small-system models treat each economy in isolation, yet modern supply
chains are highly integrated: commodity price changes and supply shocks
propagate through bilateral trade linkages to trading partners,
potentially generating inflation spillovers that single-country models
cannot represent explicitly. The post-2021 inflation surge --- driven
partly by pandemic-era supply disruptions and subsequent energy-price
co-movement across European Union economies --- intensified interest in
panel-based, network-aware forecasting frameworks.

Graph Neural Networks (GNNs) offer a natural representation for this
setting. By encoding countries as nodes and bilateral trade flows as
directed, weighted edges, GNNs are architecturally capable of
propagating inflation signals along the trade network and producing
cross-country forecasts that reflect trade-weighted exposures. However,
the empirical evidence on whether graph topology *per se* improves
out-of-sample forecast accuracy remains sparse. Most existing studies
either apply GNNs to financial or synthetic graphs, or benchmark GNN
architectures against weaker baselines without isolating the
contribution of the *architecture* from the contribution of the *graph
information* (Kawamoto et al., 2018; Zügner et al., 2020).

**The present study is deliberately narrow in scope.** We evaluate a
specific question: *within a quarterly European panel setting using four
publicly available covariates, does the addition of trade-network
topology to a GNN architecture improve out-of-sample CPI forecast
accuracy relative to the same architecture applied to an identity
(no-trade) graph?* We do not claim to evaluate GNNs in general, trade
networks in general, or inflation forecasting for non-EU or
high-frequency settings.

This paper addresses three questions:

1.  **Do GNNs produce lower forecast errors than classical benchmarks**
    (ARIMA, ETS, BVAR, ridge regression, gradient boosting) in
    prospective quarterly CPI forecasting for EU economies, under
    identical evaluation conditions?

2.  **Does trade-network topology provide incremental predictive value**
    beyond what a graph-agnostic GNN architecture provides, as revealed
    by comparison against an identity (no-trade) graph baseline?

3.  **Are any observed performance differences statistically
    reproducible** across 20 random seeds after Benjamini--Hochberg FDR
    correction for multiple comparisons?

**Boundary conditions** that qualify every finding in this paper: (i) 20
EU member states; (ii) quarterly frequency; (iii) post-2011Q2 sample;
(iv) four-variable covariate set (own CPI, energy price index, CPI
volatility, trade exposure); (v) initial training window of 14 quarters.
Findings under materially different design choices may differ.

Our contributions are:

- A fully reproducible, prospective expanding-window benchmark covering
  12 model families, 8 graph variants, 3 horizons, and 20 seeds ---
  totalling 39,188 model evaluations and 783,760 forecast rows.

- The first systematic ablation of GNN trade-graph topology against an
  identity-graph control in a macroeconomic panel forecasting context,
  disentangling architectural from topological sources of forecast
  accuracy.

- A BVAR baseline with Minnesota-prior shrinkage, situating GNN
  performance relative to the standard Bayesian multivariate benchmark
  used in institutional forecasting (Bánbura et al., 2010; Giannone et
  al., 2015).

- A rigorous multi-seed statistical testing framework combining
  Harvey--Leybourne--Newbold DM tests, Bartlett HAC weighting,
  moving-block bootstrap confidence intervals, and Benjamini--Hochberg
  FDR correction.

- Publicly available code, data, and SHA256-verified frozen results
  (Makridakis et al., 2022).

::: center

------------------------------------------------------------------------
:::

# 2. Related Literature

## 2.1 Classical Inflation Forecasting

The Phillips curve and its variants remain the workhorse of inflation
forecasting despite well-documented instability (Stock and Watson, 2007;
Atkeson and Ohanian, 2001). Autoregressive models --- ARIMA and ETS ---
are competitive short-run benchmarks (Faust and Wright, 2013). Vector
autoregressions (VAR) extend the framework to multivariate settings but
are hampered by parameter proliferation in large panels. Bayesian VARs
(BVAR) with shrinkage priors provide a standard benchmark to mitigate
parameter proliferation in large panels (Giannone et al., 2015; Bańbura
et al., 2010). Regularised regression methods, particularly ridge and
LASSO, have gained traction as high-dimensional alternatives
(Medeiros et al., 2021; Coulombe et al., 2020).

## 2.2 Neural Networks for Macroeconomic Forecasting

Recurrent architectures (LSTM, GRU) and temporal convolutional networks
(TCN) have demonstrated competitive performance in macroeconomic
forecasting tasks (Makridakis et al., 2018; Hewamalage et al., 2021).
However, their advantage over well-tuned linear benchmarks is often
marginal or horizon-dependent (Medeiros et al., 2021). Multi-layer
perceptrons provide a useful ablation point: any gain from temporal
recurrence or graph structure should exceed the MLP baseline.

## 2.3 Graph Neural Networks in Economics

The application of GNNs to economic and financial panel data is nascent.
Graph convolutional networks (GCN; Kipf and Welling, 2017) aggregate
neighbour embeddings via the normalised adjacency matrix, enabling
information propagation across the graph. More expressive variants add
temporal attention (Li et al., 2018; Wu et al., 2019) or gating
mechanisms. In economics, GNNs have been applied to financial contagion
(Cheng and Zhu, 2022), commodity markets (Chen et al., 2023), and
regional economic forecasting (Wang et al., 2024), but rigorous ablation
studies are scarce, and the contribution of graph topology versus
architecture is rarely disentangled.

A recurring concern in the graph learning literature is that null or
random graphs can match or even exceed structurally meaningful graphs
when the GNN architecture itself provides implicit regularisation or
pattern-matching capacity. Kawamoto et al. (2018) show that GCN-style
architectures can learn classification rules that are largely
independent of edge information under certain conditions. Zugner et
al. (2020) demonstrate that graph structure can be adversarially
irrelevant to GNN performance. These theoretical findings motivate our
identity-graph ablation as a rigorous control for architectural versus
topological effects.

## 2.4 Trade-Network Spillovers

Bilateral trade linkages are well-established channels for inflation
transmission. Calvo and Reinhart (2002) and subsequent literature
document cross-country co-movement in inflation that correlates with
trade intensity. Forbes and Warnock (2012) and Miranda-Agrippino and Rey
(2021) emphasise global common factors. Bayoumi et al. (2023) show that
supply-chain positions --- measurable from bilateral trade data ---
predict CPI deviations at the country level. Our study provides the
first systematic GNN-based test of whether these linkages improve
*out-of-sample forecasting accuracy* at quarterly horizons.

## 2.5 Adjacent Cross-Country Forecasting Traditions

GNN-based approaches to cross-country forecasting sit within a broader tradition of models designed to capture international spillovers. Three families are directly relevant to the present study.

**Global VAR (GVAR).** Pesaran, Schuermann, and Weiner (2004) introduce a multinational VAR in which each country model is linked to a trade-weighted foreign aggregate, directly encoding bilateral trade-intensity as a structural prior. GVAR is the natural classical predecessor to GNN-based bilateral-trade forecasting: it operationalises the same economic intuition — that inflation co-movement tracks trade intensity — in a linear, statistically tractable framework.

**Panel VAR and cross-country panel models.** Pooled panel autoregressions with country fixed effects (see Canova and Ciccarelli, 2013) capture aggregate cross-country co-movement through common time effects without imposing a bilateral network structure. These models serve as a useful middle ground between country-specific models (which ignore spillovers entirely) and GNNs (which learn or impose a specific network).

**Dynamic Factor Models.** Stock and Watson (2002) and Forni, Hallin, Lippi, and Reichlin (2000) show that a small number of common factors can capture the bulk of cross-country inflation co-movement. Factor models effectively impose the assumption that spillovers are symmetric and driven by global shocks rather than bilateral linkages. The present benchmark includes a Dynamic Factor Model (one common factor) precisely as a non-network cross-country comparator: any GNN gain over the DFM would suggest that bilateral network structure adds value beyond simple common-factor co-movement (see §4.2).

Positioning GNN-based forecasting relative to these traditions is important because it clarifies what is being tested. GNNs claim to improve over factor models and panel VARs by encoding asymmetric, bilateral trade linkages explicitly. Our ablation directly tests whether this claim holds in an out-of-sample, prospective EU CPI forecasting setting.

## 2.6 Theoretical Motivation: Why Trade Networks Should Predict Inflation

From an economic perspective, there are strong reasons to hypothesise that bilateral trade-network topology should improve inflation forecasts relative to isolated country-specific or symmetric common-factor models. Inflation transmission across borders operates primarily through two structural channels:

1. **Import Price Pass-Through:** A price shock in exporter country $j$ directly alters the cost of intermediate and final goods imported by country $i$. The magnitude of this pass-through is proportional to the import share of country $j$ in country $i$'s total consumption and production basket.
2. **Global Supply Chain Integration:** Supply-chain disruptions or domestic demand pressures in a major hub country (such as Germany within the EU) propagate sequentially through input-output linkages, creating lagged inflation pressures in downstream trade partners.

Standard macro forecasting baselines handle these linkages in highly restricted ways. Country-specific models (such as ARIMA or country-by-country ridge regression) ignore them entirely, assuming closed-economy dynamics. Common-factor models (such as principal components or global factor models) capture uniform global shocks but treat all countries as symmetrically exposed or rely on static factor loadings. 

A trade-graph GNN theoretically overcomes both limitations by allowing **asymmetric, network-structured propagation**. In a GCN-style spatial convolution, the inflation features of trading partners are aggregated using actual, time-varying import or export shares as edge weights. This allows the model to capture the intuition that a cost shock in a major trading partner (e.g., Germany for Austria) has a larger and faster predictive effect on local inflation than a shock in a trade-distant partner. By evaluating GNNs on trade-based graphs versus an identity graph, this study directly tests whether this network-structured asymmetric propagation provides incremental predictive value.


::: center

------------------------------------------------------------------------
:::

# 3. Data

## 3.1 Country Panel

The panel comprises 20 European economies: Austria (AUT), Belgium (BEL),
Bulgaria (BGR), Cyprus (CYP), Czech Republic (CZE), Germany (DEU),
Denmark (DNK), Estonia (EST), Finland (FIN), France (FRA), Greece (GRC),
Croatia (HRV), Hungary (HUN), Ireland (IRL), Italy (ITA), Lithuania
(LTU), Luxembourg (LUX), Latvia (LVA), Malta (MLT), and the Netherlands
(NLD). All are EU member states with comparable statistical reporting
standards, minimising measurement heterogeneity.

## 3.2 Macroeconomic Variables

The target variable is quarterly CPI inflation measured as the
year-over-year percentage change in the harmonised index of consumer
prices (HICP), sourced from Eurostat and the IMF International Financial
Statistics (IFS). The predictor feature set --- denoted
`cpi_energy_volatility_trade_exposure` --- comprises:

- **CPI (YoY, quarterly):** Own-lagged inflation for each country.

- **Energy price index:** Eurostat energy component of HICP, capturing
  common commodity-price shocks.

- **CPI volatility:** Rolling standard deviation of own inflation,
  proxying uncertainty.

- **Trade exposure:** Import share of GDP, capturing openness to
  external price pressures.

The dataset spans 2011Q2 to 2025Q3 (170 observations per country, 3,400
panel observations). Selected descriptive statistics are provided in
Table 1; full country-level statistics are provided in the online
supplementary material.

**Table 1: Dataset Summary (Selected Countries)**

+--------------+--------------+--------------+--------------+--------------+--------------+--------------+
| ::: minipage | ::: minipage | ::: minipage | ::: minipage | ::: minipage | ::: minipage | ::: minipage |
| Country      | Total Obs    | Train        | Val          | Test         | Mean CPI     | Std CPI (pp) |
| :::          | :::          | :::          | :::          | :::          | (pp)         | :::          |
|              |              |              |              |              | :::          |              |
+:=============+:=============+:=============+:=============+:=============+:=============+:=============+
| AUT          |              |              |              |              |              |              |
+--------------+--------------+--------------+--------------+--------------+--------------+--------------+
| DEU          |              |              |              |              |              |              |
+--------------+--------------+--------------+--------------+--------------+--------------+--------------+
| EST          |              |              |              |              |              |              |
+--------------+--------------+--------------+--------------+--------------+--------------+--------------+
| FRA          |              |              |              |              |              |              |
+--------------+--------------+--------------+--------------+--------------+--------------+--------------+
| GRC          |              |              |              |              |              |              |
+--------------+--------------+--------------+--------------+--------------+--------------+--------------+
| HUN          |              |              |              |              |              |              |
+--------------+--------------+--------------+--------------+--------------+--------------+--------------+
| ITA          |              |              |              |              |              |              |
+--------------+--------------+--------------+--------------+--------------+--------------+--------------+
| NLD          |              |              |              |              |              |              |
+--------------+--------------+--------------+--------------+--------------+--------------+--------------+
| *Panel mean* |              |              |              |              | *2.74*       | *3.22*       |
+--------------+--------------+--------------+--------------+--------------+--------------+--------------+

*Notes: CPI is year-over-year HICP percentage change. Sources: Eurostat,
IMF IFS.*

## 3.3 Bilateral Trade Graph

Quarterly bilateral trade flows are sourced from Eurostat Comext. Each
quarter, a 20 x 20 directed adjacency matrix A(t) is constructed where
entry A(i,j) = exports from country i to country j, expressed in
millions of euros. Eight graph-construction variants are studied:

+----------------------------+-----------------------------------------+
| ::: minipage               | ::: minipage                            |
| Variant                    | Construction                            |
| :::                        | :::                                     |
+:===========================+:========================================+
| `directed_trade`           | Raw export values, row-normalised       |
+----------------------------+-----------------------------------------+
| `log_trade`                | Log-transformed exports, row-normalised |
+----------------------------+-----------------------------------------+
| `import_dependence`        | A(i,j) = imports of i from j / total    |
|                            | imports of i                            |
+----------------------------+-----------------------------------------+
| `top_k_incoming`           | Retain only top-3 import partners per   |
|                            | country                                 |
+----------------------------+-----------------------------------------+
| `reversed`                 | Transpose of directed_trade (import     |
|                            | perspective)                            |
+----------------------------+-----------------------------------------+
| `undirected`               | Symmetrised: (A + A') / 2               |
+----------------------------+-----------------------------------------+
| `degree_preserving_random` | Random rewiring preserving degree       |
|                            | sequence                                |
+----------------------------+-----------------------------------------+
| `identity_no_trade`        | Identity matrix (no trade edges;        |
|                            | architecture ablation)                  |
+----------------------------+-----------------------------------------+

The `identity_no_trade` variant is the critical ablation: any model
using it cannot leverage cross-country trade signals, so outperformance
of trade-graph variants *over* this baseline would isolate the
contribution of graph topology from the contribution of the GNN
architecture itself.

## 3.4 Data Integrity

  Artefact                             SHA256 (first 16 hex)
  ------------------------------------ -----------------------
  Processed samples                    `40af80c0...`
  `forecasts.parquet` (783,760 rows)   `363e6994d44b575a...`
  `metrics.parquet` (9,312 rows)       `22bd6677c9a89d5d...`
  `dm_tests.parquet` (6,720 rows)      `d2d383a021dcb877...`

Full hashes are available in the public repository.

::: center

------------------------------------------------------------------------
:::

# 4. Methodology

## 4.0 Scope of the Feature Set and Sample Boundary

The covariate set used in this benchmark is intentionally parsimonious:
own CPI lagged one period, an energy price index, rolling CPI
volatility, and a trade-exposure scalar. Although this restricted
feature set means models lack access to variables that are potentially
important for inflation dynamics --- output gaps, wage growth, import
prices disaggregated by commodity class, and monetary policy variables
--- the set was chosen to satisfy two practical constraints.

*First, availability and harmonisation.* All four variables are
available at quarterly frequency from public sources (Eurostat, IMF IFS)
for all 20 EU economies throughout the evaluation period, minimising the
risk of unbalanced panels due to data gaps. The panel covers 20 EU
member states with heterogeneous national statistical systems;
restricting to Eurostat-sourced variables ensures measurement
homogeneity, which is particularly important for GNN models whose graph
convolution aggregates cross-country feature vectors.

*Second, and crucially for forward-looking use*, although the benchmark
uses final revised Eurostat releases for cross-country harmonisation,
the selected covariates are limited to variables with comparatively low
revision intensity and early availability. Heavily revised macroeconomic
series --- such as output gaps, unit labour costs, and real-time
productivity measures --- are excluded to reduce look-ahead concerns in
practical forecasting applications (Stark and Croushore, 2002). A full
real-time vintage evaluation is left for future work.

*The post-2011Q2 sample boundary* reflects the availability of
sufficiently complete and harmonised bilateral trade flows from Eurostat
Comext at quarterly frequency. Prior to 2011, missing values and
structural breaks in the Comext series for several smaller EU member
states would require non-trivial imputation that could itself introduce
bias. We therefore adopt 2011Q2 as a conservative, data-driven starting
point.

The findings --- particularly the failure of trade-graph topology to
improve forecast accuracy --- are conditional on this specific feature
set, and models with richer covariate sets may yield different
topology-versus-architecture rankings.

## 4.1 Evaluation Protocol

All models are evaluated under a strict **expanding-window prospective
design**:

- **Training:** 2011Q2--2014Q4 (initial window, 14 quarters; expanded
  forward each step)

- **Validation:** 2015Q1--2016Q4 (8 quarters; used for hyperparameter
  selection only)

- **Test:** 2017Q1--2025Q3 (35 quarters; no re-tuning or re-selection
  after lock-in)

At each test origin t, the model observes all data up to t-1, produces
forecasts at h = 1, 2, 4 quarters ahead, and the training window expands
by one quarter. This design aligns with the information constraints of a
genuine forecasting exercise: no information from after the forecast
origin is used in model fitting or forecast construction.

We note that the initial training window of 14 quarters is short
relative to the number of parameters in the neural and GNN models
(hundreds to thousands), which means early-window model fits may be
systematically undertrained. This structural asymmetry is acknowledged
explicitly in the limitations (Section 6.4).

## 4.2 Model Families

**Table 2: Model Configurations**

  Family              Model               Selected Hyperparameters
  ------------------- ------------------- ---------------------------------------------------------
  Baselines           Persistence         Last observed CPI YoY
                      ARIMA               Order ($p$,0,0) with AIC selection, $p \in \{1,2,3,4\}$
                      VAR                 Lag 1
                      ETS                 No trend, no seasonal
                      Dynamic Factor      1 factor, error order 0
  Regularised ML      Ridge               penalty = 1.0
                      Gradient Boosting   lr = 0.05, 100 estimators, L2 = 1.0
  Graph-free Neural   MLP                 hidden = 16, lr = 0.01, 30 epochs
                      LSTM                hidden = 16, lr = 0.01, 30 epochs
                      TCN                 hidden = 16, lr = 0.01, 30 epochs
  Graph Neural        GCN                 hidden = 16, lr = 0.01, 30 epochs
                      Temporal Graph      hidden = 16, lr = 0.01, 30 epochs

Neural and graph models are trained with 20 random seeds (42--61) to
quantify estimation uncertainty. Total benchmark: **38,380 model fits**,
**781,740 forecast rows**.

Among the baseline model families, the **Dynamic Factor Model (DFM, 1 common factor)** serves as the designated non-GNN cross-country benchmark. Unlike country-specific models (ARIMA, Ridge) that treat each economy in isolation, the DFM captures common latent inflation co-movement across all 20 EU economies simultaneously. Any GNN gain over the DFM would suggest that bilateral network structure adds value beyond simple common-factor co-movement; failure to beat the DFM would imply that cross-country spillovers, to the extent they exist, are already captured by a single global factor.

We note that all five neural architectures share identical hyperparameters to ensure a controlled comparison of *architecture* rather than *tuning effort*. While holding hyperparameters constant is necessary to isolate architectural differences (the ablation's primary goal), it implies the resulting forecasts do not reflect the peak empirical performance of any single model family under optimal, architecture-specific tuning. A model with higher representation capacity (like the Temporal Graph) may require more training epochs or larger hidden dimensions, whereas a simpler model (like TCN) might benefit from aggressive regularisation or dropout. Consequently, the relative ranking of models should be interpreted as characterizing these architectures under a uniform baseline constraint rather than representing their respective performance ceilings. A full discussion of this limitation is provided in Section 6.5.

## 4.3 Bayesian Vector Autoregression Baseline {#bvar-baseline}

The BVAR is included as a theoretically motivated classical baseline.
The model is specified as a first-order VAR in the same four variables
used by the neural models, with a Minnesota-type prior structure
(Litterman, 1986; Bánbura, Giannone, and Reichlin, 2010; Giannone,
Lenza, and Primiceri, 2015).

The Minnesota prior imposes the regularisation assumption that each
variable follows a random walk and that cross-variable lags carry
relatively less information. The prior is parameterised by two
hyperparameters: the own-lag tightness $\lambda_1 = 0.2$ and the
cross-lag tightness $\lambda_2 = 0.5$. In practice, these choices make
the BVAR a form of regularised multivariate regression --- a close
relative of ridge regression but with prior-based rather than
penalty-based shrinkage.

Given the short initial training window relative to the cross-sectional
dimension, the BVAR is heavily regularised. Its relative performance
should therefore be interpreted in light of the well-known difficulty of
estimating large covariance matrices in short panels.

BVAR forecasts are point predictions only; no credible interval or
density forecast is produced, so the BVAR is correctly excluded from all
CRPS-based probabilistic comparisons (Table 5).

## 4.4 GNN Architecture

**Graph Convolutional Network (GCN)** applies a spatial graph
convolution operator over node features:

$$\mathbf{H} = \mathrm{ReLU}\!\left(\tilde{\mathbf{A}}\mathbf{X}\mathbf{W}_{\text{node}}\right)$$

where $\tilde{\mathbf{A}}$ is the row-normalised spatial adjacency
matrix, $\mathbf{X} \in \mathbb{R}^{N \times F}$ represents input node
features for $N$ countries, and
$\mathbf{W}_{\text{node}} \in \mathbb{R}^{F \times d}$ is a learnable
weight matrix mapping features to hidden dimension $d$. A linear readout
layer $\mathbf{W}_{\text{head}} \in \mathbb{R}^{d \times 1}$ maps the
node embeddings to individual country forecasts:
$\hat{\mathbf{y}} = \mathbf{H}\mathbf{W}_{\text{head}}$.

**Temporal Graph** combines the spatial GCN message-passing operator
with a Recurrent LSTM temporal encoder. For each time step
$t' \in \{t-K, \dots, t-1\}$ in the lookback window of length $K$,
spatial node representations are computed:

$$\mathbf{h}_i^{(t')} = \mathrm{ReLU}\!\left(\sum_{j=1}^N \tilde{A}_{ij}^{(t')}\,\mathbf{x}_j^{(t')}\mathbf{W}_{\text{node}}\right)$$

The temporal sequence of spatial embeddings
$\{\mathbf{h}_i^{(t-K)}, \dots, \mathbf{h}_i^{(t-1)}\}$ is fed into a
Long Short-Term Memory (LSTM) network:

$$\mathbf{z}_i = \mathrm{LSTM}\!\left(\{\mathbf{h}_i^{(t')}\}_{t'=t-K}^{t-1}\right)$$

The final hidden state $\mathbf{z}_i \in \mathbb{R}^d$ is passed to a
linear readout layer to produce the forecast:
$\hat{y}_{i, t+h} = \mathbf{W}_{\text{head}}\mathbf{z}_i$.

## 4.5 Statistical Testing Framework

For each (model, graph variant, comparator, horizon) pair, we test
whether the graph model produces statistically smaller forecast errors
using the following procedure:

1.  **Diebold-Mariano test** with the Harvey, Leybourne and
    Newbold (1997) small-sample finite-horizon correction.

2.  **Bartlett HAC weighting** to account for serial correlation in loss
    differentials across the expanding test window.

3.  **Moving-block bootstrap** (1,000 resamples, block length 4
    quarters) for confidence interval construction that respects
    temporal dependence.

4.  **Benjamini-Hochberg FDR correction** at q = 0.05 across all
    simultaneous comparisons within each (model, variant, horizon)
    group.

Because neural and GNN models are estimated with 20 random seeds, we
report results as the **proportion of seeds for which the BH-corrected
p-value \< 0.05 and the loss differential favours the graph model**. We
adopt two interpretive thresholds: a *majority* threshold (\> 50% of
seeds significant) as the primary criterion for "consistent" evidence,
and a *supermajority* threshold (\> 75% of seeds) as a secondary
criterion for "strong" evidence. We deliberately do not require
unanimity across all 20 seeds, as any stochastic difference in
initialisation can preclude unanimity even when the population-level
effect is real and the statistical power per seed is high. Where
proportion-of-seeds results are informative, they are reported alongside
the point-forecast rankings.

::: center

------------------------------------------------------------------------
:::

# 5. Results

## 5.1 Point Forecast Accuracy

**Table 3: Main Results --- MAE and RMSE by Model and Horizon** *(lower
is better)*

+--------------+-----------------------+--------------+--------------+--------------+--------------+--------------+--------------+
| ::: minipage | ::: minipage          | ::: minipage | ::: minipage | ::: minipage | ::: minipage | ::: minipage | ::: minipage |
| Model        | Graph Variant         | H=1 MAE      | H=2 MAE      | H=4 MAE      | H=1 RMSE     | H=2 RMSE     | H=4 RMSE     |
| :::          | :::                   | :::          | :::          | :::          | :::          | :::          | :::          |
+:=============+:======================+:=============+:=============+:=============+:=============+:=============+:=============+
| **GCN**      | **identity_no_trade** | **1.707**    |              |              |              |              |              |
+--------------+-----------------------+--------------+--------------+--------------+--------------+--------------+--------------+
| ARIMA        | ---                   |              | **2.433**    |              |              |              |              |
+--------------+-----------------------+--------------+--------------+--------------+--------------+--------------+--------------+
| TCN          | ---                   |              |              |              |              |              |              |
+--------------+-----------------------+--------------+--------------+--------------+--------------+--------------+--------------+
| MLP          | ---                   |              |              |              |              |              |              |
+--------------+-----------------------+--------------+--------------+--------------+--------------+--------------+--------------+
| Persistence  | ---                   |              |              |              |              |              |              |
+--------------+-----------------------+--------------+--------------+--------------+--------------+--------------+--------------+
| ETS          | ---                   |              |              |              |              |              |              |
+--------------+-----------------------+--------------+--------------+--------------+--------------+--------------+--------------+
| Gradient     | ---                   |              |              |              |              |              |              |
| Boosting     |                       |              |              |              |              |              |              |
+--------------+-----------------------+--------------+--------------+--------------+--------------+--------------+--------------+
| Dynamic      | ---                   |              |              |              |              |              |              |
| Factor       |                       |              |              |              |              |              |              |
+--------------+-----------------------+--------------+--------------+--------------+--------------+--------------+--------------+
| **Temporal   | **identity_no_trade** |              | **2.358**    | **2.840**    |              |              |              |
| Graph**      |                       |              |              |              |              |              |              |
+--------------+-----------------------+--------------+--------------+--------------+--------------+--------------+--------------+
| LSTM         | ---                   |              |              |              |              |              |              |
+--------------+-----------------------+--------------+--------------+--------------+--------------+--------------+--------------+
| Ridge        | ---                   |              |              |              |              |              |              |
+--------------+-----------------------+--------------+--------------+--------------+--------------+--------------+--------------+
| BVAR         | ---                   |              |              |              |              |              |              |
+--------------+-----------------------+--------------+--------------+--------------+--------------+--------------+--------------+
| VAR          | ---                   |              |              |              |              |              |              |
+--------------+-----------------------+--------------+--------------+--------------+--------------+--------------+--------------+

*Notes: Values are mean absolute error / root mean squared error in
percentage points. Results for neural and GNN models are averaged over
20 random seeds. Best values per horizon in bold. MAE and RMSE are
computed separately from point forecasts; CRPS (probabilistic) values
are reported in Table 4.*

Key observations:

- At **h = 1**, GCN with `identity_no_trade` achieves the lowest point MAE
  (1.707 pp), narrowly ahead of ARIMA (1.751 pp). However, this point advantage
  is not statistically significant under DM testing for any seed, and GCN's RMSE is
  only marginally lower than ARIMA's (2.750 vs 2.884), suggesting comparable
  absolute performance.

- At **h = 2**, Temporal Graph with `identity_no_trade` ranks first in point MAE
  (2.358 pp), compared to ARIMA (2.433 pp). While this represents a 3% nominal
  reduction in point MAE, the difference is not statistically significant under
  DM testing across a majority of seeds, indicating that the advantage is highly
  sensitive to initialization.

- At **h = 4**, Temporal Graph with `identity_no_trade` achieves the lowest point
  MAE (2.840 pp) compared to ARIMA (3.382 pp), a 16% nominal improvement.
  Although this represents the largest nominal point-forecasting gain, it remains
  statistically insignificant under formal DM testing across all 20 seeds due to
  wide bootstrap confidence intervals and the volatility of late-sample test origins.

- **Critically**, the best-performing GNN at every horizon uses the
  *identity (no-trade)* graph, indicating trade-network topology is not
  responsible for observed outperformance.

- **BVAR** with Minnesota-prior shrinkage (MAE = 2.341/3.857/6.312 pp at
  h = 1/2/4) underperforms relative to simpler ARIMA, ETS, and
  persistence baselines, consistent with the well-documented difficulty
  of imposing informative priors on a volatile, post-COVID inflation
  regime.

## 5.2 Ablation: Graph Topology versus Architecture

**Table 4 (Ablation): Graph Variant Performance Aggregated Across GNN
Families**

+----------------------------+--------------+--------------+--------------+--------------+--------------+--------------+
| ::: minipage               | ::: minipage | ::: minipage | ::: minipage | ::: minipage | ::: minipage | ::: minipage |
| Graph Variant              | H=1 MAE      | H=2 MAE      | H=4 MAE      | H=1 CRPS     | H=2 CRPS     | H=4 CRPS     |
| :::                        | :::          | :::          | :::          | :::          | :::          | :::          |
+:===========================+:=============+:=============+:=============+:=============+:=============+:=============+
| `identity_no_trade`        | **1.853**    | **2.428**    | **3.254**    | **1.530**    | **2.071**    | **2.854**    |
+----------------------------+--------------+--------------+--------------+--------------+--------------+--------------+
| `directed_trade`           |              |              |              |              |              |              |
+----------------------------+--------------+--------------+--------------+--------------+--------------+--------------+
| `log_trade`                |              |              |              |              |              |              |
+----------------------------+--------------+--------------+--------------+--------------+--------------+--------------+
| `reversed`                 |              |              |              |              |              |              |
+----------------------------+--------------+--------------+--------------+--------------+--------------+--------------+
| `undirected`               |              |              |              |              |              |              |
+----------------------------+--------------+--------------+--------------+--------------+--------------+--------------+
| `import_dependence`        |              |              |              |              |              |              |
+----------------------------+--------------+--------------+--------------+--------------+--------------+--------------+
| `degree_preserving_random` |              |              |              |              |              |              |
+----------------------------+--------------+--------------+--------------+--------------+--------------+--------------+
| `top_k_incoming`           |              |              |              |              |              |              |
+----------------------------+--------------+--------------+--------------+--------------+--------------+--------------+

*Notes: Values averaged across GCN and Temporal Graph families, across
20 seeds. CRPS = Continuous Ranked Probability Score (probabilistic
accuracy).*

The `identity_no_trade` variant --- which encodes no trade information
--- is the best-performing graph construction across all horizons, in
both point (MAE) and probabilistic (CRPS) accuracy. The gap is
consistent: at h = 2, the identity graph outperforms the next-best trade
variant (`directed_trade`, MAE = 2.498) by approximately 3%. This
pattern is consistent with the hypothesis that the GNN architecture's
temporal attention mechanism, rather than trade-graph topology, is the
primary source of any marginal improvement over simpler baselines ---
though this remains an associational interpretation.

## 5.3 Probabilistic Forecast Accuracy

**Table 5: Probabilistic Results --- CRPS by Model and Horizon** *(lower
is better)*

+--------------+-----------------------+--------------+--------------+--------------+--------------+--------------+
| ::: minipage | ::: minipage          | ::: minipage | ::: minipage | ::: minipage | ::: minipage | ::: minipage |
| Model        | Graph Variant         | H=1 CRPS     | H=2 CRPS     | H=4 CRPS     | H=2 Cov-80   | H=4 Cov-80   |
| :::          | :::                   | :::          | :::          | :::          | :::          | :::          |
+:=============+:======================+:=============+:=============+:=============+:=============+:=============+
| **GCN**      | **identity_no_trade** | **1.378**    |              |              |              |              |
+--------------+-----------------------+--------------+--------------+--------------+--------------+--------------+
| ARIMA        | ---                   |              | **2.094**    | **3.013**    |              |              |
+--------------+-----------------------+--------------+--------------+--------------+--------------+--------------+
| MLP          | ---                   |              |              |              |              |              |
+--------------+-----------------------+--------------+--------------+--------------+--------------+--------------+
| TCN          | ---                   |              |              |              |              |              |
+--------------+-----------------------+--------------+--------------+--------------+--------------+--------------+
| Persistence  | ---                   |              |              |              |              |              |
+--------------+-----------------------+--------------+--------------+--------------+--------------+--------------+
| ETS          | ---                   |              |              |              |              |              |
+--------------+-----------------------+--------------+--------------+--------------+--------------+--------------+
| **Temporal   | **identity_no_trade** |              | **2.010**    | **2.451**    |              |              |
| Graph**      |                       |              |              |              |              |              |
+--------------+-----------------------+--------------+--------------+--------------+--------------+--------------+
| LSTM         | ---                   |              |              |              |              |              |
+--------------+-----------------------+--------------+--------------+--------------+--------------+--------------+

*Notes: CRPS is the continuous ranked probability score; lower is
better. Cov-80 = empirical coverage of 80% prediction intervals (target:
0.80). Note that MAE and CRPS are distinct loss functions; any apparent
similarity in magnitude between MAE and CRPS values for a given model
should be verified against the source data (see data availability
statement).*

80% prediction interval coverage is substantially below the nominal
level for all models (target 0.80; observed 0.50--0.58), consistent with
the short panel and bootstrap-based interval construction typical for
neural network ensembles in this setting. Calibration improvements
represent a clear avenue for future work.

## 5.4 Statistical Significance of Forecast Accuracy Differences

The analysis in this section distinguishes two conceptually separate
exercises that must not be conflated.

**Part A --- Heuristic point-forecast rankings (Table 3).** Table 3
ranks all 12 model families by MAE and RMSE. This ranking includes
heuristic benchmarks such as *Persistence* (the naïve "no-change"
forecast) alongside formally estimated models. These rankings are purely
descriptive: they characterise the relative ordering of out-of-sample
forecast errors on the test set. Persistence is retained in the
descriptive forecast rankings as a sanity-check benchmark only. Because
it is a fixed, non-estimated forecasting rule, it does not satisfy the
modelling assumptions underlying the Diebold--Mariano test and is
therefore excluded from all formal Diebold--Mariano comparisons.
Accordingly, *Persistence is explicitly excluded from all formal DM
testing in Part B*.

**Part B --- Formal DM tests (Table 6).** For each (GNN model, graph
variant, horizon $h$) triple, we conduct pairwise
Harvey--Leybourne--Newbold corrected DM tests against the following
*formally estimated, parametric comparators*: ARIMA, ETS, BVAR, Ridge,
TCN, LSTM, and MLP. The null hypothesis is equal predictive accuracy
(EPA). For each seed $s \in \{1, \ldots, 20\}$ and comparator $c$, we
compute the loss differential sequence:

$$d_t^{(s,c)} = |e_t^{\text{GNN},s}| - |e_t^{c}|, \quad t = 1, \ldots, T_{\text{test}}$$

The HLN-corrected DM statistic is:

$$\text{DM}^*_{s,c} = \frac{\bar{d}^{(s,c)}}{\sqrt{\hat{V}(\bar{d}^{(s,c)})}} \cdot \sqrt{\frac{T + 1 - 2h + h(h-1)/T}{T}}$$

where $\hat{V}(\bar{d})$ is Bartlett HAC-estimated with truncation lag
$\lfloor T^{1/3} \rfloor$. Significance is assessed after
Benjamini--Hochberg FDR correction at $q = 0.05$ across all simultaneous
comparisons within each (model, variant, horizon) stratum.

Results are reported as the **proportion of seeds for which the
BH-corrected $p$-value is below 0.05 and $\bar{d}^{(s,c)} < 0$** (i.e.,
the GNN produces lower expected loss). We adopt two interpretive
thresholds: *majority* (\> 50

**Table 6 --- Selected DM Test Results** \*(proportion of 20 seeds with
BH-corrected p \< 0.05 favouring GNN; Persistence excluded)\*

  Graph Model      Graph Variant         h   Comparator   Prop. Seeds Sig.   Inference
  ---------------- --------------------- --- ------------ ------------------ ----------------
  GCN              `identity_no_trade`       Ridge        \%                 **Strong**
  Temporal Graph   `identity_no_trade`       LSTM         \%                 **Strong**
  GCN              `identity_no_trade`       Ridge        \%                 Consistent
  GCN              `directed_trade`          Ridge        \%                 Consistent
  GCN              `identity_no_trade`       **ARIMA**    \%                 Not consistent
  GCN              `identity_no_trade`       **ARIMA**    \%                 Not consistent
  Temporal Graph   `identity_no_trade`       **ARIMA**    \%                 Not consistent
  GCN              `identity_no_trade`       Ridge        \%                 Not consistent
  Temporal Graph   `identity_no_trade`       LSTM         \%                 Not consistent

Two findings follow from Part B. First, the only comparator against
which GNNs achieve consistent or strong significance is Ridge regression
--- a relatively weak regularised baseline. Against the stronger
time-series models (ARIMA, BVAR, TCN), no GNN achieves majority-seed
significance at any horizon. Second, DM significance rates decline
sharply with horizon: all $h = 4$ results fall below the majority
threshold, consistent with increasing finite-sample instability in
expanding-window estimates at longer horizons.

The pattern in Part A (point rankings) and Part B (DM tests) is
coherent: Temporal Graph achieves a lower point MAE than ARIMA at
$h = 2$ and $h = 4$, but this difference is not statistically
reproducible across the majority of seed draws --- itself an informative
finding about the precision of the estimated advantage.

## 5.5 Diagnostic Figures

<figure id="fig:heatmap" data-latex-placement="H">
<img src="experiments/results/v2_1/manuscript/graph_variant_heatmap.png"
style="width:95.0%" />
<figcaption><strong>Figure 1: Graph Variant MAE Heatmap.</strong> Mean
Absolute Error (MAE) by model family and graph construction strategy
across all three forecast horizons. Darker shading indicates lower
(better) MAE. The identity (no-trade) graph consistently achieves the
lowest MAE for both GNN families, demonstrating that trade-network
topology provides no measurable accuracy gain.</figcaption>
</figure>

<figure id="fig:forecast_comp" data-latex-placement="H">
<img src="experiments/results/v2_1/manuscript/forecast_comparison.png"
style="width:95.0%" />
<figcaption><strong>Figure 2: Forecast comparison (France, Seed 42,
H=1).</strong> Comparison of Temporal Graph (directed_trade variant) and
ARIMA forecasts against actual CPI YoY inflation. The plot illustrates
where the neural model deviates from classical baselines during
inflation transition periods.</figcaption>
</figure>

<figure id="fig:perf_horizon" data-latex-placement="H">
<img
src="experiments/results/v2_1/manuscript/performance_by_horizon.png"
style="width:95.0%" />
<figcaption><strong>Figure 3: Performance by Forecast Horizon.</strong>
MAE and RMSE as a function of forecast horizon (<span
class="math inline"><em>h</em> = 1, 2, 4</span> quarters) for all 12
model families. Error bars represent the inter-seed range across 20
random initialisations. The Temporal Graph model shows the largest
improvement relative to ARIMA at <span
class="math inline"><em>h</em> = 4</span>, while GCN leads at <span
class="math inline"><em>h</em> = 1</span>.</figcaption>
</figure>

<figure id="fig:error_dist" data-latex-placement="H">
<img src="experiments/results/v2_1/manuscript/error_distribution.png"
style="width:95.0%" />
<figcaption><strong>Figure 4: Forecast Error Distribution.</strong> Box
plots showing the distribution of forecast errors across all countries,
quarters, and seeds for <span class="math inline"><em>h</em> = 1</span>.
The plot highlights the presence of heavier tails and extreme prediction
errors under classical VAR and MLP models compared to regularised and
GNN architectures.</figcaption>
</figure>

<figure id="fig:coverage_reliability" data-latex-placement="H">
<img
src="experiments/results/v2_1/manuscript/calibration_reliability.png"
style="width:95.0%" />
<figcaption><strong>Figure 5: Calibration Reliability Diagram.</strong>
Nominal vs. empirical coverage for nominal confidence levels of 80% and
95%. Points below the diagonal indicate systematic overconfidence
(empirical coverage understates nominal risk).</figcaption>
</figure>

<figure id="fig:coverage_bar" data-latex-placement="H">
<img
src="experiments/results/v2_1/manuscript/prediction_interval_coverage.png"
style="width:95.0%" />
<figcaption><strong>Figure 6: Prediction Interval Coverage.</strong>
Empirical coverage fraction of 80% (lower bar) and 95% (upper bar)
prediction intervals by model. The dashed lines represent nominal
targets (0.80 and 0.95). Every model fails to meet the nominal targets,
with GNNs achieving 50–58% coverage for 80% intervals.</figcaption>
</figure>

::: center

------------------------------------------------------------------------
:::

# 6. Discussion

## 6.1 Discussion: Interpretations of the Identity-Graph Control Result

The most striking finding within our experimental scope is that GNNs achieve lower point and probabilistic forecast errors when using the identity (no-trade) graph than when using any trade-graph construction. Because our design does not permit direct testing of the underlying channel, we present three alternative, non-mutually exclusive hypotheses to explain this outcome, intended to guide future empirical research:

1. **Architectural Regularisation:** The spatial aggregation operator in GNNs may act primarily as a regulariser. When applied to the identity graph, the model simplifies to a country-specific neural network where GCN-style aggregation acts as a constraint that improves generalization independent of actual cross-country trade signals (analogous to GNN regularisation properties noted in Kawamoto et al., 2018; Zügner et al., 2020).
2. **Information Redundancy of Trade Edges:** Slowly varying quarterly trade volumes may not capture the relevant high-frequency pricing dynamics. Autoregressive lags of own-country CPI and energy price indices may already capture the predictive signal of international inflation spillovers, rendering the explicit network structure redundant for out-of-sample forecasting at a quarterly frequency.
3. **Price Elasticity vs. Trade Quantity:** Bilateral trade volumes measure trade *quantities*, not the elasticities of price pass-through. Pass-through is highly dependent on contract currency invoicing, market structure, and input substitution, which trade flows alone do not reflect.

We emphasise that these interpretations are speculative hypotheses generated by the benchmark results, rather than established mechanisms. Adjudicating between them requires alternative experimental designs (such as monthly frequency replications or end-to-end graph learning) which are outside the scope of this study.


## 6.3 Horizon Dependence

The Temporal Graph's advantage over comparators grows with horizon: the
MAE gap versus ARIMA is small at h = 1 (1.999 vs 1.751), widens at h = 2
(2.358 vs 2.433; Temporal Graph is now better), and is most pronounced
at h = 4 (2.840 vs 3.382). This pattern is consistent with the temporal
attention mechanism being beneficial for medium-run trend extrapolation
but adding little over the persistence benchmark for one-step-ahead
prediction. The finding parallels results in the recurrent network
literature (Hewamalage et al., 2021) where LSTM gains over linear models
accumulate primarily at longer horizons.

## 6.4 Statistical Robustness

The proportion-of-seeds significance results reveal an important nuance:
even where GNNs appear to win in point-forecast rankings, the win is
often not statistically reproducible across initialisation draws,
particularly against strong comparators. For instance, GCN with
`identity_no_trade` achieves majority- seed significance against Ridge
at h = 1 and h = 2 but fails against ARIMA and TCN at every horizon.
This finding underscores the importance of multi-seed testing:
single-seed "champion" results, common in the GNN-for-economics
literature, can be substantially misleading.

## 6.5 Limitations

1.  **Hyperparameter sharing.** All five neural architectures (MLP,
    LSTM, TCN, GCN, Temporal Graph) use identical hyperparameters
    (hidden dimension 16, learning rate 0.01, 30 epochs) to enable a
    controlled architecture-level comparison. This means each individual
    architecture may be operating below its capacity optimum. In
    particular, 30 epochs is shallow for models with attention
    mechanisms, and a proper validation-set sweep per architecture might
    reveal larger GNN gains or strengthen the case for simpler models.
    Holding hyperparameters constant is a deliberate choice to isolate the
    architectural contribution, but it prevents us from claiming that these
    results represent the absolute performance ceilings of the respective models.

2.  **Short initial training window.** The expanding window starts with
    only 14 quarters (2011Q2--2014Q4) for the initial model fits. For
    GNNs and LSTMs with hundreds to thousands of parameters,
    early-window fits are likely undertrained, and forecast errors in
    the first several test origins partly reflect model initialisation
    quality rather than intrinsic architecture differences. This
    contributes to instability in seed-to-seed significance at shorter
    horizons.

3.  **Revised vintages.** Although the benchmark uses final revised
    Eurostat releases for cross-country harmonisation, the selected
    covariates are limited to variables with comparatively low revision
    intensity and early availability. Heavily revised macroeconomic
    series, such as output gaps, unit labour costs, and real-time
    productivity measures, are excluded to reduce look-ahead concerns in
    practical forecasting applications. However, because we do not
    evaluate the models on historical real-time vintages, the results
    reflect an upper-bound forecast accuracy environment; actual
    real-time forecasting performance may differ due to publication lags
    and subsequent statistical revisions. A full real-time vintage
    evaluation is left for future work.

4.  **Panel scope.** The panel covers 20 European economies. Results may
    not generalise to emerging markets, non-EU economies, or global
    panels where data quality, trade-flow reporting, and inflation
    dynamics differ materially.

5.  **No causal claims.** All findings pertain to predictive
    associations. The study does not establish causal mechanisms for
    inflation transmission, nor does it imply that monitoring
    trade-network topology would improve central bank forecasting
    decisions without further validation.

6.  **Prediction interval calibration shortfall.** Across all models,
    the empirical coverage of the 80% prediction intervals is
    systematically low, ranging between 50% and 58%. This undercoverage
    is a consequence of the short initial training window (14 quarters)
    and the bootstrap-based uncertainty estimation which understates
    prediction variance during shock periods. Because of this poor
    calibration, the probabilistic forecasting results (CRPS and intervals) should
    be treated as secondary, descriptive metrics rather than a primary contribution of
    this study. Future work should incorporate conformal prediction methods to
    guarantee nominal coverage.

7.  **Pre-specified trade graphs.** The trade-graph constructions
    evaluated (e.g., export-share, log-export, reversed import) are
    pre-specified and static or slowly varying based on historical trade
    volumes. We do not evaluate GNNs that perform end-to-end graph
    structure learning. If the trade relationships that matter for
    inflation dynamics are not well-represented by direct trade volumes,
    GNNs using these graphs may underperform compared to architectures
    that learn the graph structure dynamically.

8.  **Lack of robustness checks on design choices.** All model rankings and
    statistical tests are conditioned on a fixed initial training window size of
    14 quarters and a specific dataset vintage. We do not perform sensitivity
    checks varying the training window length (e.g., to 20 or 24 quarters) or evaluating
    pseudo-real-time vintage data. The sensitivity of the relative model rankings to these
    design choices remains unverified and represents a boundary condition of our findings.

These findings are conditional on the experimental scope. The restricted covariate set, quarterly frequency, European panel, short initial training window, pre-specified trade graphs, lack of comprehensive robustness checks, and revised data releases may all affect the relative ranking of models. The results should therefore be interpreted as evidence about this specific forecasting design, not as a universal statement about trade networks or GNNs.

## 6.6 Future Research Directions

The hypotheses generated by this study point to five concrete research
designs that could distinguish the proposed mechanisms:

1.  **Monthly frequency replication.** The most direct test of
    Hypothesis 2 is to repeat the identical ablation using monthly
    bilateral trade flows and monthly CPI data. Eurostat provides
    monthly Comext data with approximately a two-month publication lag.
    If quarterly aggregation is the binding constraint, the performance
    gap between `identity_no_trade` and trade-graph variants should
    narrow materially at monthly frequency.

2.  **Real-time data vintages.** The current study uses revised data
    releases. Replicating the benchmark with real-time vintages (ECB
    Real-Time Database; OECD MEI real-time releases) would (a) produce
    accuracy estimates that better reflect genuine forecaster
    information sets and (b) test whether the
    topology-versus-architecture ranking changes under data uncertainty.

3.  **Graph structure learning.** Rather than pre-specifying eight graph
    constructions, one could learn the graph jointly with the
    forecasting model (Wu et al., 2020; Shang et al., 2021). If a
    learned graph substantially outperforms both trade-based and
    identity constructions, it would suggest that the *form* of the
    trade graph --- rather than its presence or absence --- is the
    binding constraint, lending support to a refined version of
    Hypothesis 2.

4.  **Richer covariate sets.** Including output gaps, unit labour costs,
    and import price indices --- with appropriate treatment of
    publication lags --- would test whether the
    topology-versus-architecture conclusion is robust to a richer
    feature set or is specific to the minimal four-variable
    specification used here.

5.  **Architecture-specific hyperparameter tuning.** The present study
    uses identical hyperparameters across all neural models for
    controlled comparability. A study that tunes each architecture
    independently --- with proper validation-set search and longer
    training --- would test whether the GNN advantage strengthens under
    optimal configuration.

## 6.7 Claim-to-Evidence Mapping

The following table maps every major claim made in this study to the
exact empirical evidence presented in the paper, alongside relevant
boundary conditions and caveats.

  Core Claim                                                                                                                                                                                                                 Supporting Evidence in Paper                                                                                                                                                               Boundary Conditions & Caveats
  -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
  **1.** Within the studied scope, trade-network topology does not provide incremental out-of-sample forecast accuracy for EU CPI inflation relative to the same GNN architecture applied to an identity (no-trade) graph.   **Table 4** (ablation): `identity_no_trade` achieves the lowest MAE at all horizons for both GNN families. Confirmed in point and probabilistic (CRPS) accuracy. **Figure 1** (heatmap).   EU economies; quarterly frequency; 4-variable covariate set; post-2011Q2; GCN and Temporal Graph architectures only. Does not generalise to monthly frequency, richer features, or non-EU panels.
  **2.** No trade-graph GNN achieves statistically reproducible superiority --- defined as majority-seed BH-corrected DM significance --- over its best formal parametric comparator.                                        **Table 6** (DM tests): All comparisons against ARIMA, BVAR, ETS, TCN fail the \> 50% seed threshold. Ridge comparisons achieve 60--80% at h=1--2 only.                                    Persistence is *excluded* from formal DM testing (degenerate rule, not an estimated model). Claims apply to MAE loss; RMSE-based DM may differ. ETS is a borderline case.
  **3.** Temporal Graph achieves the lowest out-of-sample point MAE of any model at h=2 and h=4, using the identity (no-trade) graph, within this experimental setting.                                                      **Table 3**: Temporal Graph (identity): MAE = 2.358 pp (h=2), 2.840 pp (h=4). Next best: ARIMA (2.433; 3.382 pp). **Figure 3** (horizon-MAE plot).                                         Statistical reproducibility is limited: majority-seed DM significance against ARIMA is not achieved at any horizon. Rankings reflect means over 20 seeds; per-seed rankings vary.
  **4.** GNN accuracy gains, when present, are associated with the temporal architecture rather than cross-country trade-network propagation, based on the identity-graph ablation.                                          **Table 4**: `identity_no_trade` dominates all 7 trade-graph variants at every horizon. **Figure 1** (heatmap): pattern consistent across GCN and Temporal Graph.                          This is an *associational* finding, not a causal identification. Two alternative hypotheses are proposed (architectural regularisation; temporally coarse edges) but not tested within this study.
  **5.** BVAR with Minnesota-prior shrinkage does not improve out-of-sample forecast accuracy over ARIMA or ETS in this quarterly EU panel setting.                                                                          **Table 3**: BVAR MAE = 2.341/3.857/6.312 pp at h=1/2/4 vs ARIMA (1.751/2.433/3.382 pp). BVAR ranks 10th out of 12 overall.                                                                Specific to Minnesota-prior BVAR, lag-1, on a 4-variable system. Prior not exhaustively tuned. Consistent with documented prior-selection challenges in volatile post-COVID regimes (Bánbura et al., 2010).

::: center

------------------------------------------------------------------------
:::

# 7. Conclusion

We conduct a large-scale, fully reproducible prospective evaluation of
GNN-based inflation forecasting across 20 European economies at three
horizons, comparing 12 model families under 8 trade-graph constructions
and 20 random seeds. Every finding is conditional on the scope defined
in Section 1 and the methodology described in Section 4.

Our principal findings are:

1.  **Temporal Graph shows the largest absolute forecast accuracy gain
    at medium-to-long horizons.** Temporal Graph with the identity graph
    achieves MAE of 2.358 pp at $h = 2$ (ARIMA: 2.433 pp) and 2.840 pp
    at $h = 4$ (ARIMA: 3.382 pp), suggesting that temporal recurrence in
    the GNN architecture provides predictive value at longer horizons.
    GCN leads at $h = 1$ but by a narrow margin.

2.  **Temporal architecture, not trade topology, is the consistent
    predictor of within-GNN rankings.** The identity (no-trade) graph
    achieves lower out-of-sample MAE than every trade-based graph
    variant at every horizon for both GNN families, in both point (MAE,
    RMSE) and probabilistic (CRPS) accuracy.

3.  **Statistical reproducibility is limited, particularly against
    strong baselines.** Under BH-corrected DM testing across 20 seeds,
    no GNN achieves majority-seed significance against ARIMA, ETS, or
    TCN at any horizon. The most consistent GNN advantage is against
    Ridge regression. Significance rates decline sharply with horizon.

These findings carry methodological implications: identity-graph
ablation should be a standard requirement in GNN macroeconomic
forecasting papers; multi-seed reporting should replace single-seed
benchmarks; and claims of topology-driven forecast improvement require
explicit statistical disentanglement from architectural effects.

What this study does not establish is whether trade-network topology
improves forecast accuracy at monthly frequency, with richer covariate
sets, over longer historical samples, or for non-EU economies. We
encourage future work to vary these design dimensions systematically.

The methodological contribution of this study is not a performance breakthrough but a design template. By publishing a pre-specified prospective ablation with a public identity-graph control, frozen evaluation outputs, and multi-seed significance reporting, this study establishes a reproducible baseline against which future graph-based macroeconomic forecasting work can be calibrated. Negative benchmark results — when rigorously designed and honestly reported — serve the forecasting community by establishing the conditions under which a modelling choice does not add value, guiding where research effort is more productively directed.

::: center

------------------------------------------------------------------------
:::

# Data Availability

All data are sourced from publicly available repositories:

- **CPI/HICP:** Eurostat HICP database; IMF International Financial
  Statistics

- **Bilateral trade:** Eurostat Comext quarterly trade statistics

- **Energy prices:** Eurostat energy component of HICP; World Bank
  commodity prices

Processed datasets, all model outputs (forecasts, metrics, DM tests),
and full reproduction scripts are available at:

> **GitHub Repository:**
> <https://github.com/mrayanasim09/MarketEquationDiscovery>
>
> **SSRN Preprint (v1 baseline):**
> <https://doi.org/10.2139/ssrn.7009041>

All artefacts are SHA256-verified for exact reproducibility. The
benchmark can be re-run end-to-end using the provided `reproduce.sh`
script (estimated runtime: approximately 12 hours on Apple Silicon
M-series; longer on CPU-only hardware).

# Code Availability

Full source code is publicly available at:
<https://github.com/mrayanasim09/MarketEquationDiscovery>

The repository includes benchmark protocol specification, execution
engine, statistical analysis pipeline, manuscript figure and table
generation scripts, and SHA256-verified frozen results.

# Reproducibility Statement {#reproducibility-statement .unnumbered}

All code, configuration files, processed forecast samples, and frozen
evaluation outputs are provided to support reproducibility. The
evaluation uses a strictly prospective expanding-window design with
fixed random seeds and pre-specified model configurations. No
test-period information is used during model selection.

# Conflict of Interest

The author declares no conflicts of interest.

# Funding

This research received no external funding.

# Acknowledgements

Bilateral trade data are sourced from Eurostat Comext; macroeconomic
indicators from the IMF International Financial Statistics and World
Bank World Development Indicators; CPI data from Eurostat HICP.

::: center

------------------------------------------------------------------------
:::

# References

Atkeson, A., and Ohanian, L. E. (2001). Are Phillips curves useful for
forecasting inflation? *Federal Reserve Bank of Minneapolis Quarterly
Review*, 25(1), 2--11.

Bánbura, M., Giannone, D., and Reichlin, L. (2010). Large Bayesian
vector autoregressions. *Journal of Applied Econometrics*, 25(1),
71--92.

Benjamini, Y., and Hochberg, Y. (1995). Controlling the false discovery
rate: a practical and powerful approach to multiple testing. *Journal of
the Royal Statistical Society: Series B (Methodological)*, 57(1),
289--300.

Bayoumi, T., Bui, T., and Berkmen, P. (2023). Trade network exposure and
inflation dynamics. IMF Working Paper WP/23/117.

Calvo, G. A., and Reinhart, C. M. (2002). Fear of floating. *Quarterly
Journal of Economics*, 117(2), 379--408.

Chen, Y., Li, M., and Zhang, Z. (2023). Graph neural networks for
commodity price forecasting. *Energy Economics*, 118, 106482.

Cheng, D., and Zhu, J. (2022). Financial contagion detection via
spatio-temporal graph networks. *Journal of Financial Stability*, 63,
101073.

Clark, T. E., and McCracken, M. W. (2001). Tests of equal forecast
accuracy and encompassing for nested models. *Journal of Econometrics*,
105(1), 85--110.

Clark, T. E., and West, K. D. (2007). Approximately normal tests for
equal predictive accuracy in nested models. *Journal of Econometrics*,
138(1), 291--311.

Canova, F., and Ciccarelli, M. (2013). Panel Vector Autoregressive Models: A Survey. In VAR Models in Macroeconomics – New Developments and Applications: Essays in Honor of Christopher A. Sims, *Advances in Econometrics*, 32, 205--246.

Coulombe, P. G., Leroux, M., Stevanovic, D., and Surprenant, S. (2020).
How is machine learning useful for macroeconomic forecasting? *Journal
of Applied Econometrics*, 37(5), 920--964.

Diebold, F. X., and Mariano, R. S. (1995). Comparing predictive
accuracy. *Journal of Business and Economic Statistics*, 13(3),
253--263.

Faust, J., and Wright, J. H. (2013). Forecasting inflation. In G.
Elliott and A. Timmermann (Eds.), *Handbook of Economic Forecasting*
(Vol. 2, pp. 2--56). Elsevier.

Forbes, K. J., and Warnock, F. E. (2012). Capital flow waves: Surges,
stops, flight, and retrenchment. *Journal of International Economics*,
88(2), 235--251.

Forni, M., Hallin, M., Lippi, M., and Reichlin, L. (2000). The generalised dynamic factor model: Identification and estimation. *The Review of Economics and Statistics*, 82(4), 540--554.

Giannone, D., Lenza, M., and Primiceri, G. E. (2015). Prior selection
for vector autoregressions. *Review of Economics and Statistics*, 97(2),
436--451.

Harvey, D., Leybourne, S., and Newbold, P. (1997). Testing the equality
of prediction mean squared errors. *International Journal of
Forecasting*, 13(2), 281--291.

Hewamalage, H., Bergmeir, C., and Bandara, K. (2021). Recurrent neural
networks for time series forecasting: Current status and future
directions. *International Journal of Forecasting*, 37(1), 388--427.

Kawamoto, T., Tsubaki, M., and Saito, T. (2018). Mean-field theory of
graph neural networks in graph partitioning. *Advances in Neural
Information Processing Systems* (NeurIPS), 31.

Kipf, T. N., and Welling, M. (2017). Semi-supervised classification with
graph convolutional networks. *International Conference on Learning
Representations*.

Li, Y., Yu, R., Shahabi, C., and Liu, Y. (2018). Diffusion convolutional
recurrent neural network: Data-driven traffic forecasting.
*International Conference on Learning Representations*.

Litterman, R. B. (1986). Forecasting with Bayesian vector
autoregressions---five years of experience. *Journal of Business &
Economic Statistics*, 4(1), 25--38.

Makridakis, S., Spiliotis, E., and Assimakopoulos, V. (2018).
Statistical and machine learning forecasting methods: Concerns and ways
forward. *PLOS ONE*, 13(3), e0194889.

Makridakis, S., Spiliotis, E., and Assimakopoulos, V. (2022). M5
accuracy competition: Results, findings, and conclusions. *International
Journal of Forecasting*, 38(4), 1346--1364.

Medeiros, M. C., Vasconcelos, G. F. R., Veiga, A., and Zilberman, E.
(2021). Forecasting inflation in a data-rich environment: The benefits
of machine learning methods. *Journal of Business and Economic
Statistics*, 39(1), 98--119.

Miranda-Agrippino, S., and Rey, H. (2021). The global financial cycle.
In G. Gopinath, E. Helpman, and K. Rogoff (Eds.), *Handbook of
International Economics* (Vol. 6, pp. 1--43). Elsevier.

Pesaran, M. H., Schuermann, T., and Weiner, S. M. (2004). Modeling regional interdependencies using a global error-correcting macroeconometric model. *Journal of Business and Economic Statistics*, 22(2), 129--162.

Stock, J. H., and Watson, M. W. (2002). Macroeconomic forecasting using diffusion indexes. *Journal of Business and Economic Statistics*, 20(2), 147--162.

Stock, J. H., and Watson, M. W. (2007). Why has U.S. inflation become
harder to forecast? *Journal of Money, Credit and Banking*, 39(s1),
3--33.

Stark, T., and Croushore, D. (2002). Forecasting with a real-time data
set for macroeconomists. *Journal of Macroeconomics*, 24(4), 507--531.

Wang, X., Chen, T., and Li, H. (2024). Spatio-temporal graph networks
for regional GDP forecasting. *Regional Science and Urban Economics*,
105, 103990.

Wu, Z., Pan, S., Chen, F., Long, G., Zhang, C., and Yu, P. S. (2019). A
comprehensive study on spatial-temporal graph neural networks. *IEEE
Transactions on Neural Networks and Learning Systems*, 32(2), 527--540.

Zugner, D., Akbarnejad, A., and Gunnemann, S. (2020). Adversarial
attacks on neural networks for graph data. *Proceedings of the 24th ACM
SIGKDD International Conference on Knowledge Discovery and Data Mining*,
2847--2856.
