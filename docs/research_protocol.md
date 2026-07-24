# Research Protocol: Inflation Contagion in Trade Networks with ST-GNNs

**Version:** 1.1  
**Date:** 2026-06-27  
**Status:** Locked — v1.1 data pipeline fix (native CPI, 23 countries, CEPII trade).

---

## 1. Research question

Can a spatio-temporal graph neural network (ST-GNN) built on dynamic bilateral trade networks forecast **next-quarter CPI inflation (YoY)** across a panel of major economies **more accurately than standard macro baselines**, and can explainability analysis identify **which trade partners and edges** contribute most to inflation spillovers?

**Primary hypothesis:** Supply shocks propagate through trade network topology; a GCN+LSTM model that explicitly uses adjacency structure captures cross-country inflation spillovers that univariate ARIMA and panel VAR miss.

**Contribution framing (SSRN):** Model inflation as a contagion process over a dynamic trade graph, not as independent country time series.

---

## 2. Target variable

| Field | Decision |
|-------|----------|
| Variable | Consumer Price Index inflation, year-over-year percent change |
| Symbol | `cpi_yoy` |
| Frequency | Quarterly from **native monthly CPI index** (FRED/OECD); never annual broadcast |
| CPI source | FRED monthly CPI index (primary); OECD SDMX (fallback for SAU, SGP) |
| Forecast target | One-step ahead: predict `cpi_yoy` at quarter *t+1* using information through quarter *t* |
| Robustness (later) | 2- and 4-quarter ahead horizons (Milestone 7 only) |

**Formula:** `cpi_yoy_q = ((CPI_q / CPI_{q-4}) - 1) * 100`

---

## 3. Country set

**Panel:** 23 countries — G20 members plus key exporters (Netherlands, Singapore, Switzerland, Vietnam). **Taiwan (TWN) excluded** (not formal G20; documented in Methods).

| ISO3 | Country | Rationale |
|------|---------|-----------|
| ARG | Argentina | G20 |
| AUS | Australia | G20 |
| BRA | Brazil | G20 |
| CAN | Canada | G20 |
| CHN | China | G20 |
| FRA | France | G20 |
| DEU | Germany | G20 |
| IND | India | G20 |
| IDN | Indonesia | G20 |
| ITA | Italy | G20 |
| JPN | Japan | G20 |
| MEX | Mexico | G20 |
| RUS | Russia | G20 |
| SAU | Saudi Arabia | G20 |
| ZAF | South Africa | G20 |
| KOR | South Korea | G20 |
| TUR | Turkey | G20 |
| GBR | United Kingdom | G20 |
| USA | United States | G20 |
| NLD | Netherlands | Key exporter / trade hub |
| SGP | Singapore | Key exporter / hub |
| CHE | Switzerland | Key exporter |
| VNM | Vietnam | Key exporter (manufacturing) |

**Note:** EU is excluded as a single node (double-counts members). Russia retained for historical coverage but flagged in limitations post-2022.

**Drop rule:** Exclude any country with >20% missing CPI observations in sample window after imputation attempts.

---

## 4. Sample period and splits

| Split | Quarters | Purpose |
|-------|----------|---------|
| Full sample | 2000Q1 – 2024Q4 | Data collection window |
| Training | 2000Q1 – 2018Q4 | Model fitting |
| Validation | 2019Q1 – 2020Q4 | Hyperparameter tuning, early stopping |
| Test (hold-out) | 2021Q1 – 2024Q4 | Final out-of-sample evaluation |

**Rolling evaluation (Milestone 7):** Expanding-window pseudo-real-time forecasts on test period; document whether models retrain each quarter or use fixed weights from validation-tuned checkpoint.

---

## 5. Data sources (free)

| Data | Source | Access |
|------|--------|--------|
| Bilateral trade | CEPII BACI bulk CSV | Manual download + local filter |
| CPI (monthly index) | FRED (primary), OECD SDMX (fallback) | Free CSV / REST |
| GDP, NEER, policy rate | World Bank WDI | `wbgapi` |
| Energy/commodity prices | FRED Brent spot | CSV |
| ISO country codes | UN M49 | Reference mapping |

All raw downloads stored under `data/raw/` with download timestamp in manifest.

---

## 6. Graph construction

| Field | Decision |
|-------|----------|
| Nodes | 23 countries (see §3) |
| Edge weight | Total bilateral trade = imports + exports between pair (*i*, *j*) in USD, log(1 + x) transformed |
| Direction | Undirected aggregated adjacency (sum of both directions) for primary model; directed variant as robustness |
| Time dimension | One adjacency matrix **A_t** per quarter |
| Filtering | Optional disparity/backbone filter (Serrano et al.) applied at 95th percentile significance — document if used |
| Normalization | Symmetric normalized adjacency: D̃⁻½ Ã D̃⁻½ for GCN |

---

## 7. Node features (inputs at time *t*)

| Feature | Symbol | Transform |
|---------|--------|-----------|
| CPI inflation YoY | `cpi_yoy` | From monthly index → quarterly YoY (also target) |
| Real GDP growth YoY | `gdp_yoy` | Level (World Bank annual) |
| Nominal effective exchange rate change | `neer_chg` | YoY % change |
| Policy interest rate | `policy_rate` | Level; forward-fill gaps ≤2 quarters only |
| Global energy price proxy | `energy_idx` | YoY % change (same for all nodes) |
| COVID shock dummy | `covid` | 1 for 2020Q1–2021Q4, else 0 |

**Dropped features:** `ppi_yoy` (collinear; IMF unavailable). **Excluded country:** TWN.

---

## 8. Models

### Primary model: ST-GNN (GCN + LSTM)

- **Spatial:** 2-layer GCN with ReLU, dropout 0.2
- **Temporal:** 1-layer LSTM, hidden dim = 64 (tune 32–128 on validation)
- **Output:** Linear head → scalar `cpi_yoy` forecast per node
- **Loss:** MSE across nodes and time steps in training window
- **Optimizer:** Adam, lr ∈ {1e-4, 5e-4, 1e-3}, early stopping on validation MSE

### Baselines (required)

| Model | Specification | Library |
|-------|---------------|---------|
| ARIMA | ARIMA(0,1,1) per country on `cpi_yoy` (Monken et al. baseline) | `statsmodels` |
| VAR | VAR(4) on [`cpi_yoy`, `gdp_yoy`, `neer_chg`, `energy_idx`] | `statsmodels` |

### Structural baseline (optional, Milestone 5)

- Simplified Leontief input-output shock propagation if OECD ICIO free slice is obtainable; **fallback:** skip and note in limitations.

---

## 9. Evaluation metrics

Computed per country and pooled (macro average):

| Metric | Definition |
|--------|------------|
| RMSE | Root mean squared error of forecast vs actual |
| MAE | Mean absolute error |
| MAPE | Mean absolute percentage error (exclude \|actual\| < 0.5%) |
| Direction accuracy | Share of quarters where sign(forecast − lag) = sign(actual − lag) |
| Diebold-Mariano | Paired test: ST-GNN vs each baseline on test-period forecast errors |

**Success criterion (SSRN):** ST-GNN beats at least one baseline on pooled RMSE or MAE with DM p < 0.10, **or** delivers superior explainability case studies with comparable accuracy.

---

## 10. Explainability (Milestone 8)

- GAT attention weights **or** Integrated Gradients (Captum) on edge weights
- **Three required case studies:**
  1. COVID supply-chain period (2020–2021)
  2. 2022 global energy price shock
  3. One target country where partner spillovers dominate domestic features

---

## 11. Reproducibility commitments

- Fixed random seeds: `numpy=42`, `torch=42`
- Python ≥ 3.11, pinned dependencies in `requirements.txt`
- All scripts runnable from repo root via `python -m src.<module>`
- Dataset version frozen at end of Milestone 4 (`data/processed/dataset_v1/`)

---

## 12. Related work differentiation

| Prior work | Our distinction |
|------------|-----------------|
| Monken et al. (2021) | They forecast **trade flows**; we forecast **inflation** with trade graph as channel |
| Nason & Palasciano (2025) | UK CPI components only; we use **cross-country trade network** |
| Generic LSTM/ARIMA inflation papers | We integrate **dynamic bilateral trade topology** explicitly |

---

## 13. Milestone 1 completion checklist

- [x] Research question stated
- [x] Target variable defined (`cpi_yoy`, quarterly, 1-step ahead)
- [x] Country list locked (24 ISO3 codes)
- [x] Train / val / test splits defined
- [x] Data sources documented
- [x] Graph and feature specs documented
- [x] Baselines and metrics defined
- [x] Related-work differentiation written

**Milestone 1 status: COMPLETE**
