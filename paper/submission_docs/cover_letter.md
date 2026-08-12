# Cover Letter — Major Revision Submission

**Manuscript Title:** Temporal Architecture or Trade Topology? A Prospective Ablation Study of Spatio-Temporal Graph Neural Networks for EU Inflation Forecasting  
**Author:** Rayyan Asim  
**Target Journals:** *International Journal of Forecasting* (Primary / Stretch), *Journal of Forecasting*, *Empirical Economics*, *Computational Economics*  
**Date:** August 2026

---

Dear Editor,

Please consider our manuscript, **"Temporal Architecture or Trade Topology? A Prospective Ablation Study of Spatio-Temporal Graph Neural Networks for EU Inflation Forecasting,"** for publication in your journal.

This paper presents a prospective expanding-window benchmark across **13 model families**, 8 trade-graph constructions, 3 forecast horizons ($h = 1, 2, 4$), and 20 random seeds (totalling 39,188 model fits and 783,760 forecast rows) for quarterly CPI forecasting across 20 European Union economies. Its central contribution is a prospective identity-graph ablation showing that bilateral trade-network topology does not improve out-of-sample forecast accuracy over an identity (no-trade) graph control within this experimental design.

**Key Contributions and Revisions in this Submission:**

1. **Disciplined Negative Benchmark:** Isolates architectural temporal recurrence from bilateral trade topology, establishing that spatial GNN gains should not be attributed to network structure without passing an identity-graph control.
2. **Comprehensive 13-Model Benchmark:** Evaluates standard linear baselines (ARIMA, ETS, VAR, Persistence), regularised ML (Ridge, Gradient Boosting), Bayesian VAR (Minnesota prior), non-network cross-country baselines (Dynamic Factor Model), graph-free neural networks (MLP, LSTM, TCN), and spatial GNNs (GCN, Temporal Graph).
3. **Explicit Boundary Conditions & Associational Interpretation:** Rigorously qualifies all findings by the studied scope (20 EU economies, quarterly frequency, 4-variable low-revision feature set, 14-quarter initial training window, pre-specified trade graphs). Interpretations are strictly framed as associational rather than proven causal mechanisms.
4. **Statistical Testing and Calibration Contextualisation:** Applies Harvey–Leybourne–Newbold DM tests with Bartlett HAC weighting, moving-block bootstrap, and Benjamini–Hochberg FDR correction. Contextualises prediction interval under-coverage (~0.50–0.58 empirical coverage for 80% target) as a systemic post-2020 macro volatility phenomenon across all model families.
5. **Full Transparency and Reproducibility:** Accompanied by a public code repository, frozen evaluation outputs (`forecasts.parquet`, `metrics.parquet`, `dm_tests.parquet`), and a formal Reproducibility Statement.

We believe this negative but policy-relevant and methodologically rigorous result fits your journal's interest in transparent forecast evaluation, empirical benchmarking, and econometric methodology.

Thank you for your consideration.

Yours sincerely,

**Rayyan Asim**  
Independent Researcher  
mrayanasim09@gmail.com  
ORCID: 0000-0003-2461-5638

