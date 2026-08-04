# Cover Letter — Revised Submission

**Journal:** International Journal of Forecasting  
**Manuscript Title:** Temporal Architecture or Trade Topology? A Prospective Ablation Study of Graph Neural Networks for Inflation Forecasting  
**Authors:** Rayyan Asim  
**Date:** August 2026

---

Dear Editor,

Please consider our manuscript, "Temporal Architecture or Trade Topology? A Prospective Ablation Study of Graph Neural Networks for Inflation Forecasting," for publication in the *International Journal of Forecasting* as a revised submission. 

The study presents a controlled prospective ablation benchmark for quarterly CPI inflation forecasting across 20 EU economies. Its central contribution is to isolate whether bilateral trade-network topology adds out-of-sample predictive value beyond graph architecture alone. Across 12 model families, 8 graph constructions, 3 horizons, and 20 random seeds (amounting to over 38,000 model fits), identity-graph controls consistently outperform trade-based graphs within the studied scope. 

We believe this negative but methodologically important result is highly relevant to readers interested in forecasting evaluation, graph learning, and macroeconomic prediction. Specifically, it establishes the necessity of identity-graph ablation as a standard control when evaluating network-based models, highlighting the risk of over-parameterisation in macroeconomic panel forecasting.

We have addressed the comments from the editorial team and reviewers in this revision, including:
1. **Reframing around the negative benchmark:** The title and abstract have been rewritten to foreground the negative ablation results and clarify why this control design is crucial for the forecasting community.
2. **Adding a Theoretical Motivation Section:** Added Section 2.5 explicitly discussing the economic channels (import price pass-through, supply chains) and justifying why trade networks could plausibly improve forecasts over country-specific or common-factor baselines.
3. **Addressing the BVAR Baseline:** Added a Minnesota-prior BVAR baseline as a formally estimated competitor.
4. **DM-Testing and Persistence Consistency:** Persistence is consistently excluded from all formal Diebold–Mariano comparisons since it is a degenerate, non-estimated baseline.
5. **Hyperparameter and Calibration Disclaimers:** Added detailed disclaimers in the Methodology (§4.2) and Limitations (§6.5) sections regarding the shared hyperparameters, interval calibration shortfall, and the lack of sensitivity checks.
6. **Reference Audit:** Replaced the out-of-domain Garcia-Martos citation with Medeiros et al. (2021) and Coulombe et al. (2020), and added Stark and Croushore (2002) in §4.0.

The manuscript is original, not under review elsewhere, and all code, configuration files, and frozen outputs are publicly available to support complete reproducibility.

Yours sincerely,

Rayyan Asim  
Independent Researcher  
mrayanasim09@gmail.com  
ORCID: 0000-0003-2461-5638
