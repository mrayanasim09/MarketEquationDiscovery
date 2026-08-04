# Cover Letter

**Manuscript Title:** Temporal Architecture or Trade Topology? A Prospective Ablation Study of Spatio-Temporal Graph Neural Networks for Quarterly CPI Inflation Forecasting  
**Authors:** Rayyan Asim  
**Date:** August 2026

---

Dear Editors,

We are pleased to submit a substantially revised version of our manuscript for your consideration. We are grateful to the reviewers for their detailed and constructive feedback, which has materially improved the paper.

The original submission presented an ablation benchmark evaluating whether bilateral trade-network topology improves out-of-sample CPI inflation forecasting accuracy relative to an identity (no-trade) graph. The reviewers correctly identified a number of important issues — in particular, a tendency toward overclaiming a negative result that exceeds what the experimental design permits, the absence of a Bayesian VAR baseline, and inconsistencies in the treatment of the Diebold–Mariano test. We have addressed each concern systematically, as summarised below and detailed in the accompanying Response to Reviewers.

**Principal changes in the revised manuscript:**

1. **Reframing as a conditional result.** The title, abstract, introduction, and conclusion have been rewritten to reflect that all findings are explicitly conditioned on the scope of the study: 20 EU economies, quarterly frequency, a four-variable covariate set with low revision intensity, a 14-quarter initial training window, and revised data releases. We no longer claim a general negative result about GNNs or trade networks; instead, we present evidence about this specific forecasting design.

2. **Bayesian VAR baseline added.** A Minnesota-prior BVAR (VAR(1), λ₁ = 0.2, λ₂ = 0.5) has been added as a formally estimated classical baseline, with a dedicated methodology subsection (§4.3) explaining the prior structure, shrinkage parameters, and MAP estimation. BVAR results are included in all point-forecast tables and excluded from probabilistic (CRPS) comparisons, as it produces only point forecasts.

3. **DM test consistency.** The treatment of the Diebold–Mariano test has been clarified and made fully consistent throughout. Persistence is now explicitly documented as a sanity-check heuristic excluded from all formal DM comparisons, with the rationale stated precisely: it is a fixed, non-estimated rule that does not satisfy the modelling assumptions underlying the DM test. All formal DM comparisons are against estimated parametric models (ARIMA, ETS, BVAR, Ridge, TCN, LSTM, MLP) only.

4. **Real-time data wording.** The language around look-ahead bias has been revised to acknowledge that we use final revised Eurostat releases, while noting that the selected covariates are restricted to variables with comparatively low revision intensity and early availability. The phrase "prevents look-ahead bias" has been replaced throughout with language that accurately describes the prospective expanding-window design.

5. **Strengthened Limitations section.** The Limitations section now explicitly covers: the parsimonious four-variable covariate set; quarterly frequency; 20-country European panel; short initial training window; revised data releases; shared hyperparameters across architectures; and pre-specified (rather than learned) trade graphs. A concluding paragraph explicitly frames the findings as conditional on this specific experimental design.

6. **References.** Three new references have been added: Litterman (1986), Benjamini and Hochberg (1995), and Stark and Croushore (2002), alongside the previously included Bánbura, Giannone, and Reichlin (2010) and Giannone, Lenza, and Primiceri (2015).

7. **JEL codes.** JEL classification codes (C32, C33, C53, E31, E37, F14) have been added near the abstract.

8. **Reproducibility.** A formal Reproducibility Statement section has been added, confirming that all code, configuration files, processed forecast samples, and frozen evaluation outputs are publicly available and that the evaluation uses a strictly prospective expanding-window design with fixed random seeds and pre-specified model configurations.

The revised manuscript is accompanied by a detailed Response to Reviewers document that maps each reviewer concern to specific manuscript locations and the changes made.

We believe the revised manuscript makes a clearer, better-calibrated, and more defensible contribution to the literature on GNN-based macroeconomic forecasting. We hope you will find it suitable for publication.

Yours sincerely,

Rayyan Asim  
Independent Researcher  
mrayanasim09@gmail.com  
ORCID: 0000-0003-2461-5638
