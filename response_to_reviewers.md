# Response to Reviewers

**Manuscript:** Temporal Architecture or Trade Topology? A Prospective Ablation Study of Spatio-Temporal Graph Neural Networks for Quarterly CPI Inflation Forecasting  
**Authors:** Rayyan Asim  
**Revision:** Major Revision  
**Date:** August 2026

---

We thank the reviewer for a thorough and constructive assessment. Below we provide a point-by-point response to each concern raised.

---

## Summary Table

| # | Reviewer Concern | Location in Revised MS | Change Made |
|---|---|---|---|
| 1 | **Overclaiming a negative result.** The original framing implied a universal conclusion about GNNs and trade networks, beyond what the experimental design supports. | §1 (Boundary conditions, para. 4); §7 (Conclusion); Abstract; Title | Title reworded to emphasise the ablation scope. Abstract and Introduction now state explicit boundary conditions: 20 EU economies, quarterly frequency, 4-variable set, 14-quarter initial window, revised data. Conclusion rewritten to frame all findings as conditional on the experimental design. |
| 2 | **Missing strong macro baseline (BVAR/shrinkage).** No Bayesian VAR or equivalent shrinkage estimator was included, making it impossible to assess GNN gains relative to a strong classical benchmark. | §4.3 (new section); Tables 3, 6 | Added a Minnesota-prior BVAR baseline (VAR(1), λ₁ = 0.2, λ₂ = 0.5). New §4.3 explains prior structure, shrinkage parameters, and MAP estimation. BVAR point forecasts are included in Table 3 (MAE/RMSE rankings) and Table 6 (DM tests). BVAR is excluded from Table 5 (CRPS) as it produces point forecasts only. A caveat notes that the BVAR is heavily regularised due to the short initial window. |
| 3 | **Sparse feature set and short initial training window.** The 4-variable covariate set and 14-quarter initial window limit the generalisability of results. | §4.0 (new section — Scope of the Feature Set); §6.5 (Limitations, items 2 and 3) | Added new §4.0 explaining the practical justification for the parsimonious feature set (availability, low revision intensity). Limitations §6.5 now explicitly lists: short initial window (item 2), revised vintages (item 3), and the restricted covariate scope. Closing paragraph in §6.5 restates that findings are conditional on the experimental scope. |
| 4 | **Inconsistencies regarding Diebold–Mariano (DM) comparators.** It was unclear whether Persistence was included in or excluded from formal DM tests, and the rationale was inconsistent across sections. | §5.4 (Part A description); §6.7 (Claim-to-evidence table, row 2) | Persistence is now consistently described as a sanity-check benchmark only. A standardised sentence appears in §5.4: *"Persistence is retained in the descriptive forecast rankings as a sanity-check benchmark only. Because it is a fixed, non-estimated forecasting rule, it does not satisfy the modelling assumptions underlying the Diebold–Mariano test and is therefore excluded from all formal Diebold–Mariano comparisons."* The claim-to-evidence table (§6.7) notes this exclusion explicitly in the Caveats column. |
| 5 | **Speculative claims in the discussion.** Several sentences in the Discussion section made inferences that went beyond what the ablation design can establish. | §6.2 (Mechanism interpretation); §6.3; Results §5.2 | Causal language has been removed or qualified throughout. The sentence referring to the temporal attention mechanism as a "driver" has been replaced with: *"This pattern is consistent with the hypothesis that the GNN architecture's temporal attention mechanism, rather than trade-graph topology, is the primary source of any marginal improvement over simpler baselines — though this remains an associational interpretation."* Two alternative hypotheses (architectural regularisation; temporally coarse edges) are now presented explicitly as hypotheses, not conclusions. |
| 6 | **Real-time data / look-ahead bias wording.** The phrase "prevents look-ahead bias" overstated the real-time integrity of the data, since final revised Eurostat releases are used. | §4.1 (Evaluation Protocol); §6.5 (item 3 — Revised vintages); Abstract; §6.7 (Claim-to-evidence table, row 1) | The phrase "prevents any form of look-ahead bias" has been replaced with "aligns with the information constraints of a genuine forecasting exercise." Abstract and §4.0 now state: *"the selected covariates are limited to variables with comparatively low revision intensity and early availability."* Limitations item 3 now explicitly notes that we use final revised releases (upper-bound accuracy environment) and recommends a full real-time vintage evaluation as future work. |
| 7 | **Missing references.** Key references for BVAR methodology, multiple testing correction, and real-time data were absent. | §4.3; §4.5; §6.5; References section | Added: Litterman (1986); Benjamini and Hochberg (1995); Stark and Croushore (2002). Verified presence of: Bánbura, Giannone, and Reichlin (2010); Giannone, Lenza, and Primiceri (2015); Harvey, Leybourne, and Newbold (1997); Diebold and Mariano (1995). |
| 8 | **Missing JEL codes and submission metadata.** | YAML frontmatter; preamble of LaTeX source | JEL codes C32, C33, C53, E31, E37, F14 added near the abstract in both Markdown and LaTeX sources. |

---

## Detailed Responses

### Concern 1: Overclaiming the Negative Result

> *The reviewer noted that the paper's framing implied that trade-network topology is uninformative for inflation forecasting in general, whereas the study only covers a specific, narrow experimental setting.*

**Response:** We fully agree. The original framing was too strong. In the revised manuscript, every major claim section begins with an explicit restatement of the boundary conditions. The title now includes the phrase "A Prospective Ablation Study" to signal methodological scope. The Introduction (§1) now contains a dedicated paragraph listing five boundary conditions that qualify every finding. The Conclusion (§7) has been rewritten to avoid any inference beyond the experimental scope. The Limitations section (§6.5) closes with the following caveat:

> *"These findings are conditional on the experimental scope. The restricted covariate set, quarterly frequency, European panel, short initial training window, pre-specified trade graphs, and revised data releases may all affect the relative ranking of models. The results should therefore be interpreted as evidence about this specific forecasting design, not as a universal statement about trade networks or GNNs."*

---

### Concern 2: Missing BVAR/Shrinkage Baseline

> *The reviewer noted that BVAR and shrinkage models are standard in the macro forecasting literature and their absence made it impossible to assess GNN performance against competitive baselines.*

**Response:** We have added a Minnesota-prior BVAR as a formally estimated baseline. The new §4.3 (Bayesian Vector Autoregression Baseline) explains the model specification: VAR(1) in the same four variables as the neural models, with Minnesota prior hyperparameters λ₁ = 0.2 (own-lag tightness) and λ₂ = 0.5 (cross-lag tightness). We note that, given the short initial training window (14 quarters), the BVAR is heavily regularised; its relative performance should be interpreted accordingly.

BVAR point forecasts are now included in the point-forecast comparison tables (Table 3: MAE/RMSE) and in the DM-test table (Table 6). Since BVAR produces only point predictions, it is correctly excluded from Table 5 (CRPS probabilistic comparisons).

---

### Concern 3: Sparse Feature Set and Short Initial Training Window

> *The reviewer noted that the 4-variable set excludes important inflation predictors and that the 14-quarter initial window is short for neural models.*

**Response:** We have added a new §4.0 (Scope of the Feature Set) explaining the practical rationale: all four variables are publicly available at quarterly frequency for all 20 EU economies throughout the evaluation period, and they were selected for comparatively low revision intensity and early availability, which is important for any forward-looking application. We acknowledge explicitly that models with richer covariate sets — including output gaps, wage growth, and monetary policy variables — may yield different topology-versus-architecture rankings.

The short initial window is now item 2 in the Limitations section (§6.5), with an explicit note that GNN and LSTM error rates in early test origins partly reflect model initialisation quality rather than intrinsic architecture differences.

---

### Concern 4: Diebold–Mariano Consistency

> *The reviewer identified inconsistencies in how Persistence was treated in the DM testing framework.*

**Response:** We have standardised the treatment of Persistence across the entire manuscript. The following sentence now appears in §5.4 (Statistical Significance):

> *"Persistence is retained in the descriptive forecast rankings as a sanity-check benchmark only. Because it is a fixed, non-estimated forecasting rule, it does not satisfy the modelling assumptions underlying the Diebold–Mariano test and is therefore excluded from all formal Diebold–Mariano comparisons."*

This same caveat is noted in the claim-to-evidence table (§6.7). We have verified that all DM test results in Table 6 compare against estimated parametric models only: ARIMA, ETS, BVAR, Ridge, TCN, LSTM, and MLP.

---

### Concern 5: Speculative Claims in the Discussion

> *The reviewer flagged that some mechanistic explanations in the Discussion went beyond what the ablation can establish.*

**Response:** We have reviewed all mechanistic language and replaced causal phrasing with explicitly associational framing. In particular:
- "strongly implicates ... as the driver" → "is consistent with the hypothesis that ... is the primary source ... though this remains an associational interpretation"
- Two alternative hypotheses (architectural regularisation; temporally coarse edges) are now framed as hypotheses to be tested in future work, not as conclusions of this study
- The claim-to-evidence table (§6.7, row 4) now labels the mechanism finding as "associational, not causal"

---

*We believe the revised manuscript is substantially stronger and more carefully calibrated. We look forward to the editors' and reviewers' further assessment.*
