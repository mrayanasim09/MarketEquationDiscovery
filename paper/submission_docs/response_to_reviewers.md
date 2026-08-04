# Response to Reviewers

**Manuscript:** Temporal Architecture or Trade Topology? A Prospective Ablation Study of Graph Neural Networks for Inflation Forecasting  
**Authors:** Rayyan Asim  
**Revision:** Major Revision (Round 2)  
**Date:** August 2026

---

We thank the reviewers and the editorial team for their thorough and constructive assessments across both rounds of review. Below we provide a detailed point-by-point response to all concerns raised, mapping each to specific changes in the revised manuscript.

---

## Summary Table of Revisions

| # | Reviewer Concern | Location in Revised MS | Change Made |
|---|---|---|---|
| 1 | **Overclaiming a negative result.** The original framing implied a universal conclusion about GNNs/trade networks beyond our experimental boundary conditions. | §1; §7 (Conclusion); Abstract; Title | Rewrote abstract and conclusion to frame findings strictly as a conditional ablation benchmark. Added five explicit boundary conditions to §1. |
| 2 | **Missing strong macro baseline (BVAR/shrinkage).** No Bayesian VAR baseline was included to benchmark GNN performance against standard macro models. | §4.3; Tables 3, 6 | Added a Minnesota-prior BVAR baseline (VAR(1), λ₁ = 0.2, λ₂ = 0.5, MAP estimation) in §4.3. Included BVAR in all point forecast and DM test tables. |
| 3 | **Sparse feature set and short initial window.** The 4-variable set and 14-quarter window limit model capacity and generalisability. | §4.0; §6.5 (item 2) | Added §4.0 explaining the practical justification for the covariate set (availability, low-revision intensity). Listed short training window as a limitation in §6.5 (item 2). |
| 4 | **Diebold–Mariano comparator consistency.** Inconsistent treatment of Persistence in formal DM statistical tests. | §5.4; §6.7; Table 6 | Explicitly documented Persistence as a fixed, non-estimated rule excluded from formal DM tests. Verified Table 6 compares against estimated models only. |
| 5 | **Speculative claims in the Discussion.** Speculative explanations of underlying propagation mechanisms without direct empirical testing. | §6.1 | Consolidated §6.1 and §6.2 into a single Section 6.1. Replaced causal explanations with explicitly labeled, untested hypotheses (architectural regularisation, edge redundancy, elasticity). |
| 6 | **Look-ahead bias and real-time wording.** Overstated real-time integrity given the use of revised Eurostat current-release snapshots. | §4.1; §6.5 (item 3); Abstract | Replaced "prevents look-ahead bias" with "aligns with information constraints of a genuine forecasting exercise." Noted upper-bound forecast environment in §6.5. |
| 7 | **Missing references.** Absence of standard references for BVAR, multiple testing correction, and real-time data. | References section; §4.3; §4.5 | Added Litterman (1986), Benjamini and Hochberg (1995), and Stark and Croushore (2002). |
| 8 | **Submission metadata.** Missing keywords and JEL codes. | Frontmatter; preamble | Added JEL codes C32, C33, C53, E31, E37, F14 and standard keywords near the abstract. |
| 9 | **Long and awkward title.** The original title risked sounding like "two papers stapled together." | Title (p. 1) | Shortened and streamlined the title to: *"Temporal Architecture or Trade Topology? A Prospective Ablation Study of Graph Neural Networks for Inflation Forecasting."* |
| 10 | **Abstract focus and significance.** The abstract focused heavily on statistical machinery and lacked practical significance for the forecasting community. | Abstract (p. 1) | Completely reframed the abstract to lead with the negative-result hook and highlight why the forecasting audience should care (i.e. establishing the identity-graph control as a standard requirement to prevent over-parameterised misattribution). |
| 11 | **Motivation for trade networks.** The paper lacked a theoretical defense for why trade flows should be expected to predict inflation over common factors. | §2.5 (new section) | Added §2.5 (*Theoretical Motivation: Why Trade Networks Should Predict Inflation*) detailing the import price pass-through and supply chain integration channels. |
| 12 | **Shared hyperparameter ceilings.** Shared hyperparameters across architectures restrict claims of fair performance ceilings. | §4.2; §6.5 (item 1) | Added detailed disclaimers in §4.2 and §6.5 (item 1) acknowledging that holding hyperparameters constant prevents us from claiming absolute performance ceilings for any model. |
| 13 | **Interval calibration prominence.** Empirical 80% coverage (50–58%) is poor, yet presented as a primary contribution. | §5.3; §6.5 (item 6) | Added explicit caveats in §5.3 and §6.5 (item 6) stating that due to poor interval calibration, the probabilistic results are secondary, descriptive metrics rather than a primary contribution. |
| 14 | **Robustness check gaps.** Lack of sensitivity tests on window size or data vintage structure. | §6.5 (item 8) | Added item 8 to §6.5 explicitly documenting the lack of window size or data vintage sensitivity checks as a boundary condition of our findings. |
| 15 | **Reference domain mismatch.** Garcia-Martos et al. (2015) was cited for macro regularisation but is an electricity forecasting paper. | Literature Review; References | Replaced Garcia-Martos with Medeiros et al. (2021) and Coulombe et al. (2020), which are directly relevant to macro forecasting. Removed Garcia-Martos from the bibliography. |

---

## Detailed Responses to Round 2 Concerns

### Concern 9: Title Streamlining
*The reviewer noted that the title was long and slightly awkward, risking sounding like two papers stapled together.*

**Response:** We have shortened the title to make it punchier and more integrated: *"Temporal Architecture or Trade Topology? A Prospective Ablation Study of Graph Neural Networks for Inflation Forecasting."* We removed the redundant "Spatio-Temporal" and "Quarterly CPI" details, which are now properly introduced in the abstract and introduction.

### Concern 10: Abstract Focus and Significance
*The reviewer suggested that the abstract focused too much on statistical machinery and did not explain why the negative result is practically significant for a forecasting audience.*

**Response:** We have rewritten the abstract to lead directly with the negative ablation result and clarify its methodological importance. The abstract now explicitly frames the contribution for the forecasting community: establishing that spatial or network-based models should not be assumed to capture meaningful spillovers without passing an identity-graph ablation control. This highlights the risk of over-parameterisation and sets a new control standard for macroeconomic network forecasting.

### Concern 11: Trade Network Motivation
*The reviewer noted that the motivation leaned on macro intuition rather than outlining why bilateral trade volumes should be expected to improve forecasts relative to isolated autoregressions or common factors.*

**Response:** We have added a new subsection **§2.5 Theoretical Motivation: Why Trade Networks Should Predict Inflation** in the literature review. This section formally outlines the two primary transmission channels—Import Price Pass-Through and Global Supply Chain Integration—and explains why a GNN's asymmetric, network-structured propagation is theoretically superior to closed-economy autoregressive models or symmetric common-factor models (which treat exposures uniformly).

### Concern 12: Shared Hyperparameters and Performance Ceilings
*The reviewer highlighted that using shared hyperparameters across neural models prevents us from claiming that we have tested their true empirical performance ceilings.*

**Response:** We agree and have added explicit disclaimers in the Model Families section (§4.2) and Limitations (§6.5, item 1). We state clearly that holding hyperparameters constant is a deliberate choice necessary to isolate the architectural contribution (our primary goal), but it implies that our results do not reflect the peak performance of any single model under independent, architecture-specific tuning.

### Concern 13: Interval Calibration Prominence
*The reviewer noted that the empirical coverage of the 80% intervals is very poor (50–58%) and should not be presented as a primary contribution.*

**Response:** We have added caveats in §5.3 (Probabilistic Results) and §6.5 (Limitations, item 6). We now explicitly state that due to the undercoverage (driven by the short training window and bootstrap assumptions), the probabilistic results must be interpreted as secondary, descriptive comparisons rather than a primary contribution, and we suggest conformal prediction as the necessary path forward.

### Concern 14: Sensitivity / Robustness Checks
*The reviewer requested a robustness design check, such as varying the training window size or data vintages.*

**Response:** We have added a dedicated section in §6.5 (Limitations, item 8) acknowledging that the sensitivity of our model rankings to the initial window size (14 quarters) and dataset vintage is unverified and remains a boundary condition of our findings.

### Concern 15: Reference Audit
*The reviewer identified a domain mismatch with Garcia-Martos et al. (2015), which is an electricity price forecasting paper, cited for regularisation methods in macroeconomics.*

**Response:** We have removed Garcia-Martos et al. (2015) from the text and bibliography. We replaced it with Medeiros et al. (2021) and Coulombe et al. (2020), which are directly relevant to macroeconomic and inflation regularisation. We also added Stark and Croushore (2002) as a citation in §4.0 to support our discussion of real-time data and vintage limitations.
