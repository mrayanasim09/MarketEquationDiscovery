# Research Limitations

## Panel Scope
The empirical analysis is strictly confined to 20 highly integrated European countries. The dense regulatory and monetary alignment within the Eurozone and broader EU may suppress the variance in cross-border inflation dynamics. The findings regarding graph network efficacy may not generalize to more heterogeneous global trade networks (e.g., emerging markets vs. developed economies).

## Data Revisions
The benchmark utilizes final revised macroeconomic data (as available in late 2025). Real-time forecasting in practice relies on preliminary releases, which are subsequently revised. The absence of a real-time vintage database implies that the absolute predictive performance observed may be slightly optimistic compared to operational deployment.

## Training Window Length
Neural network architectures typically require vast amounts of data to converge effectively. The initial training window (2011Q2-2014Q4) is exceedingly short for deep learning models. While the expanding window approach gradually mitigates this, the initial test phase (2017-2019) performance is heavily constrained by data scarcity.

## Causal Claims
The spatio-temporal models capture associative predictive correlations, not structural causality. The identification of predictive superiority via trade linkages does not constitute proof of mechanistic contagion channels.

## Computational Requirements
Executing the full 20-seed evaluation across 13 models and expanding windows requires substantial computational resources. The requirement for repeated training limits the depth of hyperparameter tuning that could be performed for the deep learning models compared to deterministic baselines.

## Baseline Convergence
Certain deterministic baselines, notably VAR and dynamic factor models, exhibited numerical instability and convergence warnings during the earliest windows of the expanding evaluation due to low degrees of freedom.
