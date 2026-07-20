# Statistical Testing Methodology

## Overview
Due to the overlapping nature of multi-step forecasts and the inherent autocorrelation in quarterly macroeconomic data, standard statistical tests (e.g., Diebold-Mariano) suffer from severe size distortions. To ensure robust conclusions, this repository implements a specialized testing pipeline.

## Harvey-Leybourne-Newbold (HLN) Diebold-Mariano Test
The core statistical engine utilizes the HLN modification of the Diebold-Mariano test. This adjustment corrects for small-sample bias, which is particularly relevant given our test window of 35 quarters.

### Bartlett HAC Kernel
To account for serial correlation in the forecast error differentials up to horizon $h-1$, we apply a Heteroskedasticity and Autocorrelation Consistent (HAC) estimator using a Bartlett kernel.

## Robust Confidence Intervals
In addition to asymptotic p-values, we compute moving-block bootstrap confidence intervals for the test statistics. The block length is selected dynamically based on the forecast horizon ($L = h + 1$) to preserve the dependence structure of the errors.

## Multiple Testing Correction
Given the vast combinatorial space of evaluations (12 models $\times$ 8 graph variants $\times$ 3 horizons $\times$ 20 seeds), the probability of Type I errors (false positives) is significantly inflated.
- We apply the **Benjamini-Hochberg False Discovery Rate (FDR)** correction across all pairwise model comparisons within a given horizon.
- Significance is established only if the FDR-corrected p-value remains below $\alpha = 0.05$.

## Interpretation Guidelines
When analyzing the `statistics/` directory outputs:
- A significant positive `dm_stat` indicates the competitor model significantly outperforms the baseline.
- Focus on `p_value_fdr` rather than `p_value_raw` to draw definitive conclusions.
- The empirical results highlighted in the paper demonstrate that while isolated runs showed significance, the ensemble performance across 20 seeds failed to surpass the FDR thresholds.
