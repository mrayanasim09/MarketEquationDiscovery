# Release Notes - v2.1.0

## V2.1 Journal Benchmark Release

This release corresponds to the final codebase utilized for the manuscript "Graph Neural Networks for Macroeconomic Forecasting" submitted for publication.

### Summary of the Benchmark

Version 2.1.0 executes a comprehensive prospective forecasting evaluation across 20 European economies. It utilizes Graph Neural Networks to model international trade networks and capture cross-border inflationary spillovers. The evaluation relies on an expanding window framework to mimic realistic macroeconomic forecasting scenarios.

### Key Statistics & Reproducibility

The benchmark pipeline generates an extensive set of results, encompassing:
- **38,380** independent model fits across 20 seeds and 8 graph variants.
- **781,740** individual point and probabilistic forecasts.
- Rigorous statistical evaluation using moving-block bootstrap Diebold-Mariano tests.

To guarantee reproducibility, researchers can utilize the provided `reproduce.sh` or `reproduce.ps1` scripts.

### Verification Hashes (SHA-256)

Upon successful replication (using Git commit `a7fef297346868ddf2ae178e45d47ab27cc55317`), the generated output artifacts should perfectly match the following cryptographic hashes:

*   `forecasts.parquet` (781,740 rows): `f988dfb1249fd77739ffdecae6323073c8cbd363c48412d4e3248454f98b3798`
*   `metrics.parquet` (9,288 rows): `231b349f6ab2d5bba4cc42cf047c2cedd88300f5836302b5e79e4ff9071abcaa`
*   `dm_tests.parquet` (6,720 rows): `a57490a079580c48458143961f9e8dde6f7cf72e77818a71246d73496c397048`

### Download Instructions

1. Clone the repository at the specific commit:
   ```bash
   git clone https://github.com/mrayanasim/gnn-macro-forecasting.git
   cd gnn-macro-forecasting
   git checkout a7fef297346868ddf2ae178e45d47ab27cc55317
   ```
2. Execute the reproduction script corresponding to your operating system.
