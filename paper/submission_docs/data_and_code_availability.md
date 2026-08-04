# Data and Code Availability

## Data

All data used in this study are sourced from publicly available repositories:

- **CPI/HICP (target variable and energy component):** Eurostat HICP database
  — https://ec.europa.eu/eurostat/data/database
- **Bilateral trade flows:** Eurostat Comext quarterly trade statistics
  — https://ec.europa.eu/eurostat/web/international-trade-in-goods/data/database
- **Macroeconomic indicators:** IMF International Financial Statistics (IFS)
  — https://www.imf.org/en/Data
- **Energy / commodity prices:** World Bank commodity price data
  — https://www.worldbank.org/en/research/commodity-markets

Processed datasets for v2.1 are included in the repository under
`data/processed/v2/` and are SHA256-verified via `data/processed/v2/contract.json`.
No proprietary or restricted data sources were used.

## Code

Full source code and all experiment outputs (forecasts, metrics, DM test results)
are available at:

**GitHub:** [Anonymized for blind review]

All results are reproducible end-to-end using the provided `reproduce.sh` script.
SHA256 hashes of all outputs are pre-registered in `experiments/results/v2_1/contract/`
and verified automatically during reproduction.

## Archival

The code and processed datasets are recommended for archival at Zenodo prior to
final publication. A permanent DOI will be added here once archived.

## License

- **Code:** MIT (see `LICENSE` in the repository root)
- **Data:** Subject to original provider terms (Eurostat, IMF, World Bank — all open/public licences)