Data and Code Availability

Data
- Processed datasets are included in the `data/processed/dataset_v1` folder of this repository under `repository/`.
- External sources:
  - CEPII BACI trade data: http://www.cepii.fr/CEPII/en/bdd_modele/bdd_modele.asp
  - FRED CPI and Brent crude series: https://fred.stlouisfed.org/
  - World Bank indicators: https://data.worldbank.org/

Code
- Code to reproduce all experiments and tables is in this repository under `src/` and `experiments/`.
- The exact commit used for the manuscript is available at the repository snapshot (provide commit hash or DOI when archiving).

Archival and DOIs
- We recommend archiving the code and processed data at Zenodo or OSF and adding the DOI here before submission.

Access
- If parts of the raw trade dataset are too large to host, we will provide a script `src/download_trade_baci.py` to fetch and pre-process the needed slices of BACI.

License
- Code: MIT (see repository/LICENSE)
- Data: subject to original provider terms (CEPII, FRED, World Bank).