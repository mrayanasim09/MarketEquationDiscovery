# Data Description

## Overview
This study utilizes a balanced panel dataset of 20 European countries, spanning from 2011Q2 to 2025Q3. The dataset merges bilateral trade flow statistics with macroeconomic indicators to construct dynamic spatio-temporal graphs.

## Country Panel
The 20 countries included (ISO-3 codes):
AUT, BEL, BGR, CYP, CZE, DEU, ESP, EST, FIN, FRA, GBR, GRC, HUN, IRL, ITA, LTU, LVA, NLD, POL, PRT.

## Primary Data Sources

### Eurostat Comext
- **Content**: Quarterly bilateral trade flows (Imports and Exports).
- **Usage**: Used to define the topological structure of the graph networks. Trade volumes are utilized as edge weights connecting sovereign nodes.

### IMF International Financial Statistics (IFS) & World Bank
- **Content**: Consumer Price Indices (CPI), Energy Price Indices.
- **Usage**: Derived as the primary node features and the target variable (inflation rate).

## Feature Sets
Two distinct feature sets are evaluated:
1. `cpi_energy_sequence`: Historical lagged values of domestic CPI and energy indices.
2. `cpi_energy_volatility_trade_exposure`: Includes the base sequence plus historical volatility metrics and aggregate trade exposure metrics (total imports/exports over GDP).

## Processing Pipeline
1. **Temporal Alignment**: All monthly series are aggregated to quarterly frequencies using end-of-period values for stock variables and averages for flow variables.
2. **Differencing**: CPI is transformed into Quarter-over-Quarter (QoQ) percentage changes to achieve stationarity.
3. **Imputation**: Missing bilateral trade flows (rare within the EU panel) are interpolated using linear splines.
4. **Normalization**: Features are cross-sectionally standardized (Z-score) using only data available up to the current training window ($t-1$) to prevent data leakage.

## Cryptographic Hashes
Raw data integrity is validated against pre-computed SHA-256 hashes defined in the configuration, ensuring consistency across reproduction attempts.
