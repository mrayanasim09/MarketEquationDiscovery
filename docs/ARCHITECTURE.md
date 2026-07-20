# System Architecture

## Modular Layout
The repository is structured as a modular Python package located in `src/`. This design ensures separation of concerns between data ingestion, model definition, and benchmark execution.

### `src/acquisition/`
Handles raw data retrieval from external sources (Eurostat Comext APIs, IMF bulk downloads). Includes rate-limiting and local caching logic.

### `src/ingestion/`
Responsible for parsing raw data files, standardizing country codes to ISO-3 formats, and handling missing value interpolation. 

### `src/transform/`
Contains the core feature engineering pipelines. 
- Generates the target variable (Quarter-over-Quarter inflation).
- Constructs the feature sets: `cpi_energy_sequence` and `cpi_energy_volatility_trade_exposure`.
- Produces normalized tensors for neural network consumption.

### `src/models/`
The core modeling directory containing:
- `baselines.py`: Deterministic and classical machine learning models.
- `graphs/`: Adjacency matrix construction (`graph_builder.py`) and network variants.
- `neural/`: PyTorch implementations of MLP, LSTM, TCN, GCN, and Temporal Graph architectures.
- `tuning/`: Optuna-based hyperparameter optimization routines.

### `src/benchmark/`
The evaluation engine.
- `engine.py`: Implements the expanding window logic.
- `evaluator.py`: Computes deterministic and probabilistic scoring rules.
- `storage.py`: Manages transactional writes to Parquet files to prevent data corruption during long runs.

## Data Pipeline Flow
1. Raw CSV/SDMX files are loaded by `ingestion/`.
2. `transform/` applies temporal alignments and scaling.
3. `models/graphs/` builds static and dynamic adjacency matrices for the specified quarterly slice.
4. `benchmark/engine.py` orchestrates the training, prediction, and scoring loop over the predefined grid of models, horizons, and random seeds.
