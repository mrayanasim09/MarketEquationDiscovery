# Replication Package: Inflation Contagion ST-GNN

This archive contains all code required to reproduce the empirical results in *Predicting Inflation Contagion: A Spatio-Temporal Graph Neural Network Approach to Global Trade*.

## Requirements

- Python 3.11+
- ~4 GB disk space for processed data (excluding raw BACI bulk)

## Setup

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Data (not included in this zip)

### Macro data (automatic)

```bash
python -m src.download_macro
python -m src.validate_macro
```

Downloads FRED CPI, World Bank controls, and Brent energy prices into `data/raw/`.

### CEPII BACI trade data (manual, one-time)

1. Register and download the CEPII BACI bulk CSV from http://www.cepii.fr/CEPII/en/bdd_modele/presentation.asp?id=37
2. Extract to project root as `BACI_HS96_V202601/` **or** place under `data/raw/bulk/`
3. Run:

```bash
python -m src.download_trade_baci
python -m src.build_graph
```

The BACI files are several GB and are **not** redistributed in this zip.

## Full pipeline

```bash
python -m src.build_dataset        # Milestone 4: freeze dataset_v1
python -m src.run_baselines        # Milestone 5: ARIMA + VAR
python -m src.train_stgnn          # Milestone 6: initial ST-GNN
python -m src.run_evaluation       # Milestone 7: grid search, rolling refit, DM tests
python -m src.run_explainability   # Milestone 8: Integrated Gradients case studies
bash paper/build.sh                # Milestone 9: compile PDFs + replication zip
```

## Expected outputs

| Path | Description |
|------|-------------|
| `results/baselines/metrics.json` | ARIMA/VAR benchmark metrics |
| `results/evaluation/evaluation_report.json` | ST-GNN test metrics + DM tests |
| `results/explainability/explainability_report.json` | IG case studies |
| `results/explainability/figures/*.png` | Explainability charts |

## Random seeds

Fixed at `numpy=42`, `torch=42` (see `src/stgnn_train.py`).

## Protocol

Design decisions are locked in `docs/research_protocol.md` and `config/protocol.yaml`.
