Reproducibility Guide

This document explains how to reproduce the experiments and figures in the manuscript.

Environment
- Python 3.10+ recommended
- Create a virtual environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

- Optionally use `conda` with `environment.yml` in repository root.

Data
- Raw data files are stored under `data/raw` (CEPII BACI, FRED, World Bank).
- Processed datasets used for training and evaluation are under `data/processed/dataset_v1`.
- If external data must be downloaded, run `python download_macro.py` and `python download_trade_baci.py` in `src/`.

Running experiments
- To reproduce baseline and ST-GNN training and evaluation run:

```bash
python -m src.train_stgnn --config experiments/configs/v2_benchmark.json
```

- To reproduce results tables and figures used in the manuscript:

```bash
python src/build_paper_tables.py --results-dir experiments/results/v2_1
python src/build_graph.py --save-dir experiments/results/v2_1/graphs
```

Random seeds
- Experiments use a fixed random seed `42` by default; set `SEED` env var to override.

Computational notes
- Full rolling-refit experiments require substantial time (GPU recommended for ST-GNN training).

Contact
- For reproduction issues contact mrayanasim09@gmail.com