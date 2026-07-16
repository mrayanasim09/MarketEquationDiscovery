"""Milestone 6: Train ST-GNN (delegates to stgnn_train; use run_evaluation for M7)."""

from __future__ import annotations

from pathlib import Path

import torch

from src.config import RESULTS, ensure_dirs
from src.stgnn_data import load_panel_tensors, timesteps_for_split
from src.stgnn_model import GCN_LSTM
from src.stgnn_train import TrainConfig, build_edge_lists, set_seed, train_on_timesteps

OUT_DIR = RESULTS / "stgnn"


def main() -> None:
    ensure_dirs()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    set_seed()

    data = load_panel_tensors()
    train_idx = timesteps_for_split(data["split"], "train")
    val_idx = timesteps_for_split(data["split"], "val")

    x_raw = data["x"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    edge_indices, edge_weights = build_edge_lists(data["adj"], device)

    cfg = TrainConfig(lr=1e-3, dropout=0.3, gcn_layers=2)
    model = GCN_LSTM(x_raw.shape[-1], cfg.gcn_hidden, cfg.lstm_hidden, cfg.gcn_layers, cfg.dropout).to(device)
    train_on_timesteps(
        model, x_raw, data["y"], data["target_mask"], list(train_idx), edge_indices, edge_weights,
        cfg, device, val_timesteps=list(val_idx), verbose=True,
    )

    torch.save({"state_dict": model.state_dict(), "config": cfg.__dict__}, OUT_DIR / "model.pt")
    print(f"Saved {OUT_DIR / 'model.pt'} — run python -m src.run_evaluation for Milestone 7")


if __name__ == "__main__":
    main()
