"""Integrated Gradients attribution for ST-GNN trade-network spillovers."""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch

from src.build_dataset import FEATURE_COLS
from src.stgnn_data import adj_to_edge_index, normalize_features
from src.stgnn_model import GCN_LSTM
from src.stgnn_train import SEQ_LEN, TrainConfig, cpi_baseline_at, set_seed, train_on_timesteps

IG_STEPS = 32


@dataclass
class ExplainConfig:
    quarter: str
    target_iso3: str
    t_idx: int


def load_best_train_config(report_path) -> TrainConfig:
    report = json.loads(report_path.read_text())
    cfg = report["phase1"]["best_config"]
    return TrainConfig(
        lr=cfg["lr"],
        dropout=cfg["dropout"],
        gcn_layers=cfg["gcn_layers"],
        gcn_hidden=cfg["gcn_hidden"],
        lstm_hidden=cfg["lstm_hidden"],
        max_epochs=cfg.get("max_epochs", 60),
        patience=cfg.get("patience", 10),
        huber_delta=cfg.get("huber_delta", 1.0),
    )


def refit_model(
    x_raw: np.ndarray,
    y: np.ndarray,
    target_mask: np.ndarray,
    t_idx: int,
    edge_indices: list[torch.Tensor],
    edge_weights: list[torch.Tensor],
    cfg: TrainConfig,
    device: torch.device,
    refit_epochs: int = 30,
) -> GCN_LSTM:
    train_end = list(range(SEQ_LEN - 1, t_idx))
    model = GCN_LSTM(
        in_features=x_raw.shape[-1],
        gcn_hidden=cfg.gcn_hidden,
        lstm_hidden=cfg.lstm_hidden,
        gcn_layers=cfg.gcn_layers,
        dropout=cfg.dropout,
    ).to(device)
    refit_cfg = TrainConfig(
        lr=cfg.lr,
        dropout=cfg.dropout,
        gcn_layers=cfg.gcn_layers,
        gcn_hidden=cfg.gcn_hidden,
        lstm_hidden=cfg.lstm_hidden,
        max_epochs=refit_epochs,
        patience=8,
        huber_delta=cfg.huber_delta,
    )
    set_seed()
    train_on_timesteps(
        model, x_raw, y, target_mask, train_end, edge_indices, edge_weights, refit_cfg, device
    )
    return model


def _prepare_inputs(
    x_raw: np.ndarray,
    t_idx: int,
    norm_timesteps: list[int],
    edge_indices: list[torch.Tensor],
    edge_weights: list[torch.Tensor],
    device: torch.device,
) -> tuple[torch.Tensor, list[torch.Tensor], list[torch.Tensor], torch.Tensor]:
    x, _, _ = normalize_features(x_raw, np.array(norm_timesteps))
    seq_idx = list(range(t_idx - SEQ_LEN + 1, t_idx + 1))
    x_seq = torch.tensor(x[seq_idx], dtype=torch.float32, device=device)
    ei = [edge_indices[i] for i in seq_idx]
    ew = [edge_weights[i].clone().detach() for i in seq_idx]
    baseline = cpi_baseline_at(x_raw, t_idx, device)
    return x_seq, ei, ew, baseline


def integrated_gradients_edges(
    model: GCN_LSTM,
    x_seq: torch.Tensor,
    edge_indices: list[torch.Tensor],
    edge_weights: list[torch.Tensor],
    cpi_baseline: torch.Tensor,
    target_idx: int,
    steps: int = IG_STEPS,
) -> list[torch.Tensor]:
    """IG attribution on edge weights for a single-node prediction."""
    baselines = [torch.zeros_like(w) for w in edge_weights]
    avg_grads = [torch.zeros_like(w) for w in edge_weights]

    for step in range(1, steps + 1):
        alpha = step / steps
        interp = []
        for b, actual in zip(baselines, edge_weights):
            w = (b + alpha * (actual - b)).detach().clone()
            w.requires_grad_(True)
            interp.append(w)

        model.zero_grad()
        pred = model(x_seq, edge_indices, interp, cpi_baseline)
        pred[target_idx].backward()

        for i, w in enumerate(interp):
            if w.grad is not None:
                avg_grads[i] += w.grad.detach() / steps

    attrs = []
    for actual, base, grad in zip(edge_weights, baselines, avg_grads):
        attrs.append((actual - base).detach() * grad)
    return attrs


def integrated_gradients_features(
    model: GCN_LSTM,
    x_seq: torch.Tensor,
    edge_indices: list[torch.Tensor],
    edge_weights: list[torch.Tensor],
    cpi_baseline: torch.Tensor,
    target_idx: int,
    steps: int = IG_STEPS,
) -> torch.Tensor:
    """IG attribution on input features [seq, N, F] for target-node prediction."""
    baseline = torch.zeros_like(x_seq)
    avg_grad = torch.zeros_like(x_seq)

    for step in range(1, steps + 1):
        alpha = step / steps
        interp = (baseline + alpha * (x_seq - baseline)).detach().clone()
        interp.requires_grad_(True)
        model.zero_grad()
        pred = model(interp, edge_indices, edge_weights, cpi_baseline)
        pred[target_idx].backward()
        if interp.grad is not None:
            avg_grad += interp.grad.detach() / steps

    return (x_seq - baseline).detach() * avg_grad


def edge_attributions_to_frame(
    edge_attrs: list[torch.Tensor],
    edge_indices: list[torch.Tensor],
    seq_quarters: list[str],
    countries: list[str],
    target_iso3: str,
    target_idx: int,
) -> pd.DataFrame:
    records = []
    for step, (attr, ei, quarter) in enumerate(zip(edge_attrs, edge_indices, seq_quarters)):
        src = ei[0].cpu().numpy()
        dst = ei[1].cpu().numpy()
        vals = attr.detach().cpu().numpy()
        for s, d, v in zip(src, dst, vals):
            if d != target_idx:
                continue
            records.append(
                {
                    "quarter_in_seq": quarter,
                    "seq_step": step,
                    "source_iso3": countries[s],
                    "target_iso3": target_iso3,
                    "edge_attr": float(v),
                    "abs_edge_attr": float(abs(v)),
                }
            )
    frame = pd.DataFrame(records)
    if frame.empty:
        return frame
    agg = (
        frame.groupby(["source_iso3", "target_iso3"], as_index=False)
        .agg(edge_attr=("edge_attr", "sum"), abs_edge_attr=("abs_edge_attr", "sum"))
        .sort_values("abs_edge_attr", ascending=False)
    )
    return agg


def feature_attributions_summary(
    feat_attr: torch.Tensor,
    countries: list[str],
    target_idx: int,
    feature_names: list[str],
) -> dict:
    """Split domestic (target node) vs foreign (all other nodes) feature attribution."""
    arr = feat_attr.detach().cpu().numpy()
    domestic = float(np.abs(arr[:, target_idx, :]).sum())
    foreign = float(np.abs(arr[:, [i for i in range(arr.shape[1]) if i != target_idx], :]).sum())
    per_feature = {
        feature_names[f]: float(np.abs(arr[:, target_idx, f]).sum()) for f in range(len(feature_names))
    }
    return {
        "domestic_abs_attr": domestic,
        "foreign_abs_attr": foreign,
        "spillover_ratio": foreign / (domestic + 1e-8),
        "per_feature_domestic": per_feature,
    }


def explain_timestep(
    model: GCN_LSTM,
    x_raw: np.ndarray,
    t_idx: int,
    target_idx: int,
    countries: list[str],
    quarters: list[str],
    edge_indices: list[torch.Tensor],
    edge_weights: list[torch.Tensor],
    device: torch.device,
    feature_names: list[str] | None = None,
) -> dict:
    feature_names = feature_names or FEATURE_COLS
    norm_ts = list(range(SEQ_LEN - 1, t_idx))
    x_seq, ei, ew, baseline = _prepare_inputs(
        x_raw, t_idx, norm_ts, edge_indices, edge_weights, device
    )
    target_iso3 = countries[target_idx]
    seq_quarters = [quarters[i] for i in range(t_idx - SEQ_LEN + 1, t_idx + 1)]

    model.eval()
    edge_attrs = integrated_gradients_edges(model, x_seq, ei, ew, baseline, target_idx)
    feat_attr = integrated_gradients_features(model, x_seq, ei, ew, baseline, target_idx)

    edges = edge_attributions_to_frame(
        edge_attrs, ei, seq_quarters, countries, target_iso3, target_idx
    )
    feat_summary = feature_attributions_summary(feat_attr, countries, target_idx, feature_names)
    edge_total = float(edges["abs_edge_attr"].sum()) if not edges.empty else 0.0

    with torch.no_grad():
        pred = model(x_seq, ei, ew, baseline)[target_idx].item()

    return {
        "quarter": quarters[t_idx],
        "target_iso3": target_iso3,
        "prediction": pred,
        "edge_attributions": edges,
        "feature_summary": feat_summary,
        "edge_total_abs": edge_total,
        "trade_vs_domestic_ratio": edge_total / (feat_summary["domestic_abs_attr"] + 1e-8),
    }


def find_spillover_country(
    model: GCN_LSTM,
    x_raw: np.ndarray,
    t_idx: int,
    countries: list[str],
    quarters: list[str],
    edge_indices: list[torch.Tensor],
    edge_weights: list[torch.Tensor],
    device: torch.device,
) -> tuple[str, dict]:
    """Country with highest trade-edge attribution relative to domestic features."""
    best_iso3 = countries[0]
    best_ratio = -1.0
    best_detail = {}

    for idx, iso3 in enumerate(countries):
        detail = explain_timestep(
            model, x_raw, t_idx, idx, countries, quarters, edge_indices, edge_weights, device
        )
        ratio = detail["trade_vs_domestic_ratio"]
        if ratio > best_ratio:
            best_ratio = ratio
            best_iso3 = iso3
            best_detail = detail

    return best_iso3, best_detail
