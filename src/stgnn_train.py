"""Shared ST-GNN training, loss, and rolling evaluation utilities."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from src.stgnn_data import adj_to_edge_index, normalize_features
from src.stgnn_model import GCN_LSTM

SEQ_LEN = 4
CPI_FEATURE_IDX = 0
SEED = 42


@dataclass
class TrainConfig:
    lr: float = 1e-3
    dropout: float = 0.3
    gcn_layers: int = 2
    gcn_hidden: int = 64
    lstm_hidden: int = 64
    max_epochs: int = 80
    patience: int = 12
    huber_delta: float = 1.0


def set_seed(seed: int = SEED) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


def masked_huber(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    delta: float = 1.0,
) -> torch.Tensor:
    valid = mask > 0
    if valid.sum() == 0:
        return torch.tensor(0.0, device=pred.device, requires_grad=True)
    pred_v = pred[valid]
    target_v = target[valid]
    return nn.functional.smooth_l1_loss(pred_v, target_v, beta=delta)


def build_edge_lists(adj: np.ndarray, device: torch.device) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    edge_indices = []
    edge_weights = []
    for t in range(adj.shape[0]):
        ei, ew = adj_to_edge_index(adj[t])
        edge_indices.append(ei.to(device))
        edge_weights.append(ew.to(device))
    return edge_indices, edge_weights


def cpi_baseline_at(x_raw: np.ndarray, t: int, device: torch.device) -> torch.Tensor:
    return torch.tensor(x_raw[t, :, CPI_FEATURE_IDX], dtype=torch.float32, device=device)


def train_on_timesteps(
    model: GCN_LSTM,
    x_raw: np.ndarray,
    y: np.ndarray,
    target_mask: np.ndarray,
    train_timesteps: list[int],
    edge_indices: list[torch.Tensor],
    edge_weights: list[torch.Tensor],
    cfg: TrainConfig,
    device: torch.device,
    val_timesteps: list[int] | None = None,
    verbose: bool = False,
) -> float:
    x, _, _ = normalize_features(x_raw, np.array(train_timesteps))
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    best_val = float("inf")
    best_state = None
    stale = 0
    train_ts = [t for t in train_timesteps if t >= SEQ_LEN - 1]

    for epoch in range(1, cfg.max_epochs + 1):
        model.train()
        np.random.shuffle(train_ts)
        losses = []

        for t in train_ts:
            seq_idx = list(range(t - SEQ_LEN + 1, t + 1))
            x_seq = torch.tensor(x[seq_idx], dtype=torch.float32, device=device)
            ei = [edge_indices[i] for i in seq_idx]
            ew = [edge_weights[i] for i in seq_idx]
            baseline = cpi_baseline_at(x_raw, t, device)
            pred = model(x_seq, ei, ew, baseline)
            target = torch.tensor(y[t], dtype=torch.float32, device=device)
            mask = torch.tensor(target_mask[t], dtype=torch.float32, device=device)
            if mask.sum() == 0:
                continue
            loss = masked_huber(pred, target, mask, delta=cfg.huber_delta)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            losses.append(loss.item())

        if val_timesteps is not None:
            val_loss, _ = predict_timesteps(
                model, x_raw, y, target_mask, edge_indices, edge_weights,
                val_timesteps, device, norm_timesteps=train_timesteps,
            )
            if val_loss < best_val:
                best_val = val_loss
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                stale = 0
            else:
                stale += 1
                if stale >= cfg.patience:
                    break
            if verbose and epoch % 20 == 1:
                print(f"    epoch {epoch} train={np.mean(losses):.3f} val={val_loss:.3f}")

    if best_state is not None:
        model.load_state_dict(best_state)
    return best_val if val_timesteps is not None else float(np.mean(losses))


def predict_one(
    model: GCN_LSTM,
    x_raw: np.ndarray,
    y: np.ndarray,
    target_mask: np.ndarray,
    t: int,
    edge_indices: list[torch.Tensor],
    edge_weights: list[torch.Tensor],
    device: torch.device,
    norm_timesteps: list[int],
) -> tuple[np.ndarray, float]:
    if t < SEQ_LEN - 1:
        return np.full(y.shape[1], np.nan), np.nan
    x, _, _ = normalize_features(x_raw, np.array(norm_timesteps))
    model.eval()
    with torch.no_grad():
        seq_idx = list(range(t - SEQ_LEN + 1, t + 1))
        x_seq = torch.tensor(x[seq_idx], dtype=torch.float32, device=device)
        ei = [edge_indices[i] for i in seq_idx]
        ew = [edge_weights[i] for i in seq_idx]
        baseline = cpi_baseline_at(x_raw, t, device)
        pred = model(x_seq, ei, ew, baseline).cpu().numpy()
        target = y[t]
        mask = target_mask[t]
        valid = mask > 0
        if valid.sum() == 0:
            return pred, np.nan
        err = pred[valid] - target[valid]
        rmse = float(np.sqrt(np.mean(err**2)))
        return pred, rmse


def predict_timesteps(
    model: GCN_LSTM,
    x_raw: np.ndarray,
    y: np.ndarray,
    target_mask: np.ndarray,
    edge_indices: list[torch.Tensor],
    edge_weights: list[torch.Tensor],
    timesteps: list[int] | np.ndarray,
    device: torch.device,
    norm_timesteps: list[int] | None = None,
) -> tuple[float, pd.DataFrame]:
    records = []
    rmses = []
    norm_ts = norm_timesteps if norm_timesteps is not None else list(range(SEQ_LEN - 1, int(max(timesteps)) + 1))
    for t in timesteps:
        if t < SEQ_LEN - 1:
            continue
        pred, rmse = predict_one(
            model, x_raw, y, target_mask, t, edge_indices, edge_weights, device, norm_ts
        )
        if not np.isnan(rmse):
            rmses.append(rmse)
        for n, p in enumerate(pred):
            if target_mask[t, n] > 0:
                records.append({"t_idx": t, "n_idx": n, "actual": float(y[t, n]), "pred": float(p)})
    avg = float(np.mean(rmses)) if rmses else np.nan
    return avg, pd.DataFrame(records)


def rolling_eval(
    x_raw: np.ndarray,
    y: np.ndarray,
    target_mask: np.ndarray,
    edge_indices: list[torch.Tensor],
    edge_weights: list[torch.Tensor],
    eval_timesteps: list[int],
    cfg: TrainConfig,
    device: torch.device,
    countries: list[str],
    quarters: list[str],
    refit_epochs: int = 25,
    warm_start: GCN_LSTM | None = None,
    verbose: bool = False,
) -> pd.DataFrame:
    """Expanding-window refit before each forecast (matches baseline methodology)."""
    records = []
    eval_sorted = sorted(eval_timesteps)
    prev_model = warm_start

    for step, t in enumerate(eval_sorted):
        if t < SEQ_LEN - 1:
            continue
        train_end = list(range(SEQ_LEN - 1, t))
        if len(train_end) < 4:
            continue

        if prev_model is not None:
            model = GCN_LSTM(
                in_features=x_raw.shape[-1],
                gcn_hidden=cfg.gcn_hidden,
                lstm_hidden=cfg.lstm_hidden,
                gcn_layers=cfg.gcn_layers,
                dropout=cfg.dropout,
            ).to(device)
            model.load_state_dict(prev_model.state_dict())
        else:
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
        train_on_timesteps(
            model, x_raw, y, target_mask, train_end, edge_indices, edge_weights, refit_cfg, device
        )
        prev_model = model

        pred, _ = predict_one(
            model, x_raw, y, target_mask, t, edge_indices, edge_weights, device, train_end
        )
        for n, iso3 in enumerate(countries):
            if target_mask[t, n] > 0:
                records.append(
                    {
                        "iso3": iso3,
                        "quarter": quarters[t],
                        "actual": float(y[t, n]),
                        "stgnn_pred": float(pred[n]),
                    }
                )
        if verbose and (step + 1) % 5 == 0:
            print(f"  rolling step {step + 1}/{len(eval_sorted)} quarter={quarters[t]}")

    return pd.DataFrame(records)


def diebold_mariano(
    actual: np.ndarray,
    pred_a: np.ndarray,
    pred_b: np.ndarray,
    loss: str = "squared",
    h: int = 1,
) -> dict[str, float]:
    """DM test: model A vs B. Negative stat => A has lower average loss."""
    e_a = actual - pred_a
    e_b = actual - pred_b
    if loss == "squared":
        d = e_a**2 - e_b**2
    else:
        d = np.abs(e_a) - np.abs(e_b)

    d = d[np.isfinite(d)]
    T = len(d)
    if T < 5:
        return {"statistic": np.nan, "p_value": np.nan, "n": T}

    d_mean = d.mean()
    gamma0 = np.var(d, ddof=1)
    lr_var = gamma0
    for lag in range(1, h):
        w = 1 - lag / h
        cov = np.cov(d[lag:], d[:-lag], ddof=1)[0, 1]
        lr_var += 2 * w * cov

    stat = d_mean / np.sqrt(lr_var / T)
    from scipy import stats

    p_value = 2 * (1 - stats.norm.cdf(abs(stat)))
    return {"statistic": float(stat), "p_value": float(p_value), "n": int(T), "mean_diff": float(d_mean)}


def diebold_mariano_cluster_robust(
    actual: np.ndarray,
    pred_a: np.ndarray,
    pred_b: np.ndarray,
    cluster_ids: np.ndarray,
    loss: str = "squared",
    h: int = 1,
) -> dict[str, float]:
    """Cluster-robust DM test: clusters by time period to account for cross-sectional correlation.
    
    For panel data, errors are correlated across countries within each quarter.
    This clusters standard errors by quarter to get valid inference.
    """
    e_a = actual - pred_a
    e_b = actual - pred_b
    if loss == "squared":
        d = e_a**2 - e_b**2
    else:
        d = np.abs(e_a) - np.abs(e_b)

    # Remove NaN values
    mask = np.isfinite(d) & np.isfinite(cluster_ids)
    d = d[mask]
    cluster_ids = cluster_ids[mask]
    
    T = len(d)
    if T < 5:
        return {"statistic": np.nan, "p_value": np.nan, "n": T, "n_clusters": 0}
    
    d_mean = d.mean()
    
    # Cluster-robust variance estimation
    unique_clusters = np.unique(cluster_ids)
    n_clusters = len(unique_clusters)
    
    # Sum within each cluster
    cluster_sums = np.array([np.sum(d[cluster_ids == c]) for c in unique_clusters])
    
    # Compute cluster-robust variance
    cluster_var = np.var(cluster_sums, ddof=1) / n_clusters
    
    # Long-run variance adjustment for serial correlation
    lr_var = cluster_var
    for lag in range(1, h):
        w = 1 - lag / h
        cov = np.cov(cluster_sums[lag:], cluster_sums[:-lag], ddof=1)[0, 1]
        lr_var += 2 * w * cov
    
    stat = d_mean / np.sqrt(lr_var / T)
    from scipy import stats

    p_value = 2 * (1 - stats.norm.cdf(abs(stat)))
    return {
        "statistic": float(stat), 
        "p_value": float(p_value), 
        "n": int(T), 
        "n_clusters": int(n_clusters),
        "mean_diff": float(d_mean)
    }


def tost_equivalence_test(
    actual: np.ndarray,
    pred_a: np.ndarray,
    pred_b: np.ndarray,
    margin: float,
    loss: str = "squared",
    alpha: float = 0.05,
) -> dict[str, float]:
    """Two One-Sided Test (TOST) for equivalence.
    
    Tests whether the mean difference in losses is within a pre-specified equivalence margin.
    H0: |mean_diff| >= margin (not equivalent)
    H1: |mean_diff| < margin (equivalent)
    
    Returns p-value for equivalence test (reject H0 if p < alpha).
    """
    e_a = actual - pred_a
    e_b = actual - pred_b
    if loss == "squared":
        d = e_a**2 - e_b**2
    else:
        d = np.abs(e_a) - np.abs(e_b)

    d = d[np.isfinite(d)]
    T = len(d)
    if T < 5:
        return {"p_value": np.nan, "equivalence_rejected": False, "n": T, "mean_diff": np.nan}

    d_mean = d.mean()
    d_std = np.std(d, ddof=1)
    se = d_std / np.sqrt(T)
    
    from scipy import stats
    
    # Two one-sided tests
    # Test 1: H0: d_mean >= margin
    t1 = (d_mean - margin) / se
    p1 = stats.norm.cdf(t1)  # P(T <= t1) under H0: d_mean >= margin
    
    # Test 2: H0: d_mean <= -margin
    t2 = (d_mean + margin) / se
    p2 = 1 - stats.norm.cdf(t2)  # P(T >= t2) under H0: d_mean <= -margin
    
    # TOST p-value is the maximum of the two one-sided p-values
    p_tost = max(p1, p2)
    
    return {
        "p_value": float(p_tost),
        "equivalence_rejected": str(p_tost < alpha),  # True if we can claim equivalence
        "n": int(T),
        "mean_diff": float(d_mean),
        "margin": margin,
        "t1": float(t1),
        "t2": float(t2),
        "p1": float(p1),
        "p2": float(p2),
    }
