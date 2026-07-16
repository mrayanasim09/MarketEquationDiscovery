"""Load dataset_v1 tensors for ST-GNN training."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.build_dataset import FEATURE_COLS
from src.config import DATA_PROCESSED

DATASET_DIR = DATA_PROCESSED / "dataset_v1"


def load_panel_tensors() -> dict:
    nodes = pd.read_csv(DATASET_DIR / "nodes.csv")
    countries = json.loads((DATASET_DIR / "countries.json").read_text())
    quarters = json.loads((DATASET_DIR / "quarters.json").read_text())
    adj = np.load(DATASET_DIR / "adjacency.npy").astype(np.float32)

    nodes["quarter_period"] = nodes["quarter"].apply(lambda q: pd.Period(q, freq="Q"))
    nodes = nodes.sort_values(["quarter_period", "iso3"])

    T, N = len(quarters), len(countries)
    F = len(FEATURE_COLS)

    x = np.zeros((T, N, F), dtype=np.float32)
    y = np.full((T, N), np.nan, dtype=np.float32)
    target_mask = np.zeros((T, N), dtype=np.float32)
    split_arr = np.empty(T, dtype=object)

    for t, quarter in enumerate(quarters):
        split_arr[t] = nodes.loc[nodes["quarter"] == quarter, "split"].iloc[0]
        for n, iso3 in enumerate(countries):
            row = nodes[(nodes["quarter"] == quarter) & (nodes["iso3"] == iso3)]
            if row.empty:
                continue
            row = row.iloc[0]
            for f, col in enumerate(FEATURE_COLS):
                val = row[col]
                x[t, n, f] = float(val) if pd.notna(val) else 0.0
            if pd.notna(row["cpi_yoy_next"]):
                y[t, n] = float(row["cpi_yoy_next"])
                target_mask[t, n] = 1.0

    return {
        "x": x,
        "y": y,
        "target_mask": target_mask,
        "adj": adj,
        "countries": countries,
        "quarters": quarters,
        "split": split_arr,
        "feature_names": FEATURE_COLS,
    }


def normalize_features(x: np.ndarray, train_idx: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Z-score using train timesteps only; NaN positions already zero."""
    train_x = x[train_idx]
    mean = train_x.mean(axis=(0, 1), keepdims=True)
    std = train_x.std(axis=(0, 1), keepdims=True)
    std = np.where(std < 1e-8, 1.0, std)
    x_norm = (x - mean) / std
    return x_norm.astype(np.float32), mean.astype(np.float32), std.astype(np.float32)


def normalize_adjacency(adj: np.ndarray) -> np.ndarray:
    """Symmetric normalization D^{-1/2} A D^{-1/2} with light self-loop."""
    a = adj.copy()
    np.fill_diagonal(a, 0)
    a += np.eye(a.shape[0], dtype=np.float32) * 1e-3
    deg = a.sum(axis=1)
    inv_sqrt = np.zeros_like(deg, dtype=np.float64)
    np.power(deg, -0.5, out=inv_sqrt, where=deg > 0)
    inv_sqrt[~np.isfinite(inv_sqrt)] = 0.0
    d = np.diag(inv_sqrt)
    return (d @ a @ d).astype(np.float32)


def adj_to_edge_index(adj: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert weighted adjacency [N,N] to PyG edge_index + edge_weight."""
    a = normalize_adjacency(adj)
    rows, cols = np.where(a > 0)
    if len(rows) == 0:
        n = a.shape[0]
        rows = np.arange(n)
        cols = np.arange(n)
        weights = np.ones(n, dtype=np.float32)
    else:
        weights = a[rows, cols].astype(np.float32)
    edge_index = torch.tensor(np.stack([rows, cols], axis=0), dtype=torch.long)
    edge_weight = torch.tensor(weights, dtype=torch.float32)
    return edge_index, edge_weight


def timesteps_for_split(split_arr: np.ndarray, split_name: str) -> np.ndarray:
    return np.where(split_arr == split_name)[0]
