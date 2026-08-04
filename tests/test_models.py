import pytest
import numpy as np
import torch
from src.models.neural import SequenceLSTM, TemporalConvNet, GraphConvolutionForecaster, TemporalGraphForecaster
from src.models.run_benchmark_engine_v2 import MLP


# ── MLP ──────────────────────────────────────────────────────────────────────
def test_mlp_forward_shape():
    model = MLP(input_dim=10, hidden_dim=16)
    x = torch.randn(5, 10)       # batch of 5 with 10 features
    out = model(x)
    assert out.shape == (5,), f"MLP output shape was {out.shape}, expected (5,)"


def test_mlp_no_nan():
    model = MLP(input_dim=8, hidden_dim=16)
    x = torch.randn(4, 8)
    out = model(x)
    assert not torch.isnan(out).any(), "MLP output contains NaN"


# ── SequenceLSTM ──────────────────────────────────────────────────────────────
def test_lstm_forward_shape():
    model = SequenceLSTM(input_dim=6, hidden_dim=16)
    x = torch.randn(4, 4, 6)    # [batch=4, seq_len=4, features=6]
    out = model(x)
    assert out.shape == (4,), f"LSTM output shape was {out.shape}, expected (4,)"


def test_lstm_no_nan():
    model = SequenceLSTM(input_dim=6, hidden_dim=16)
    x = torch.randn(3, 4, 6)
    out = model(x)
    assert not torch.isnan(out).any(), "LSTM output contains NaN"


# ── TemporalConvNet ───────────────────────────────────────────────────────────
def test_tcn_forward_shape():
    model = TemporalConvNet(input_dim=6, hidden_dim=16)
    x = torch.randn(4, 4, 6)    # [batch=4, seq_len=4, features=6]
    out = model(x)
    assert out.shape == (4,), f"TCN output shape was {out.shape}, expected (4,)"


def test_tcn_no_nan():
    model = TemporalConvNet(input_dim=6, hidden_dim=16)
    x = torch.randn(3, 4, 6)
    out = model(x)
    assert not torch.isnan(out).any(), "TCN output contains NaN"


# ── GraphConvolutionForecaster (GCN) ──────────────────────────────────────────
def test_gcn_forward_shape():
    n_nodes, input_dim, hidden_dim = 5, 8, 16
    model = GraphConvolutionForecaster(input_dim=input_dim, hidden_dim=hidden_dim)
    x = torch.randn(n_nodes, input_dim)
    adj = torch.eye(n_nodes)          # identity adjacency (simplest valid graph)
    out = model(x, adj)
    assert out.shape == (n_nodes,), f"GCN output shape was {out.shape}, expected ({n_nodes},)"


def test_gcn_no_nan():
    model = GraphConvolutionForecaster(input_dim=8, hidden_dim=16)
    x = torch.randn(5, 8)
    adj = torch.softmax(torch.rand(5, 5), dim=1)
    out = model(x, adj)
    assert not torch.isnan(out).any(), "GCN output contains NaN"


# ── TemporalGraphForecaster ───────────────────────────────────────────────────
def test_temporal_graph_forward_shape():
    k, n_nodes, input_dim, hidden_dim = 4, 5, 8, 16
    model = TemporalGraphForecaster(input_dim=input_dim, hidden_dim=hidden_dim)
    x = torch.randn(k, n_nodes, input_dim)      # [timesteps, nodes, features]
    adj = torch.eye(n_nodes).unsqueeze(0).expand(k, -1, -1)  # [k, n, n]
    out = model(x, adj)
    assert out.shape == (n_nodes,), f"TemporalGraph output shape was {out.shape}, expected ({n_nodes},)"


def test_temporal_graph_no_nan():
    k, n_nodes, input_dim, hidden_dim = 4, 5, 8, 16
    model = TemporalGraphForecaster(input_dim=input_dim, hidden_dim=hidden_dim)
    x = torch.randn(k, n_nodes, input_dim)
    adj = torch.eye(n_nodes).unsqueeze(0).expand(k, -1, -1)
    out = model(x, adj)
    assert not torch.isnan(out).any(), "TemporalGraph output contains NaN"
