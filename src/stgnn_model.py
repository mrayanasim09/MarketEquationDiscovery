"""GCN + LSTM spatio-temporal model for inflation forecasting."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class GCN_LSTM(nn.Module):
    def __init__(
        self,
        in_features: int,
        gcn_hidden: int = 64,
        lstm_hidden: int = 64,
        gcn_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.gcn_layers = nn.ModuleList()
        self.gcn_layers.append(GCNConv(in_features, gcn_hidden, normalize=False))
        for _ in range(gcn_layers - 1):
            self.gcn_layers.append(GCNConv(gcn_hidden, gcn_hidden, normalize=False))
        self.dropout = dropout
        self.lstm = nn.LSTM(gcn_hidden, lstm_hidden, batch_first=True)
        self.head = nn.Linear(lstm_hidden, 1)
        nn.init.zeros_(self.head.bias)

    def gcn_forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
    ) -> torch.Tensor:
        h = x
        for i, conv in enumerate(self.gcn_layers):
            h = conv(h, edge_index, edge_weight=edge_weight)
            if i < len(self.gcn_layers) - 1:
                h = F.relu(h)
                h = F.dropout(h, p=self.dropout, training=self.training)
        return h

    def forward(
        self,
        x_seq: torch.Tensor,
        edge_indices: list[torch.Tensor],
        edge_weights: list[torch.Tensor],
        cpi_baseline: torch.Tensor,
    ) -> torch.Tensor:
        """
        x_seq: [seq_len, N, F]
        cpi_baseline: [N] current-quarter CPI YoY (persistence anchor)
        Returns next-quarter CPI YoY predictions [N].
        """
        gcn_out = []
        for t in range(x_seq.shape[0]):
            h = self.gcn_forward(x_seq[t], edge_indices[t], edge_weights[t])
            gcn_out.append(h)
        gcn_seq = torch.stack(gcn_out, dim=0)
        node_seq = gcn_seq.permute(1, 0, 2)
        out, _ = self.lstm(node_seq)
        delta = self.head(out[:, -1, :]).squeeze(-1)
        return cpi_baseline + delta
