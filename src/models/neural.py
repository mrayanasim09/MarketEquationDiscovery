"""Proper sequence models for the reconstructed v2 benchmark engine."""
from __future__ import annotations
import torch
from torch import nn

class SequenceLSTM(nn.Module):
    """Consumes [batch, k, features] histories ending at the allowed t-1 quarter."""
    def __init__(self,input_dim:int,hidden_dim:int):
        super().__init__(); self.lstm=nn.LSTM(input_dim,hidden_dim,batch_first=True); self.head=nn.Linear(hidden_dim,1)
    def forward(self,x:torch.Tensor)->torch.Tensor:
        return self.head(self.lstm(x)[0][:,-1]).squeeze(-1)

class TemporalConvNet(nn.Module):
    """Causal temporal convolution over the same lagged feature sequence as LSTM."""
    def __init__(self,input_dim:int,hidden_dim:int):
        super().__init__(); self.net=nn.Sequential(nn.Conv1d(input_dim,hidden_dim,2),nn.ReLU(),nn.Conv1d(hidden_dim,hidden_dim,2),nn.ReLU()); self.head=nn.Linear(hidden_dim,1)
    def forward(self,x:torch.Tensor)->torch.Tensor:
        h=self.net(x.transpose(1,2)); return self.head(h[:,:,-1]).squeeze(-1)

class GraphConvolutionForecaster(nn.Module):
    """One lagged graph snapshot and contemporaneously permitted node features."""
    def __init__(self,input_dim:int,hidden_dim:int):
        super().__init__(); self.node=nn.Linear(input_dim,hidden_dim); self.head=nn.Linear(hidden_dim,1)
    def forward(self,x:torch.Tensor,adj:torch.Tensor)->torch.Tensor:
        return self.head(torch.relu(adj@self.node(x))).squeeze(-1)

class TemporalGraphForecaster(nn.Module):
    """Applies each lagged graph to its matching node features, then an LSTM in time."""
    def __init__(self,input_dim:int,hidden_dim:int):
        super().__init__(); self.node=nn.Linear(input_dim,hidden_dim); self.lstm=nn.LSTM(hidden_dim,hidden_dim,batch_first=True); self.head=nn.Linear(hidden_dim,1)
    def forward(self,x:torch.Tensor,adj:torch.Tensor)->torch.Tensor:
        # x [k,n,f], adj [k,n,n]; no snapshot after t-1 is accepted by engine.
        hs=torch.stack([torch.relu(adj[i]@self.node(x[i])) for i in range(x.shape[0])],dim=1)
        return self.head(self.lstm(hs)[0][:,-1]).squeeze(-1)
