"""Graph variants computed solely from the persisted quarterly trade snapshots.

The eight variants are:
- directed_trade: Raw bilateral export values, row-normalised.
- log_trade: Log-transformed exports, row-normalised.
- import_dependence: A(i,j) = exports from i to j / total exports received by j.
- top_k_incoming: Retain only the top-5 import partners per country (k=5 by
  default; set to 5 rather than 3 to include meaningful partners for smaller
  economies with fewer large trading partners).
- reversed: Transpose of directed_trade (import perspective).
- undirected: Symmetrised: (A + A^T).
- degree_preserving_random: Random rewiring preserving degree sequence.
- identity_no_trade: Identity matrix (no cross-country edges; critical ablation).
"""
from __future__ import annotations

import numpy as np

GRAPH_VARIANTS = (
    "directed_trade",
    "log_trade",
    "import_dependence",
    "top_k_incoming",
    "reversed",
    "undirected",
    "degree_preserving_random",
    "identity_no_trade",
)


def row_normalize(a: np.ndarray) -> np.ndarray:
    """Row-normalise adjacency matrix; rows with zero sum are left as-is."""
    total = a.sum(axis=1, keepdims=True)
    return a / np.where(total > 0, total, 1.0)


def top_k_incoming(a: np.ndarray, k: int = 5) -> np.ndarray:
    """Retain only the top-k exporting partners for each importer.

    Default k=5 to include meaningful partners for smaller economies.
    The manuscript and methods appendix refer to this variant as
    'top_k_incoming (k=5)'.
    """
    out = np.zeros_like(a)
    for importer in range(a.shape[1]):
        keep = np.argsort(a[:, importer])[-k:]
        out[keep, importer] = a[keep, importer]
    return out


def degree_preserving_random(a: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Shuffle positive weights among existing directed edges, preserving degree/support."""
    out = np.zeros_like(a)
    mask = a > 0
    values = a[mask].copy()
    rng.shuffle(values)
    out[mask] = values
    return out


def build(raw_trade: np.ndarray, variant: str, rng: np.random.Generator, top_k: int = 5) -> np.ndarray:
    """Construct a row-normalised adjacency matrix for a given graph variant.

    Args:
        raw_trade: 2-D array of shape (n_countries, n_countries) with bilateral export flows.
        variant: One of GRAPH_VARIANTS.
        rng: NumPy random generator for stochastic variants.
        top_k: Number of top import partners to retain for 'top_k_incoming'. Default 5.

    Returns:
        Row-normalised adjacency array of shape (n_countries, n_countries).
    """
    a = np.asarray(raw_trade, dtype=float).copy()
    if variant == "directed_trade":
        out = a
    elif variant == "log_trade":
        out = np.log1p(a)
    elif variant == "import_dependence":
        # exporter->importer share of the importer's observed trade exposure
        out = a / np.where(a.sum(axis=0, keepdims=True) > 0, a.sum(axis=0, keepdims=True), 1.0)
    elif variant == "top_k_incoming":
        out = top_k_incoming(a, top_k)
    elif variant == "reversed":
        out = a.T
    elif variant == "undirected":
        out = a + a.T
    elif variant == "degree_preserving_random":
        out = degree_preserving_random(a, rng)
    elif variant == "identity_no_trade":
        out = np.eye(a.shape[0], dtype=float)
    else:
        raise ValueError(f"unknown graph variant: {variant}")
    # Add self-loops to prevent GNN from discarding own-node lagged features
    out = out + np.eye(a.shape[0], dtype=float)
    return row_normalize(out)
