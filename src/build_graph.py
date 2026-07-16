"""Build adjacency matrices from trade_bilateral_quarterly.csv (smoke test)."""

from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd

from src.config import DATA_RAW, load_protocol


def build_adjacency_for_quarter(
    trade: pd.DataFrame,
    quarter: str,
    countries: list[str],
) -> np.ndarray:
    n = len(countries)
    idx = {c: i for i, c in enumerate(countries)}
    adj = np.zeros((n, n), dtype=float)

    sub = trade[trade["quarter"] == quarter]
    for _, row in sub.iterrows():
        a, b = row["iso3_a"], row["iso3_b"]
        if a not in idx or b not in idx:
            continue
        w = np.log1p(row["trade_usd"])
        i, j = idx[a], idx[b]
        adj[i, j] += w
        adj[j, i] += w

    return adj


def symmetric_normalize(adj: np.ndarray) -> np.ndarray:
    adj = adj + np.eye(len(adj)) * 1e-8
    deg = adj.sum(axis=1)
    d_inv_sqrt = np.diag(1.0 / np.sqrt(deg + 1e-8))
    return d_inv_sqrt @ adj @ d_inv_sqrt


def main() -> int:
    trade_path = DATA_RAW / "trade_bilateral_quarterly.csv"
    if not trade_path.exists():
        print(f"ERROR: {trade_path} not found. Run python -m src.download_trade_baci first.")
        return 1

    protocol = load_protocol()
    countries = sorted(protocol["countries"])
    trade = pd.read_csv(trade_path)

    test_quarter = "2020Q1"
    adj = build_adjacency_for_quarter(trade, test_quarter, countries)
    norm = symmetric_normalize(adj)

    nonzero_edges = int((adj > 0).sum() // 2)
    density = nonzero_edges / (len(countries) * (len(countries) - 1) / 2)

    report = {
        "passed": nonzero_edges > 0 and not np.isnan(norm).any(),
        "quarter": test_quarter,
        "nodes": len(countries),
        "nonzero_undirected_edges": nonzero_edges,
        "density": round(float(density), 4),
        "adj_sum": round(float(adj.sum()), 2),
    }

    out_path = DATA_RAW / "graph_validation.json"
    out_path.write_text(json.dumps(report, indent=2))

    if not report["passed"]:
        print(f"FAILED: graph validation for {test_quarter}")
        return 1

    print(f"PASSED: adjacency for {test_quarter}")
    print(f"  nodes={report['nodes']} edges={report['nonzero_undirected_edges']} density={report['density']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
