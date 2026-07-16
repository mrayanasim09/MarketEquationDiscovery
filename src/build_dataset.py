"""Milestone 4: merge macro + trade, assign splits, freeze dataset_v1."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import DATA_PROCESSED, DATA_RAW, ensure_dirs, load_protocol

FEATURE_COLS = ["cpi_yoy", "gdp_yoy", "neer_chg", "policy_rate", "energy_idx", "covid"]
VERSION = "dataset_v1"


def assign_split(quarter: str, splits: dict) -> str:
    q = pd.Period(quarter, freq="Q")
    train_end = pd.Period(splits["train_end"], freq="Q")
    val_end = pd.Period(splits["val_end"], freq="Q")
    test_end = pd.Period(splits["test_end"], freq="Q")
    if q <= train_end:
        return "train"
    if q <= val_end:
        return "val"
    if q <= test_end:
        return "test"
    return "holdout"


def build_target(df: pd.DataFrame) -> pd.DataFrame:
    """One-step-ahead CPI YoY forecast target."""
    out = df.sort_values(["iso3", "quarter"]).copy()
    out["cpi_yoy_next"] = out.groupby("iso3")["cpi_yoy"].shift(-1)
    return out


def audit_cpi_missingness(df: pd.DataFrame, threshold: float = 0.20) -> list[str]:
    miss = df.groupby("iso3")["cpi_yoy"].apply(lambda s: s.isna().mean())
    return miss[miss > threshold].index.tolist()


def build_edges(trade: pd.DataFrame) -> pd.DataFrame:
    edges = trade.copy()
    edges["weight"] = np.log1p(edges["trade_usd"].clip(lower=0))
    return edges[["quarter", "iso3_a", "iso3_b", "trade_usd", "weight"]]


def build_adjacency_snapshot(
    edges: pd.DataFrame,
    quarter: str,
    countries: list[str],
) -> np.ndarray:
    n = len(countries)
    idx = {c: i for i, c in enumerate(countries)}
    adj = np.zeros((n, n), dtype=np.float32)
    sub = edges[edges["quarter"] == quarter]
    for _, row in sub.iterrows():
        a, b = row["iso3_a"], row["iso3_b"]
        if a not in idx or b not in idx:
            continue
        w = row["weight"]
        i, j = idx[a], idx[b]
        adj[i, j] += w
        adj[j, i] += w
    return adj


def main() -> None:
    ensure_dirs()
    protocol = load_protocol()
    countries = sorted(protocol["countries"])
    out_dir = DATA_PROCESSED / VERSION
    out_dir.mkdir(parents=True, exist_ok=True)

    macro = pd.read_csv(DATA_RAW / "macro_quarterly_panel.csv")
    trade = pd.read_csv(DATA_RAW / "trade_bilateral_quarterly.csv")

    macro = macro[macro["iso3"].isin(countries)]
    macro = macro.drop(columns=["quarter_period"], errors="ignore")

    dropped = audit_cpi_missingness(macro)
    if dropped:
        print(f"  warning: countries over 20% CPI missing (keeping): {dropped}")

    macro["split"] = macro["quarter"].map(lambda q: assign_split(q, protocol["splits"]))
    macro = build_target(macro)

    feature_df = macro[
        ["iso3", "quarter", "split", "cpi_yoy", "cpi_yoy_next"]
        + [c for c in FEATURE_COLS if c != "cpi_yoy"]
    ].copy()

    edges = build_edges(trade)
    quarters = sorted(feature_df["quarter"].unique())

    # Save long-format tables
    nodes_path = out_dir / "nodes.csv"
    edges_path = out_dir / "edges.csv"
    feature_df.to_csv(nodes_path, index=False)
    edges.to_csv(edges_path, index=False)

    countries_path = out_dir / "countries.json"
    countries_path.write_text(json.dumps(countries, indent=2))

    # Stack adjacency matrices [T, N, N]
    adj_stack = np.stack(
        [build_adjacency_snapshot(edges, q, countries) for q in quarters],
        axis=0,
    )
    adj_path = out_dir / "adjacency.npy"
    np.save(adj_path, adj_stack)

    quarters_path = out_dir / "quarters.json"
    quarters_path.write_text(json.dumps(quarters, indent=2))

    miss_cols = ["cpi_yoy", "cpi_yoy_next", *[c for c in FEATURE_COLS if c != "cpi_yoy"]]
    manifest = {
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "version": VERSION,
        "protocol_version": protocol.get("protocol_version", "1.1"),
        "countries": countries,
        "n_countries": len(countries),
        "n_quarters": len(quarters),
        "quarter_range": [quarters[0], quarters[-1]],
        "features": FEATURE_COLS,
        "target": "cpi_yoy_next",
        "splits": protocol["splits"],
        "split_counts": feature_df.groupby("split").size().to_dict(),
        "missingness_pct": {
            col: round(float(feature_df[col].isna().mean()) * 100, 2) for col in miss_cols
        },
        "edges_rows": len(edges),
        "adjacency_shape": list(adj_stack.shape),
        "files": {
            "nodes": str(nodes_path),
            "edges": str(edges_path),
            "adjacency": str(adj_path),
            "countries": str(countries_path),
            "quarters": str(quarters_path),
        },
    }
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))

    print(f"Frozen {VERSION} -> {out_dir}")
    print(f"  nodes: {len(feature_df)} rows, {len(countries)} countries")
    print(f"  edges: {len(edges)} rows")
    print(f"  adjacency: {adj_stack.shape}")
    print(f"  splits: {manifest['split_counts']}")
    print(f"  saved {manifest_path}")


if __name__ == "__main__":
    main()
