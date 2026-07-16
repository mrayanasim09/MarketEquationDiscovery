"""Milestone 7: Hyperparameter tuning, rolling refit ST-GNN, Diebold-Mariano tests."""

from __future__ import annotations

import itertools
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import torch

from src.config import DATA_PROCESSED, RESULTS, ensure_dirs, load_protocol
from src.run_baselines import compute_metrics
from src.stgnn_data import load_panel_tensors, timesteps_for_split
from src.stgnn_model import GCN_LSTM
from src.stgnn_train import (
    TrainConfig,
    build_edge_lists,
    diebold_mariano,
    diebold_mariano_cluster_robust,
    predict_timesteps,
    rolling_eval,
    set_seed,
    tost_equivalence_test,
    train_on_timesteps,
)

OUT_DIR = RESULTS / "evaluation"
STGNN_DIR = RESULTS / "stgnn"
BASELINE_DIR = RESULTS / "baselines"

GRID = {
    "lr": [5e-4, 1e-3, 3e-3],
    "dropout": [0.2, 0.3, 0.4],
    "gcn_layers": [1, 2],
}


def grid_search(
    x_raw: np.ndarray,
    y: np.ndarray,
    target_mask: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    edge_indices: list,
    edge_weights: list,
    device: torch.device,
    in_features: int,
) -> TrainConfig:
    best_cfg = None
    best_val_rmse = float("inf")
    results = []

    combos = list(itertools.product(GRID["lr"], GRID["dropout"], GRID["gcn_layers"]))
    print(f"Phase 1a: Grid search ({len(combos)} configs on val RMSE)...")

    for lr, dropout, gcn_layers in combos:
        set_seed()
        cfg = TrainConfig(lr=lr, dropout=dropout, gcn_layers=gcn_layers, max_epochs=60, patience=10)
        model = GCN_LSTM(in_features, cfg.gcn_hidden, cfg.lstm_hidden, cfg.gcn_layers, cfg.dropout).to(device)
        train_on_timesteps(
            model, x_raw, y, target_mask, list(train_idx), edge_indices, edge_weights, cfg, device,
            val_timesteps=list(val_idx),
        )

        _, val_preds = predict_timesteps(
            model, x_raw, y, target_mask, edge_indices, edge_weights, list(val_idx), device,
            norm_timesteps=list(train_idx),
        )
        if val_preds.empty:
            val_rmse = float("inf")
        else:
            m = compute_metrics(val_preds["actual"], val_preds["pred"])
            val_rmse = m["rmse"]
        results.append({"lr": lr, "dropout": dropout, "gcn_layers": gcn_layers, "val_rmse": val_rmse})
        print(f"  lr={lr} dropout={dropout} gcn_layers={gcn_layers} -> val RMSE={val_rmse:.3f}")

        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_cfg = cfg

    (OUT_DIR / "grid_search.json").write_text(json.dumps(results, indent=2))
    print(f"  best val RMSE={best_val_rmse:.3f} cfg={best_cfg}")
    return best_cfg


def main() -> None:
    ensure_dirs()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    STGNN_DIR.mkdir(parents=True, exist_ok=True)
    set_seed()
    load_protocol()

    data = load_panel_tensors()
    countries = data["countries"]
    quarters = data["quarters"]
    split = data["split"]

    train_idx = timesteps_for_split(split, "train")
    val_idx = timesteps_for_split(split, "val")
    test_idx = timesteps_for_split(split, "test")

    x_raw = data["x"]
    y = data["y"]
    target_mask = data["target_mask"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    edge_indices, edge_weights = build_edge_lists(data["adj"], device)

    best_cfg = grid_search(
        x_raw, y, target_mask, train_idx, val_idx, edge_indices, edge_weights, device, x_raw.shape[-1]
    )

    print("\nPhase 1b: Rolling refit evaluation (val + test)...")
    eval_idx = sorted(list(val_idx) + list(test_idx))
    stgnn_fc = rolling_eval(
        x_raw, y, target_mask, edge_indices, edge_weights,
        eval_idx, best_cfg, device, countries, quarters,
        refit_epochs=30, verbose=True,
    )

    nodes = pd.read_csv(DATA_PROCESSED / "dataset_v1" / "nodes.csv")
    split_map = nodes.drop_duplicates("quarter").set_index("quarter")["split"]
    stgnn_fc["split"] = stgnn_fc["quarter"].map(split_map)
    stgnn_fc.to_csv(STGNN_DIR / "forecasts_rolling.csv", index=False)

    test_fc = stgnn_fc[stgnn_fc["split"] == "test"]
    val_fc = stgnn_fc[stgnn_fc["split"] == "val"]
    test_metrics = compute_metrics(test_fc["actual"], test_fc["stgnn_pred"])
    val_metrics = compute_metrics(val_fc["actual"], val_fc["stgnn_pred"])
    test_no_arg = test_fc[test_fc["iso3"] != "ARG"]
    test_metrics_no_arg = compute_metrics(test_no_arg["actual"], test_no_arg["stgnn_pred"])

    print(f"\nRolling refit ST-GNN test: RMSE={test_metrics['rmse']:.3f} MAE={test_metrics['mae']:.3f}")
    print(f"  (excl ARG): RMSE={test_metrics_no_arg['rmse']:.3f}")

    print("\nPhase 2: Diebold-Mariano tests...")
    baseline_fc = pd.read_csv(BASELINE_DIR / "forecasts.csv")
    merged = stgnn_fc.merge(
        baseline_fc[["iso3", "quarter", "arima_pred", "var_pred"]],
        on=["iso3", "quarter"],
        how="inner",
    )
    merged = merged[merged["split"] == "test"].dropna(subset=["stgnn_pred", "arima_pred", "var_pred"], how="any")

    # Create cluster IDs for quarter-based clustering
    unique_quarters = sorted(merged["quarter"].unique())
    quarter_to_cluster = {q: i for i, q in enumerate(unique_quarters)}
    cluster_ids = merged["quarter"].map(quarter_to_cluster).values

    # Standard DM tests (for comparison)
    dm_arima = diebold_mariano(
        merged["actual"].values, merged["stgnn_pred"].values, merged["arima_pred"].values, loss="squared"
    )
    dm_var = diebold_mariano(
        merged["actual"].values, merged["stgnn_pred"].values, merged["var_pred"].values, loss="squared"
    )
    dm_arima_mae = diebold_mariano(
        merged["actual"].values, merged["stgnn_pred"].values, merged["arima_pred"].values, loss="absolute"
    )
    
    # Load baseline metrics before using them
    baseline_metrics = json.loads((BASELINE_DIR / "metrics.json").read_text())
    
    # Cluster-robust DM tests (accounts for cross-sectional correlation)
    dm_arima_cr = diebold_mariano_cluster_robust(
        merged["actual"].values, merged["stgnn_pred"].values, merged["arima_pred"].values, 
        cluster_ids=cluster_ids, loss="squared"
    )
    dm_var_cr = diebold_mariano_cluster_robust(
        merged["actual"].values, merged["stgnn_pred"].values, merged["var_pred"].values,
        cluster_ids=cluster_ids, loss="squared"
    )
    
    # TOST equivalence test with margin based on 10% of ARIMA RMSE
    arima_rmse = baseline_metrics["arima"]["metrics"]["test"]["rmse"]
    equiv_margin = 0.1 * arima_rmse  # 10% margin
    tost_arima = tost_equivalence_test(
        merged["actual"].values, merged["stgnn_pred"].values, merged["arima_pred"].values,
        margin=equiv_margin, loss="squared"
    )
    
    # Apples-to-apples comparison: same country sets
    # All models excluding Argentina
    merged_no_arg = merged[merged["iso3"] != "ARG"]
    cluster_ids_no_arg = merged_no_arg["quarter"].map(quarter_to_cluster).values
    
    dm_arima_no_arg = diebold_mariano(
        merged_no_arg["actual"].values, merged_no_arg["stgnn_pred"].values, 
        merged_no_arg["arima_pred"].values, loss="squared"
    )
    dm_var_no_arg = diebold_mariano(
        merged_no_arg["actual"].values, merged_no_arg["stgnn_pred"].values, 
        merged_no_arg["var_pred"].values, loss="squared"
    )
    
    # Compute apples-to-apples metrics for ST-GNN (excluding Argentina)
    stgnn_test_no_arg = test_fc[test_fc["iso3"] != "ARG"]
    stgnn_metrics_no_arg = compute_metrics(stgnn_test_no_arg["actual"], stgnn_test_no_arg["stgnn_pred"])

    report = {
        "run_at": datetime.now(timezone.utc).isoformat(),
        "milestone": 7,
        "phase1": {
            "loss": "Huber (Smooth L1)",
            "persistence_skip": True,
            "expanding_window_norm": True,
            "best_config": best_cfg.__dict__,
            "rolling_refit": True,
            "refit_epochs_per_step": 30,
        },
        "metrics": {
            "stgnn_rolling_val": val_metrics,
            "stgnn_rolling_test": test_metrics,
            "stgnn_rolling_test_excl_ARG": test_metrics_no_arg,
            "arima_test": baseline_metrics["arima"]["metrics"]["test"],
            "var_test": baseline_metrics["var"]["metrics"]["test"],
        },
        "diebold_mariano_test": {
            "stgnn_vs_arima_squared": dm_arima,
            "stgnn_vs_arima_absolute": dm_arima_mae,
            "stgnn_vs_var_squared": dm_var,
            "interpretation": "Negative DM stat => ST-GNN has lower average loss",
        },
        "diebold_mariano_cluster_robust": {
            "stgnn_vs_arima_squared": dm_arima_cr,
            "stgnn_vs_var_squared": dm_var_cr,
            "interpretation": "Cluster-robust by quarter to account for cross-sectional correlation",
        },
        "tost_equivalence_test": {
            "stgnn_vs_arima_squared": tost_arima,
            "margin": equiv_margin,
            "margin_pct_of_arima_rmse": 10.0,
            "interpretation": "TOST tests equivalence within pre-specified margin; p < 0.05 supports equivalence claim",
        },
        "apples_to_apples_comparison": {
            "all_models_excluding_argentina": {
                "stgnn_metrics": stgnn_metrics_no_arg,
                "arima_metrics": baseline_metrics["arima"]["metrics_excl_arg"]["test"],
                "var_metrics": baseline_metrics["var"]["metrics"]["test"],
                "dm_stgnn_vs_arima": dm_arima_no_arg,
                "dm_stgnn_vs_var": dm_var_no_arg,
            },
        },
        "target_rmse_under_4": test_metrics["rmse"] < 4.0,
        "paired_test_rows": len(merged),
    }
    (OUT_DIR / "evaluation_report.json").write_text(json.dumps(report, indent=2))
    merged.to_csv(OUT_DIR / "paired_forecasts_test.csv", index=False)

    print(f"\nSaved {OUT_DIR / 'evaluation_report.json'}")
    print("\n--- Test metrics ---")
    print(f"  ST-GNN (rolling): RMSE={test_metrics['rmse']:.3f} MAE={test_metrics['mae']:.3f}")
    print(f"  ARIMA:            RMSE={baseline_metrics['arima']['metrics']['test']['rmse']:.3f}")
    print(f"  VAR:              RMSE={baseline_metrics['var']['metrics']['test']['rmse']:.3f}")
    print("\n--- Apples-to-apples comparison (excluding Argentina) ---")
    print(f"  ST-GNN: RMSE={stgnn_metrics_no_arg['rmse']:.3f} MAE={stgnn_metrics_no_arg['mae']:.3f}")
    print(f"  ARIMA:  RMSE={baseline_metrics['arima']['metrics_excl_arg']['test']['rmse']:.3f} MAE={baseline_metrics['arima']['metrics_excl_arg']['test']['mae']:.3f}")
    print(f"  VAR:    RMSE={baseline_metrics['var']['metrics']['test']['rmse']:.3f} MAE={baseline_metrics['var']['metrics']['test']['mae']:.3f}")
    print("\n--- Diebold-Mariano (ST-GNN vs ARIMA, squared error) ---")
    print(f"  Standard: stat={dm_arima['statistic']:.3f}  p={dm_arima['p_value']:.4f}  n={dm_arima['n']}")
    print(f"  Cluster-robust: stat={dm_arima_cr['statistic']:.3f}  p={dm_arima_cr['p_value']:.4f}  n={dm_arima_cr['n']}  n_clusters={dm_arima_cr['n_clusters']}")
    print("\n--- TOST Equivalence Test (ST-GNN vs ARIMA) ---")
    print(f"  Margin: {equiv_margin:.3f} (10% of ARIMA RMSE)")
    print(f"  p-value: {tost_arima['p_value']:.4f}")
    print(f"  Equivalence supported (p<0.05): {tost_arima['equivalence_rejected']}")
    print(f"  Mean difference: {tost_arima['mean_diff']:.4f}")
    print(f"  Target RMSE<4.0: {'YES' if report['target_rmse_under_4'] else 'NO'}")


if __name__ == "__main__":
    main()
