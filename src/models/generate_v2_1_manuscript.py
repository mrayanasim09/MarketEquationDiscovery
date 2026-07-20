"""Generate Tables 1-6 (.tex and .csv) and plots under experiments/results/v2_1/manuscript/."""
from __future__ import annotations

import sys
import json
from pathlib import Path
import pandas as pd

# Use Agg backend for matplotlib to prevent GUI errors in headless environment
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from src.models.storage_v2_1 import RESULTS  # noqa: E402

def to_latex_table(df: pd.DataFrame, path: Path) -> None:
    latex = df.to_latex(index=False, float_format="%.4f")
    path.write_text(latex)

def main() -> int:
    print("Starting manuscript artifact generation...")
    manuscript_dir = RESULTS / "manuscript"
    manuscript_dir.mkdir(parents=True, exist_ok=True)

    # Load required data
    forecast_samples_path = ROOT / "data/processed/v2/forecast_samples.csv"
    countries_path = ROOT / "data/processed/v2/countries.json"
    tuning_manifest_path = RESULTS / "tuning" / "tuning_manifest.json"
    forecasts_path = RESULTS / "forecasts.parquet"
    metrics_path = RESULTS / "metrics.parquet"
    dm_path = RESULTS / "dm_tests.parquet"

    # Make sure we have the required results files
    if not (forecasts_path.exists() and metrics_path.exists() and dm_path.exists()):
        print("Error: Missing result files (forecasts, metrics, dm_tests parquet files). Execute benchmark first.")
        return 1

    forecasts = pd.read_parquet(forecasts_path)
    metrics = pd.read_parquet(metrics_path)
    dm_tests = pd.read_parquet(dm_path)
    samples = pd.read_csv(forecast_samples_path)
    countries = json.loads(countries_path.read_text())
    tuning = json.loads(tuning_manifest_path.read_text()) if tuning_manifest_path.exists() else {}

    # ----------------------------------------------------
    # Table 1: Dataset Summary
    # ----------------------------------------------------
    print("Generating Table 1: Dataset Summary...")
    t1_rows = []
    for country in sorted(countries):
        c_samples = samples[samples.country == country]
        t1_rows.append({
            "Country": country,
            "Total Obs": len(c_samples),
            "Train Obs": len(c_samples[c_samples.split == "train"]),
            "Val Obs": len(c_samples[c_samples.split == "validation"]),
            "Test Obs": len(c_samples[c_samples.split == "test"]),
            "Mean CPI YoY": float(c_samples.target_cpi_yoy.mean()),
            "Std CPI YoY": float(c_samples.target_cpi_yoy.std()),
            "Min CPI YoY": float(c_samples.target_cpi_yoy.min()),
            "Max CPI YoY": float(c_samples.target_cpi_yoy.max()),
        })
    table_1 = pd.DataFrame(t1_rows)
    table_1.to_csv(manuscript_dir / "table_1_dataset_summary.csv", index=False)
    to_latex_table(table_1, manuscript_dir / "table_1_dataset_summary.tex")

    # ----------------------------------------------------
    # Table 2: Model configurations (selected hyperparameters)
    # ----------------------------------------------------
    print("Generating Table 2: Model Configurations...")
    t2_rows = []
    selected_params = tuning.get("selected_parameters", {})
    for model_name, params in selected_params.items():
        t2_rows.append({
            "Model Family": model_name,
            "Selected Hyperparameters": json.dumps(params)
        })
    table_2 = pd.DataFrame(t2_rows)
    table_2.to_csv(manuscript_dir / "table_2_model_config.csv", index=False)
    to_latex_table(table_2, manuscript_dir / "table_2_model_config.tex")

    # ----------------------------------------------------
    # Table 3: Main forecasting results (RMSE/MAE/sMAPE)
    # ----------------------------------------------------
    print("Generating Table 3: Main forecasting results...")
    # Aggregated metrics over seeds
    metrics_summary = metrics.groupby(["model", "graph_variant", "horizon", "metric"])["value"].mean().reset_index()
    
    # Pivot metrics to wide format
    metrics_wide = metrics_summary.pivot(index=["model", "graph_variant"], columns=["horizon", "metric"], values="value").reset_index()
    
    # Flatten columns hierarchy
    columns_flat = ["Model", "Graph Variant"]
    for horizon in [1, 2, 4]:
        for metric in ["rmse", "mae", "smape"]:
            columns_flat.append(f"H{horizon}_{metric.upper()}")
            
    # Reindex columns to have them in specific order
    cols_to_extract = []
    for horizon in [1, 2, 4]:
        for metric in ["rmse", "mae", "smape"]:
            cols_to_extract.append((horizon, metric))
            
    table_3 = pd.DataFrame(metrics_wide[["model", "graph_variant"]].values, columns=["Model", "Graph Variant"])
    for i, col in enumerate(cols_to_extract):
        table_3[columns_flat[i+2]] = metrics_wide[col].values
        
    table_3.to_csv(manuscript_dir / "table_3_main_results.csv", index=False)
    to_latex_table(table_3, manuscript_dir / "table_3_main_results.tex")

    # ----------------------------------------------------
    # Table 4: Probabilistic results
    # ----------------------------------------------------
    print("Generating Table 4: Probabilistic results...")
    prob_cols_to_extract = []
    prob_columns_flat = ["Model", "Graph Variant"]
    for horizon in [1, 2, 4]:
        for metric in ["crps", "interval_coverage_80", "interval_width_80", "interval_coverage_95", "interval_width_95"]:
            prob_cols_to_extract.append((horizon, metric))
            prob_columns_flat.append(f"H{horizon}_{metric.upper()}")
            
    table_4 = pd.DataFrame(metrics_wide[["model", "graph_variant"]].values, columns=["Model", "Graph Variant"])
    for i, col in enumerate(prob_cols_to_extract):
        table_4[prob_columns_flat[i+2]] = metrics_wide[col].values
        
    table_4.to_csv(manuscript_dir / "table_4_probabilistic_results.csv", index=False)
    to_latex_table(table_4, manuscript_dir / "table_4_probabilistic_results.tex")

    # ----------------------------------------------------
    # Table 5: Statistical significance (DM tests)
    # ----------------------------------------------------
    print("Generating Table 5: Statistical significance...")
    # Seed average statistical testing results
    dm_summary = dm_tests.groupby(["model", "graph_variant", "horizon", "comparator", "loss"])[
        ["dm_stat", "p_value", "p_value_bh", "loss_difference", "ci_low", "ci_high", "origin_count"]
    ].mean().reset_index()
    
    table_5 = dm_summary.rename(columns={
        "model": "Model",
        "graph_variant": "Graph Variant",
        "horizon": "Horizon",
        "comparator": "Comparator",
        "loss": "Loss Type",
        "dm_stat": "DM Stat",
        "p_value": "P-value",
        "p_value_bh": "BH-adjusted P-value",
        "loss_difference": "Mean Loss Diff",
        "ci_low": "CI Low",
        "ci_high": "CI High",
        "origin_count": "Origins"
    })
    table_5.to_csv(manuscript_dir / "table_5_significance.csv", index=False)
    to_latex_table(table_5, manuscript_dir / "table_5_significance.tex")

    # ----------------------------------------------------
    # Table 6: Ablation studies (graph variant comparison)
    # ----------------------------------------------------
    print("Generating Table 6: Ablation studies...")
    # Filter for graph models and aggregate over seeds and models
    graph_metrics = metrics[metrics.model.isin(["gcn", "temporal_graph"])]
    ablation = graph_metrics.groupby(["graph_variant", "horizon", "metric"])["value"].mean().reset_index()
    
    # Pivot to wide format
    ablation_wide = ablation.pivot(index="graph_variant", columns=["horizon", "metric"], values="value").reset_index()
    
    ablation_cols = ["Graph Variant"]
    ablation_extract = []
    for horizon in [1, 2, 4]:
        for metric in ["rmse", "mae", "crps"]:
            ablation_extract.append((horizon, metric))
            ablation_cols.append(f"H{horizon}_{metric.upper()}")
            
    table_6 = pd.DataFrame(ablation_wide["graph_variant"].values, columns=["Graph Variant"])
    for i, col in enumerate(ablation_extract):
        table_6[ablation_cols[i+1]] = ablation_wide[col].values
        
    table_6.to_csv(manuscript_dir / "table_6_ablation_studies.csv", index=False)
    to_latex_table(table_6, manuscript_dir / "table_6_ablation_studies.tex")

    # ====================================================
    # FIGURES GENERATION
    # ====================================================
    print("Generating Figures...")

    # 1. Forecast comparison time series
    # Plot forecast vs actual for FRA and DEU at H1 for a specific seed
    plt.figure(figsize=(12, 6))
    sns.set_theme(style="whitegrid")
    
    seed_42_df = forecasts[(forecasts.seed == "42") & (forecasts.horizon == 1) & (forecasts.country.isin(["FRA", "DEU"]))]
    if not seed_42_df.empty:
        # Get best graph model (e.g. temporal_graph with directed_trade) and arima baseline
        graph_f = seed_42_df[(seed_42_df.model == "temporal_graph") & (seed_42_df.graph_variant == "directed_trade")]
        arima_f = seed_42_df[(seed_42_df.model == "arima")]
        
        # Sort values by target_quarter
        graph_f = graph_f.sort_values("target_quarter")
        arima_f = arima_f.sort_values("target_quarter")
        
        # Plot for FRA
        fra_graph = graph_f[graph_f.country == "FRA"]
        fra_arima = arima_f[arima_f.country == "FRA"]
        
        plt.plot(fra_graph.target_quarter, fra_graph.actual, label="Actual FRA", color="black", linewidth=2)
        plt.plot(fra_graph.target_quarter, fra_graph["mean"], label="Temporal Graph (directed_trade) FRA", color="blue", linestyle="--")
        plt.plot(fra_arima.target_quarter, fra_arima["mean"], label="ARIMA FRA", color="red", linestyle=":")
        
        plt.title("H1 Forecast Comparison (France) - Seed 42", fontsize=14)
        plt.xlabel("Quarter", fontsize=12)
        plt.ylabel("CPI YoY Inflation (%)", fontsize=12)
        plt.xticks(rotation=45)
        plt.legend()
        plt.tight_layout()
        plt.savefig(manuscript_dir / "forecast_comparison.png", dpi=300)
        plt.close()
        print("forecast_comparison.png generated.")

    # 2. Error distribution (box plot)
    plt.figure(figsize=(12, 6))
    forecasts_err = forecasts.copy()
    forecasts_err["error"] = forecasts_err.actual - forecasts_err["mean"]
    
    # Take a sample or seed-averaged error per model
    sns.boxplot(data=forecasts_err, x="model", y="error", palette="Set2")
    plt.title("Forecast Error Distribution by Model", fontsize=14)
    plt.xlabel("Model", fontsize=12)
    plt.ylabel("Error (Actual - Predicted)", fontsize=12)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(manuscript_dir / "error_distribution.png", dpi=300)
    plt.close()
    print("error_distribution.png generated.")

    # 3. Calibration reliability diagram (nominal vs actual coverage)
    plt.figure(figsize=(6, 6))
    # Collect coverages
    cov_80 = metrics[metrics.metric == "interval_coverage_80"]["value"].mean()
    cov_95 = metrics[metrics.metric == "interval_coverage_95"]["value"].mean()
    
    plt.plot([0, 1], [0, 1], color="gray", linestyle="--", label="Perfect Calibration")
    plt.scatter([0.80, 0.95], [cov_80, cov_95], color="red", s=100, zorder=5)
    plt.plot([0.80, 0.95], [cov_80, cov_95], color="red", linestyle="-", label="Empirical Coverage")
    
    plt.xlim(0.7, 1.0)
    plt.ylim(0.7, 1.0)
    plt.title("Calibration Reliability Diagram", fontsize=14)
    plt.xlabel("Nominal Coverage", fontsize=12)
    plt.ylabel("Empirical Coverage", fontsize=12)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(manuscript_dir / "calibration_reliability.png", dpi=300)
    plt.close()
    print("calibration_reliability.png generated.")

    # 4. Prediction interval coverage plots
    plt.figure(figsize=(10, 6))
    cov_df = metrics[metrics.metric.isin(["interval_coverage_80", "interval_coverage_95"])]
    sns.barplot(data=cov_df, x="model", y="value", hue="metric", palette="muted")
    plt.axhline(0.80, color="blue", linestyle="--", alpha=0.5, label="80% Nominal")
    plt.axhline(0.95, color="red", linestyle="--", alpha=0.5, label="95% Nominal")
    plt.title("Empirical Prediction Interval Coverage by Model", fontsize=14)
    plt.xlabel("Model", fontsize=12)
    plt.ylabel("Coverage Fraction", fontsize=12)
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig(manuscript_dir / "prediction_interval_coverage.png", dpi=300)
    plt.close()
    print("prediction_interval_coverage.png generated.")

    # 5. Graph variant performance heatmap (MAE)
    plt.figure(figsize=(10, 6))
    graph_mae = metrics[(metrics.model.isin(["gcn", "temporal_graph"])) & (metrics.metric == "mae")]
    if not graph_mae.empty:
        pivot_mae = graph_mae.pivot_table(index="graph_variant", columns="horizon", values="value", aggfunc="mean")
        sns.heatmap(pivot_mae, annot=True, fmt=".4f", cmap="YlOrRd")
        plt.title("Mean Absolute Error (MAE) by Graph Variant and Horizon", fontsize=14)
        plt.xlabel("Horizon (Quarters)", fontsize=12)
        plt.ylabel("Graph Variant", fontsize=12)
        plt.tight_layout()
        plt.savefig(manuscript_dir / "graph_variant_heatmap.png", dpi=300)
        plt.close()
        print("graph_variant_heatmap.png generated.")

    # 6. Performance by horizon bar charts (MAE)
    plt.figure(figsize=(10, 6))
    mae_df = metrics[metrics.metric == "mae"]
    sns.barplot(data=mae_df, x="horizon", y="value", hue="model", palette="colorblind")
    plt.title("Mean Absolute Error (MAE) by Model and Horizon", fontsize=14)
    plt.xlabel("Horizon (Quarters)", fontsize=12)
    plt.ylabel("MAE", fontsize=12)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(manuscript_dir / "performance_by_horizon.png", dpi=300)
    plt.close()
    print("performance_by_horizon.png generated.")

    print("All manuscript tables and figures successfully generated.")
    return 0

if __name__ == "__main__":
    sys.exit(main())
