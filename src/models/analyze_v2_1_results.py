"""Statistical analysis, rankings, and significance summaries for completed v2.1 benchmark."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from src.models.storage_v2_1 import RESULTS  # noqa: E402


def main() -> int:
    print("Starting post-execution statistical analysis...")
    analysis_dir = RESULTS / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = RESULTS / "metrics.parquet"
    dm_path = RESULTS / "dm_tests.parquet"

    if not metrics_path.exists() or not dm_path.exists():
        print("Error: metrics.parquet or dm_tests.parquet not found. Execute the benchmark first.")
        return 1

    metrics = pd.read_parquet(metrics_path)
    dm_tests = pd.read_parquet(dm_path)

    # 1. Model rankings by horizon and metric (aggregating over seeds)
    # Seed aggregation: compute mean and standard deviation of values
    # For deterministic models, seed is "deterministic", std is 0.
    metrics_summary = metrics.groupby(["model", "graph_variant", "horizon", "metric"])["value"].agg(["mean", "std", "count"]).reset_index()

    # Save aggregated metrics to CSV
    metrics_summary.to_csv(analysis_dir / "aggregated_metrics.csv", index=False)
    print(f"Aggregated metrics written to {analysis_dir / 'aggregated_metrics.csv'}")

    # Create rankings for each horizon and metric
    rankings_list = []
    for (horizon, metric), group in metrics_summary.groupby(["horizon", "metric"]):
        sorted_group = group.sort_values(by="mean", ascending=(metric not in ["interval_coverage_80", "interval_coverage_95"]))
        sorted_group = sorted_group.copy()
        sorted_group["rank"] = np.arange(1, len(sorted_group) + 1)
        rankings_list.append(sorted_group)

    rankings_df = pd.concat(rankings_list, ignore_index=True)
    rankings_df.to_csv(analysis_dir / "model_rankings.csv", index=False)
    print(f"Model rankings written to {analysis_dir / 'model_rankings.csv'}")

    # 2. Probabilistic Evaluation Summary
    prob_metrics = ["crps", "interval_coverage_80", "interval_width_80", "interval_coverage_95", "interval_width_95"]
    prob_eval = metrics_summary[metrics_summary.metric.isin(prob_metrics)]
    prob_eval.to_csv(analysis_dir / "probabilistic_evaluation.csv", index=False)
    print(f"Probabilistic evaluation summary written to {analysis_dir / 'probabilistic_evaluation.csv'}")

    # 3. Statistical Significance Summary (from dm_tests.parquet)
    # Highlight significant improvements: loss_difference < 0, ci_low < 0, ci_high < 0 (i.e. excludes 0), p_value_bh < 0.05
    # Note: loss diff = graph - comparator. If loss diff < 0, graph has LOWER loss (better).
    # Moving-block bootstrap CI excludes 0 if both bounds are negative (since loss diff is negative).
    dm_tests = dm_tests.copy()
    dm_tests["is_significant"] = (dm_tests.loss_difference < 0) & \
                                 (dm_tests.ci_high < 0) & \
                                 (dm_tests.p_value_bh < 0.05)

    # Aggregate significance across seeds (percentage of seeds where test is significant)
    sig_summary = dm_tests.groupby(["model", "graph_variant", "horizon", "comparator", "loss"])["is_significant"].agg(["mean", "sum", "count"]).reset_index()
    sig_summary.rename(columns={"mean": "fraction_seeds_significant", "sum": "count_seeds_significant", "count": "total_seeds"}, inplace=True)
    sig_summary.to_csv(analysis_dir / "statistical_significance.csv", index=False)
    print(f"Statistical significance summary written to {analysis_dir / 'statistical_significance.csv'}")

    # Print out summary statistics
    total_tests = len(sig_summary)
    fully_significant = len(sig_summary[sig_summary.fraction_seeds_significant == 1.0])
    partially_significant = len(sig_summary[(sig_summary.fraction_seeds_significant > 0) & (sig_summary.fraction_seeds_significant < 1.0)])

    print(f"Analysis completed: {total_tests} model-variant-horizon-comparator DM tests analyzed.")
    print(f"  - {fully_significant} tests are significant across ALL 20 seeds.")
    print(f"  - {partially_significant} tests are significant in SOME seeds.")

    return 0

if __name__ == "__main__":
    sys.exit(main())
