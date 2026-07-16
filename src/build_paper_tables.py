"""Emit LaTeX table fragments from frozen evaluation JSON."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVAL = ROOT / "results" / "evaluation" / "evaluation_report.json"
GRID = ROOT / "results" / "evaluation" / "grid_search.json"
OUT = ROOT / "paper" / "generated"


def fmt(x: float, digits: int = 2) -> str:
    return f"{x:.{digits}f}"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    report = json.loads(EVAL.read_text())
    grid = json.loads(GRID.read_text())

    m = report["metrics"]
    dm = report["diebold_mariano_test"]
    dm_cr = report["diebold_mariano_cluster_robust"]
    tost = report["tost_equivalence_test"]
    apples = report["apples_to_apples_comparison"]["all_models_excluding_argentina"]

    table1 = r"""
\begin{table}[ht]
  \centering
  \caption{Out-of-sample forecast accuracy on the test period (2021Q1--2024Q4).}
  \label{tab:benchmark}
  \begin{tabular}{lrrr}
    \toprule
    Model & RMSE & MAE & $N$ \\
    \midrule
    ST-GNN (rolling refit) & %s & %s & %d \\
    ARIMA(0,1,1)           & %s & %s & %d \\
    VAR(4)                 & %s & %s & %d \\
    \bottomrule
  \end{tabular}
  \vspace{0.5em}
  \begin{minipage}{\linewidth}
    \footnotesize
    \textit{Notes:} ST-GNN uses expanding-window refit with Huber loss and persistence skip connection.
    VAR excludes Argentina; $N=292$ for VAR and Diebold--Mariano comparisons.
    ST-GNN RMSE excluding Argentina: %s.
  \end{minipage}
\end{table}
""" % (
        fmt(m["stgnn_rolling_test"]["rmse"]),
        fmt(m["stgnn_rolling_test"]["mae"]),
        int(m["stgnn_rolling_test"]["n"]),
        fmt(m["arima_test"]["rmse"]),
        fmt(m["arima_test"]["mae"]),
        int(m["arima_test"]["n"]),
        fmt(m["var_test"]["rmse"]),
        fmt(m["var_test"]["mae"]),
        int(m["var_test"]["n"]),
        fmt(m["stgnn_rolling_test_excl_ARG"]["rmse"]),
    )

    table2 = r"""
\begin{table}[ht]
  \centering
  \caption{Diebold--Mariano tests: ST-GNN vs.\ baselines (squared forecast errors, $h=1$).}
  \label{tab:dm}
  \begin{tabular}{lrrrr}
    \toprule
    Comparison & DM statistic & $p$-value & $N$ & Clusters \\
    \midrule
    ST-GNN vs.\ ARIMA (standard) & %s & %s & %d & -- \\
    ST-GNN vs.\ ARIMA (cluster-robust) & %s & %s & %d & %d \\
    ST-GNN vs.\ VAR (standard) & %s & %s & %d & -- \\
    ST-GNN vs.\ VAR (cluster-robust) & %s & %s & %d & %d \\
    \bottomrule
  \end{tabular}
  \vspace{0.5em}
  \begin{minipage}{\linewidth}
    \footnotesize
    \textit{Notes:} Cluster-robust tests cluster by quarter to account for cross-sectional correlation.
    Paired on test-period observations with non-missing ARIMA and VAR forecasts ($N=%d$).
    Negative statistic implies lower average squared error for ST-GNN.
  \end{minipage}
\end{table}
""" % (
        fmt(dm["stgnn_vs_arima_squared"]["statistic"], 3),
        fmt(dm["stgnn_vs_arima_squared"]["p_value"], 3),
        int(dm["stgnn_vs_arima_squared"]["n"]),
        fmt(dm_cr["stgnn_vs_arima_squared"]["statistic"], 3),
        fmt(dm_cr["stgnn_vs_arima_squared"]["p_value"], 3),
        int(dm_cr["stgnn_vs_arima_squared"]["n"]),
        int(dm_cr["stgnn_vs_arima_squared"]["n_clusters"]),
        fmt(dm["stgnn_vs_var_squared"]["statistic"], 3),
        fmt(dm["stgnn_vs_var_squared"]["p_value"], 3),
        int(dm["stgnn_vs_var_squared"]["n"]),
        fmt(dm_cr["stgnn_vs_var_squared"]["statistic"], 3),
        fmt(dm_cr["stgnn_vs_var_squared"]["p_value"], 3),
        int(dm_cr["stgnn_vs_var_squared"]["n"]),
        int(dm_cr["stgnn_vs_var_squared"]["n_clusters"]),
        int(report["paired_test_rows"]),
    )

    rows = []
    for row in sorted(grid, key=lambda r: r["val_rmse"]):
        rows.append(
            f"    {row['lr']:.4f} & {row['dropout']:.1f} & {row['gcn_layers']} & {row['val_rmse']:.3f} \\\\"
        )
    table_a1 = r"""
\begin{longtable}{rrrr}
  \caption{Hyperparameter grid search on validation RMSE (2019Q1--2020Q4).}
  \label{tab:grid} \\
  \toprule
  Learning rate & Dropout & GCN layers & Val RMSE \\
  \midrule
  \endfirsthead
  \toprule
  Learning rate & Dropout & GCN layers & Val RMSE \\
  \midrule
  \endhead
""" + "\n".join(rows) + r"""
  \bottomrule
\end{longtable}
"""

    table_apples = r"""
\begin{table}[ht]
  \centering
  \caption{Apples-to-apples comparison: all models evaluated on same country set (excluding Argentina).}
  \label{tab:apples}
  \begin{tabular}{lrrr}
    \toprule
    Model & RMSE & MAE & $N$ \\
    \midrule
    ST-GNN (rolling refit) & %s & %s & %d \\
    ARIMA(0,1,1)           & %s & %s & %d \\
    VAR(4)                 & %s & %s & %d \\
    \bottomrule
  \end{tabular}
  \vspace{0.5em}
  \begin{minipage}{\linewidth}
    \footnotesize
    \textit{Notes:} All models evaluated on 22-country panel (Argentina excluded).
    Diebold--Mariano tests: ST-GNN vs.\ ARIMA stat=%s p=%s; ST-GNN vs.\ VAR stat=%s p=%s.
  \end{minipage}
\end{table}
""" % (
        fmt(apples["stgnn_metrics"]["rmse"]),
        fmt(apples["stgnn_metrics"]["mae"]),
        int(apples["stgnn_metrics"]["n"]),
        fmt(apples["arima_metrics"]["rmse"]),
        fmt(apples["arima_metrics"]["mae"]),
        int(apples["arima_metrics"]["n"]),
        fmt(apples["var_metrics"]["rmse"]),
        fmt(apples["var_metrics"]["mae"]),
        int(apples["var_metrics"]["n"]),
        fmt(apples["dm_stgnn_vs_arima"]["statistic"], 3),
        fmt(apples["dm_stgnn_vs_arima"]["p_value"], 3),
        fmt(apples["dm_stgnn_vs_var"]["statistic"], 3),
        fmt(apples["dm_stgnn_vs_var"]["p_value"], 3),
    )

    table_tost = r"""
\begin{table}[ht]
  \centering
  \caption{TOST equivalence test: ST-GNN vs.\ ARIMA (RMSE difference).}
  \label{tab:tost}
  \begin{tabular}{lrr}
    \toprule
    Parameter & Value \\
    \midrule
    Equivalence margin (\%% of ARIMA RMSE) & %.1f\\
    Margin (absolute RMSE) & %.3f\\
    RMSE difference (ST-GNN - ARIMA) & %.4f\\
    TOST $p$-value & %.3f\\
    Equivalence supported ($p < 0.05$) & %s\\
    $N$ & %d\\
    \bottomrule
  \end{tabular}
  \vspace{0.5em}
  \begin{minipage}{\linewidth}
    \footnotesize
    \textit{Notes:} TOST tests whether RMSE difference is within pre-specified margin.
    H0: $|\text{RMSE diff}| \geq \text{margin}$ (not equivalent); H1: $|\text{RMSE diff}| < \text{margin}$ (equivalent).
    Bootstrap SE estimation with 1000 resamples.
  \end{minipage}
\end{table}
""" % (
        tost["margin_pct_of_arima_rmse"],
        tost["margin"],
        tost["stgnn_vs_arima_squared"]["mean_diff"],
        tost["stgnn_vs_arima_squared"]["p_value"],
        "YES" if tost["stgnn_vs_arima_squared"]["equivalence_rejected"] == "True" else "NO",
        tost["stgnn_vs_arima_squared"]["n"],
    )

    (OUT / "table1.tex").write_text(table1.strip())
    (OUT / "table2.tex").write_text(table2.strip())
    (OUT / "table_apples.tex").write_text(table_apples.strip())
    (OUT / "table_tost.tex").write_text(table_tost.strip())
    (OUT / "table_a1.tex").write_text(table_a1.strip())
    print(f"Wrote LaTeX tables to {OUT}/")


if __name__ == "__main__":
    main()
