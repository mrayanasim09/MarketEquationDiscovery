"""Rebuild all broken tables in paper/manuscript.md from source parquet data.

Tables 1, 3, 4, 5, 6 have lost their data due to minipage grid-table conversion
artifacts. This script reads the authoritative parquet/CSV sources and writes
properly-formatted pandoc pipe tables back into the markdown.

Also fixes the truncated sentence in §5.4.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
MANUSCRIPT = ROOT / "paper" / "manuscript.md"
RESULTS = ROOT / "experiments" / "results" / "v2_1"
MANUSCRIPT_DIR = RESULTS / "manuscript"


def fmt(v, decimals=3):
    if pd.isna(v):
        return "---"
    return f"{v:.{decimals}f}"


def pipe_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join([":---"] + ["---:"] * (len(headers) - 1)) + "|")
    for row in rows:
        lines.append("| " + " | ".join(str(c) for c in row) + " |")
    return "\n".join(lines)


def build_table1() -> str:
    t1 = pd.read_csv(MANUSCRIPT_DIR / "table_1_dataset_summary.csv")
    headers = ["Country", "Total Obs", "Train", "Val", "Test",
               "Mean CPI YoY", "Std CPI YoY", "Min CPI YoY", "Max CPI YoY"]
    rows = []
    for _, r in t1.iterrows():
        rows.append([
            r["Country"],
            int(r["Total Obs"]),
            int(r["Train Obs"]),
            int(r["Val Obs"]),
            int(r["Test Obs"]),
            fmt(r["Mean CPI YoY"], 2),
            fmt(r["Std CPI YoY"], 2),
            fmt(r["Min CPI YoY"], 2),
            fmt(r["Max CPI YoY"], 2),
        ])
    rows.append([
        "**Panel mean**",
        int(t1["Total Obs"].mean()),
        int(t1["Train Obs"].mean()),
        int(t1["Val Obs"].mean()),
        int(t1["Test Obs"].mean()),
        fmt(t1["Mean CPI YoY"].mean(), 2),
        fmt(t1["Std CPI YoY"].mean(), 2),
        "---",
        "---",
    ])
    return pipe_table(headers, rows)


def build_table3() -> str:
    t3 = pd.read_csv(MANUSCRIPT_DIR / "table_3_main_results.csv")

    MODEL_LABELS = {
        "gcn": "GCN", "temporal_graph": "Temporal Graph", "arima": "ARIMA",
        "bvar": "BVAR", "var": "VAR", "ets": "ETS", "lstm": "LSTM",
        "tcn": "TCN", "mlp": "MLP", "ridge": "Ridge",
        "gradient_boosting": "Gradient Boosting", "dynamic_factor": "Dynamic Factor",
        "persistence": "Persistence",
    }
    display_order = [
        ("gcn", "identity_no_trade"),
        ("arima", "none"),
        ("tcn", "none"),
        ("mlp", "none"),
        ("persistence", "none"),
        ("ets", "none"),
        ("gradient_boosting", "none"),
        ("dynamic_factor", "none"),
        ("temporal_graph", "identity_no_trade"),
        ("lstm", "none"),
        ("ridge", "none"),
        ("bvar", "none"),
        ("var", "none"),
    ]

    bests_mae = {}
    bests_rmse = {}
    for h in [1, 2, 4]:
        best_mae = float("inf")
        best_rmse = float("inf")
        best_mae_key = None
        best_rmse_key = None
        for model, gv in display_order:
            row = t3[(t3["Model"] == model) & (t3["Graph Variant"] == gv)]
            if row.empty:
                continue
            mae = float(row[f"H{h}_MAE"].values[0])
            rmse = float(row[f"H{h}_RMSE"].values[0])
            if mae < best_mae:
                best_mae = mae
                best_mae_key = (model, gv)
            if rmse < best_rmse:
                best_rmse = rmse
                best_rmse_key = (model, gv)
        bests_mae[h] = best_mae_key
        bests_rmse[h] = best_rmse_key

    headers = ["Model", "Graph Variant",
               "H=1 MAE", "H=2 MAE", "H=4 MAE",
               "H=1 RMSE", "H=2 RMSE", "H=4 RMSE"]
    rows_out = []

    for model, gv in display_order:
        row = t3[(t3["Model"] == model) & (t3["Graph Variant"] == gv)]
        if row.empty:
            continue
        label = MODEL_LABELS.get(model, model)
        gv_label = f"`{gv}`" if gv != "none" else "---"

        def cell(val, bests_h, h, model=model, gv=gv):
            s = fmt(val, 3)
            if bests_h.get(h) == (model, gv):
                return f"**{s}**"
            return s

        mae1 = float(row["H1_MAE"].values[0])
        mae2 = float(row["H2_MAE"].values[0])
        mae4 = float(row["H4_MAE"].values[0])
        rmse1 = float(row["H1_RMSE"].values[0])
        rmse2 = float(row["H2_RMSE"].values[0])
        rmse4 = float(row["H4_RMSE"].values[0])

        rows_out.append([
            label, gv_label,
            cell(mae1, bests_mae, 1), cell(mae2, bests_mae, 2), cell(mae4, bests_mae, 4),
            cell(rmse1, bests_rmse, 1), cell(rmse2, bests_rmse, 2), cell(rmse4, bests_rmse, 4),
        ])

    return pipe_table(headers, rows_out)


def build_table4_ablation() -> str:
    t6 = pd.read_csv(MANUSCRIPT_DIR / "table_6_ablation_studies.csv")

    GRAPH_ORDER = [
        "identity_no_trade", "directed_trade", "log_trade", "reversed",
        "undirected", "import_dependence", "degree_preserving_random", "top_k_incoming",
    ]

    bests_mae = {h: None for h in [1, 2, 4]}
    bests_crps = {h: None for h in [1, 2, 4]}
    for h in [1, 2, 4]:
        best_mae = float("inf")
        best_crps = float("inf")
        for gv in GRAPH_ORDER:
            row = t6[t6["Graph Variant"] == gv]
            if row.empty:
                continue
            mae = float(row[f"H{h}_MAE"].values[0])
            crps = float(row[f"H{h}_CRPS"].values[0])
            if mae < best_mae:
                best_mae = mae
                bests_mae[h] = gv
            if crps < best_crps:
                best_crps = crps
                bests_crps[h] = gv

    headers = ["Graph Variant",
               "H=1 MAE", "H=2 MAE", "H=4 MAE",
               "H=1 CRPS", "H=2 CRPS", "H=4 CRPS"]
    rows_out = []

    for gv in GRAPH_ORDER:
        row = t6[t6["Graph Variant"] == gv]
        if row.empty:
            continue

        def cell_m(val, h, gv=gv):
            s = fmt(val, 3)
            return f"**{s}**" if bests_mae[h] == gv else s

        def cell_c(val, h, gv=gv):
            s = fmt(val, 3)
            return f"**{s}**" if bests_crps[h] == gv else s

        rows_out.append([
            f"`{gv}`",
            cell_m(float(row["H1_MAE"].values[0]), 1),
            cell_m(float(row["H2_MAE"].values[0]), 2),
            cell_m(float(row["H4_MAE"].values[0]), 4),
            cell_c(float(row["H1_CRPS"].values[0]), 1),
            cell_c(float(row["H2_CRPS"].values[0]), 2),
            cell_c(float(row["H4_CRPS"].values[0]), 4),
        ])

    return pipe_table(headers, rows_out)


def build_table5_crps() -> str:
    t4 = pd.read_csv(MANUSCRIPT_DIR / "table_4_probabilistic_results.csv")

    MODEL_LABELS = {
        "gcn": "GCN", "temporal_graph": "Temporal Graph", "arima": "ARIMA",
        "bvar": "BVAR", "var": "VAR", "ets": "ETS", "lstm": "LSTM",
        "tcn": "TCN", "mlp": "MLP", "ridge": "Ridge",
        "gradient_boosting": "Gradient Boosting", "dynamic_factor": "Dynamic Factor",
        "persistence": "Persistence",
    }
    display_order = [
        ("gcn", "identity_no_trade"), ("arima", "none"), ("tcn", "none"),
        ("mlp", "none"), ("persistence", "none"), ("ets", "none"),
        ("gradient_boosting", "none"), ("dynamic_factor", "none"),
        ("temporal_graph", "identity_no_trade"), ("lstm", "none"),
        ("ridge", "none"), ("bvar", "none"),
    ]

    crps_bests = {}
    for h in [1, 2, 4]:
        best = float("inf")
        best_key = None
        for model, gv in display_order:
            if model == "bvar":
                continue
            row = t4[(t4["Model"] == model) & (t4["Graph Variant"] == gv)]
            if row.empty:
                continue
            v = float(row[f"H{h}_CRPS"].values[0])
            if v < best:
                best = v
                best_key = (model, gv)
        crps_bests[h] = best_key

    headers = ["Model", "Graph Variant",
               "H=1 CRPS", "H=2 CRPS", "H=4 CRPS",
               "H=2 Cov-80", "H=4 Cov-80"]
    rows_out = []

    for model, gv in display_order:
        row = t4[(t4["Model"] == model) & (t4["Graph Variant"] == gv)]
        if row.empty:
            continue
        label = MODEL_LABELS.get(model, model)
        gv_label = f"`{gv}`" if gv != "none" else "---"

        if model == "bvar":
            rows_out.append([label, gv_label, "N/A", "N/A", "N/A", "N/A", "N/A"])
            continue

        def cell_c(val, h, model=model, gv=gv):
            s = fmt(val, 3)
            if crps_bests.get(h) == (model, gv):
                return f"**{s}**"
            return s

        rows_out.append([
            label, gv_label,
            cell_c(float(row["H1_CRPS"].values[0]), 1),
            cell_c(float(row["H2_CRPS"].values[0]), 2),
            cell_c(float(row["H4_CRPS"].values[0]), 4),
            fmt(float(row["H2_INTERVAL_COVERAGE_80"].values[0]), 3),
            fmt(float(row["H4_INTERVAL_COVERAGE_80"].values[0]), 3),
        ])

    return pipe_table(headers, rows_out)


def build_table6_dm() -> str:
    dm = pd.read_parquet(RESULTS / "dm_tests.parquet")
    dm["sig_and_better"] = (dm["p_value_bh"] < 0.05) & (dm["loss_difference"] > 0)
    dm_agg = dm.groupby(["model", "graph_variant", "horizon", "comparator", "loss"]).agg(
        prop_sig=("sig_and_better", "mean"),
        n_seeds=("seed", "nunique"),
    ).reset_index()
    dm_key = dm_agg[
        (dm_agg["graph_variant"] == "identity_no_trade") &
        (dm_agg["loss"] == "absolute_error") &
        (dm_agg["comparator"].isin(["arima", "bvar", "ets", "tcn", "ridge", "dynamic_factor"])) &
        (dm_agg["model"].isin(["gcn", "temporal_graph"]))
    ].copy()

    COMP_LABELS = {
        "arima": "ARIMA", "bvar": "BVAR", "ets": "ETS",
        "tcn": "TCN", "ridge": "Ridge", "dynamic_factor": "Dynamic Factor"
    }
    MODEL_LABELS = {"gcn": "GCN", "temporal_graph": "Temporal Graph"}
    COMP_ORDER = ["arima", "bvar", "ets", "tcn", "ridge", "dynamic_factor"]

    headers = ["Model", "Comparator",
               "h=1 Prop. Seeds Sig.", "h=2 Prop. Seeds Sig.", "h=4 Prop. Seeds Sig."]
    rows_out = []

    for model in ["gcn", "temporal_graph"]:
        for comp in COMP_ORDER:
            row_vals = [MODEL_LABELS[model], COMP_LABELS[comp]]
            for h in [1, 2, 4]:
                sub = dm_key[
                    (dm_key["model"] == model) &
                    (dm_key["comparator"] == comp) &
                    (dm_key["horizon"] == h)
                ]
                if sub.empty:
                    row_vals.append("---")
                else:
                    p = float(sub["prop_sig"].values[0])
                    n = int(sub["n_seeds"].values[0])
                    n_sig = int(round(p * n))
                    pct_str = f"{p*100:.0f}% ({n_sig}/{n})"
                    if p > 0.50:
                        pct_str = f"**{pct_str}**"
                    row_vals.append(pct_str)
            rows_out.append(row_vals)

    return pipe_table(headers, rows_out)


def main() -> int:
    print("Reading manuscript.md ...")
    text = MANUSCRIPT.read_text()
    original_len = len(text)

    # ---- Table 1 ----
    print("Rebuilding Table 1: Dataset Summary ...")
    new_t1 = build_table1()
    text = re.sub(
        r'(\*\*Table 1:[^\n]*\n\n)'
        r'(\|[^\n]+\n\|[-:| ]+\n(?:\|[^\n]+\n)+)'
        r'(?=\n\*Notes:)',
        lambda m: m.group(1) + new_t1 + "\n",
        text, flags=re.DOTALL
    )

    # ---- Table 3 ----
    print("Rebuilding Table 3: Main Results ...")
    new_t3 = build_table3()
    text = re.sub(
        r'(\*\*Table 3:[^\n]*\n(?:[^\n]*\n)??\n)'
        r'(\+[-=+| :][^\n]*(?:\n.*?)*?\n\+[-+]*\+\n?)',
        lambda m: m.group(1) + new_t3 + "\n",
        text, flags=re.DOTALL
    )

    # ---- Table 4 (ablation in paper) ----
    print("Rebuilding Table 4: Ablation ...")
    new_t4 = build_table4_ablation()
    text = re.sub(
        r'(\*\*Table 4:[^\n]*\n(?:[^\n]*\n)??\n)'
        r'(\+[-=+| :][^\n]*(?:\n.*?)*?\n\+[-+]*\+\n?)',
        lambda m: m.group(1) + new_t4 + "\n",
        text, flags=re.DOTALL
    )

    # ---- Table 5 (CRPS) ----
    print("Rebuilding Table 5: CRPS ...")
    new_t5 = build_table5_crps()
    text = re.sub(
        r'(\*\*Table 5:[^\n]*\n(?:[^\n]*\n)??\n)'
        r'(\+[-=+| :][^\n]*(?:\n.*?)*?\n\+[-+]*\+\n?)',
        lambda m: m.group(1) + new_t5 + "\n",
        text, flags=re.DOTALL
    )

    # ---- Table 6 (DM tests) ----
    print("Rebuilding Table 6: DM Tests ...")
    new_t6 = build_table6_dm()
    text = re.sub(
        r'(\*\*Table 6:[^\n]*\n(?:[^\n]*\n)??\n)'
        r'(\+[-=+| :][^\n]*(?:\n.*?)*?\n\+[-+]*\+\n?)',
        lambda m: m.group(1) + new_t6 + "\n",
        text, flags=re.DOTALL
    )

    # ---- Fix truncated §5.4 sentence (if still present) ----
    trunc = "We adopt two interpretive thresholds: *majority* (\\> 50"
    fixed = ('We adopt two interpretive thresholds: a *majority* threshold (\\> 50% of\n'
             'seeds significant) as the primary criterion for "consistent" evidence,\n'
             'and a *supermajority* threshold (\\> 75% of seeds) as a secondary\n'
             'criterion for "strong" evidence. We deliberately do not require\n'
             'unanimity across all 20 seeds, as any stochastic difference in\n'
             'initialisation can preclude unanimity even when the population-level\n'
             'effect is real and the statistical power per seed is high. Where\n'
             'proportion-of-seeds results are informative, they are reported alongside\n'
             'the point-forecast rankings.')
    if trunc in text:
        print("Fixing truncated sentence in §5.4 ...")
        # Find and replace the truncated block (it likely cuts off with a table)
        text = re.sub(
            r'We adopt two interpretive thresholds: \*majority\* \(\\> 50.*?(?=\n\n::: center|\n\n#|\n\n\*\*Table)',
            fixed,
            text, flags=re.DOTALL
        )

    MANUSCRIPT.write_text(text)
    new_len = len(text)
    print(f"Done. manuscript.md: {original_len} -> {new_len} bytes (+{new_len - original_len})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
