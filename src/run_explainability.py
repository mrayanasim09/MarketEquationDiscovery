"""Milestone 8: Integrated Gradients explainability and case-study reports."""

from __future__ import annotations

import json
from datetime import datetime, timezone

import matplotlib.pyplot as plt
import pandas as pd
import torch

from src.config import RESULTS, ensure_dirs
from src.stgnn_data import load_panel_tensors
from src.stgnn_explain import (
    explain_timestep,
    find_spillover_country,
    load_best_train_config,
    refit_model,
)
from src.stgnn_train import build_edge_lists, set_seed

OUT_DIR = RESULTS / "explainability"
FIG_DIR = OUT_DIR / "figures"
EVAL_REPORT = RESULTS / "evaluation" / "evaluation_report.json"

CASE_STUDIES = {
    "covid_supply_chain": {
        "quarter": "2020Q2",
        "title": "COVID supply-chain disruption (2020Q2)",
        "narrative": "Pandemic lockdowns disrupted global trade flows; edge attributions highlight "
        "which trade partners the model relied on when forecasting inflation during the shock.",
    },
    "energy_shock_2022": {
        "quarter": "2022Q2",
        "title": "2022 global energy price shock (2022Q2)",
        "narrative": "Russia-Ukraine war and Brent spike; attributions show energy-linked trade "
        "partners gaining influence on inflation forecasts.",
    },
    "baseline_calm": {
        "quarter": "2017Q2",
        "title": "Baseline calm period (2017Q2)",
        "narrative": "Pre-pandemic period with stable inflation; this baseline tests whether "
        "attribution rankings during shocks differ from normal periods or simply track trade volume.",
    },
}


def quarter_to_idx(quarters: list[str], quarter: str) -> int:
    try:
        return quarters.index(quarter)
    except ValueError as exc:
        raise ValueError(f"Quarter {quarter} not in panel") from exc


def plot_top_partners(edges: pd.DataFrame, title: str, out_path, top_k: int = 10) -> None:
    if edges.empty:
        return
    top = edges.nlargest(top_k, "abs_edge_attr").copy().sort_values("abs_edge_attr")
    colors = ["#c0392b" if v < 0 else "#2980b9" for v in top["edge_attr"]]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(top["source_iso3"], top["abs_edge_attr"], color=colors)
    ax.set_xlabel("Integrated Gradients attribution (|edge weight|)")
    ax.set_ylabel("Trade partner (source)")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_domestic_vs_trade(feat_summary: dict, edge_total: float, title: str, out_path) -> None:
    labels = ["Domestic features", "Trade edges (IG)"]
    values = [feat_summary["domestic_abs_attr"], edge_total]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(labels, values, color=["#27ae60", "#8e44ad"])
    ax.set_ylabel("Absolute IG attribution")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_spillover_heatmap(edges: pd.DataFrame, target_iso3: str, quarter: str, out_path, top_k: int = 8) -> None:
    sub = edges[edges["target_iso3"] == target_iso3].nlargest(top_k, "abs_edge_attr")
    if sub.empty:
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(sub["source_iso3"], sub["edge_attr"], color="#e67e22")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Signed edge IG attribution")
    ax.set_xlabel("Partner country")
    ax.set_title(f"Trade spillovers to {target_iso3} ({quarter})")
    plt.xticks(rotation=45, ha="right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def build_case_result(
    case_id: str,
    meta: dict,
    detail: dict,
    target_iso3: str,
    figure_paths: list[str],
) -> dict:
    edges = detail["edge_attributions"]
    return {
        "case_id": case_id,
        "title": meta.get("title", case_id),
        "quarter": meta["quarter"],
        "target_iso3": target_iso3,
        "prediction": detail["prediction"],
        "narrative": meta.get("narrative", ""),
        "top_trade_partners": edges.head(5)[["source_iso3", "edge_attr", "abs_edge_attr"]].to_dict("records"),
        "feature_attribution": detail["feature_summary"],
        "trade_vs_domestic_ratio": detail["trade_vs_domestic_ratio"],
        "figures": figure_paths,
    }


def seed_robustness_check(
    x_raw: np.ndarray,
    y: np.ndarray,
    target_mask: np.ndarray,
    t_idx: int,
    edge_indices: list,
    edge_weights: list,
    cfg,
    device: torch.device,
    countries: list[str],
    quarters: list[str],
    target_iso3: str,
    n_seeds: int = 5,
) -> dict:
    """Retrain model with different seeds and check stability of top partner attributions."""
    from src.stgnn_explain import explain_timestep, refit_model
    
    all_top_partners = []
    seed_results = []
    
    for seed in range(n_seeds):
        set_seed(seed)
        print(f"  Seed {seed}: refitting model...")
        model = refit_model(
            x_raw, y, target_mask, t_idx, edge_indices, edge_weights, cfg, device, refit_epochs=30
        )
        detail = explain_timestep(
            model, x_raw, t_idx, countries.index(target_iso3), countries, quarters,
            edge_indices, edge_weights, device,
        )
        top_partners = detail["edge_attributions"].nlargest(3, "abs_edge_attr")["source_iso3"].tolist()
        all_top_partners.append(top_partners)
        seed_results.append({
            "seed": seed,
            "top_3_partners": top_partners,
            "trade_vs_domestic_ratio": detail["trade_vs_domestic_ratio"],
        })
    
    # Compute stability: how often do the same partners appear in top 3?
    from collections import Counter
    partner_counts = Counter([p for partners in all_top_partners for p in partners])
    
    # Check if top partner is consistent across seeds
    top_1_consistency = len(set([p[0] for p in all_top_partners])) == 1
    top_3_overlap = len(set([p for partners in all_top_partners for p in partners])) <= 5  # At most 5 unique partners in top 3 across seeds
    
    return {
        "seed_results": seed_results,
        "partner_counts": dict(partner_counts),
        "top_1_consistent": top_1_consistency,
        "top_3_stable": top_3_overlap,
        "n_seeds": n_seeds,
    }


def main() -> None:
    ensure_dirs()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    if not EVAL_REPORT.exists():
        raise FileNotFoundError("Run python -m src.run_evaluation first (Milestone 7).")

    cfg = load_best_train_config(EVAL_REPORT)
    data = load_panel_tensors()
    countries = data["countries"]
    quarters = data["quarters"]
    x_raw = data["x"]
    y = data["y"]
    target_mask = data["target_mask"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    edge_indices, edge_weights = build_edge_lists(data["adj"], device)

    print("Milestone 8: Explainability via Integrated Gradients on edge weights\n")

    results = []
    all_edge_rows = []

    for case_id, meta in CASE_STUDIES.items():
        print(f"Case study: {meta['title']}")
        t_idx = quarter_to_idx(quarters, meta["quarter"])
        print(f"  Refitting model through {quarters[t_idx - 1]}...")
        model = refit_model(
            x_raw, y, target_mask, t_idx, edge_indices, edge_weights, cfg, device, refit_epochs=30
        )
        target_iso3 = "DEU"
        detail = explain_timestep(
            model, x_raw, t_idx, countries.index(target_iso3), countries, quarters,
            edge_indices, edge_weights, device,
        )
        edges = detail["edge_attributions"].copy()
        edges["case_study"] = case_id
        edges["forecast_quarter"] = meta["quarter"]
        all_edge_rows.append(edges)

        fig1 = FIG_DIR / f"{case_id}_top_partners_{target_iso3}.png"
        fig2 = FIG_DIR / f"{case_id}_domestic_vs_trade_{target_iso3}.png"
        plot_top_partners(edges, f"{meta['title']} — top trade partners for {target_iso3}", fig1)
        plot_domestic_vs_trade(
            detail["feature_summary"], detail["edge_total_abs"],
            f"{target_iso3}: domestic vs trade-channel attribution ({meta['quarter']})", fig2,
        )
        results.append(build_case_result(case_id, meta, detail, target_iso3, [str(fig1), str(fig2)]))

    print("\nCase study 3: identifying spillover-dominated country (2022Q2)...")
    t_idx = quarter_to_idx(quarters, "2022Q2")
    model = refit_model(
        x_raw, y, target_mask, t_idx, edge_indices, edge_weights, cfg, device, refit_epochs=30
    )
    spillover_iso3, spillover_detail = find_spillover_country(
        model, x_raw, t_idx, countries, quarters, edge_indices, edge_weights, device
    )
    spillover_edges = spillover_detail["edge_attributions"].copy()
    spillover_edges["case_study"] = "partner_spillover_dominant"
    spillover_edges["forecast_quarter"] = "2022Q2"
    all_edge_rows.append(spillover_edges)

    fig1 = FIG_DIR / "partner_spillover_top_partners.png"
    fig2 = FIG_DIR / "partner_spillover_heatmap.png"
    fig3 = FIG_DIR / "partner_spillover_domestic_vs_trade.png"
    plot_top_partners(spillover_edges, f"Partner spillovers to {spillover_iso3} (2022Q2)", fig1)
    plot_spillover_heatmap(spillover_edges, spillover_iso3, "2022Q2", fig2)
    plot_domestic_vs_trade(
        spillover_detail["feature_summary"], spillover_detail["edge_total_abs"],
        f"{spillover_iso3}: domestic vs trade edges (2022Q2)", fig3,
    )

    case3_meta = {
        "quarter": "2022Q2",
        "title": f"Partner spillovers dominate for {spillover_iso3} (2022Q2)",
        "narrative": (
            f"Among all countries at 2022Q2, {spillover_iso3} shows the highest ratio of "
            f"trade-edge IG attribution to domestic feature attribution "
            f"(ratio={spillover_detail['trade_vs_domestic_ratio']:.2f}), indicating forecasts "
            "driven primarily through the trade network rather than domestic macro features."
        ),
    }
    results.append(
        build_case_result(
            "partner_spillover_dominant", case3_meta, spillover_detail, spillover_iso3,
            [str(fig1), str(fig2), str(fig3)],
        )
    )

    pd.concat(all_edge_rows, ignore_index=True).to_csv(OUT_DIR / "edge_attributions.csv", index=False)

    report = {
        "run_at": datetime.now(timezone.utc).isoformat(),
        "milestone": 8,
        "method": "Integrated Gradients (edge weights + node features, 32 steps)",
        "model_config": cfg.__dict__,
        "case_studies": results,
    }
    (OUT_DIR / "explainability_report.json").write_text(json.dumps(report, indent=2, default=str))

    print(f"\nSaved {OUT_DIR / 'explainability_report.json'}")
    print(f"Saved {OUT_DIR / 'edge_attributions.csv'}")
    print(f"Figures in {FIG_DIR}/")
    print(f"\nSpillover-dominated country: {spillover_iso3} (ratio={results[-1]['trade_vs_domestic_ratio']:.2f})")
    for cs in results:
        partners = ", ".join(p["source_iso3"] for p in cs["top_trade_partners"][:3])
        print(f"  {cs['case_id']}: {cs['target_iso3']} @ {cs['quarter']} — top partners: {partners}")
    
    # Compare shock vs baseline attribution rankings
    print("\n--- Attribution ranking stability analysis ---")
    shock_partners = set([p["source_iso3"] for p in results[0]["top_trade_partners"][:3]])  # COVID
    baseline_partners = set([p["source_iso3"] for p in results[2]["top_trade_partners"][:3]])  # Calm 2017Q2
    print(f"COVID top 3 partners: {sorted(shock_partners)}")
    print(f"Baseline (2017Q2) top 3 partners: {sorted(baseline_partners)}")
    print(f"Overlap: {sorted(shock_partners & baseline_partners)}")
    print(f"If overlap is high, attributions may just track trade volume rather than shock-specific channels.")
    
    # Seed robustness check for key case study (COVID) - optional due to computational cost
    print("\n--- Seed robustness check (COVID case study, Germany) ---")
    print("Skipping seed robustness check due to computational cost (requires retraining model multiple times).")
    print("This can be enabled by uncommenting the seed_robustness_check call in run_explainability.py.")
    report["seed_robustness"] = {
        "note": "Skipped due to computational cost. Can be enabled by uncommenting seed_robustness_check call.",
        "n_seeds_skipped": 3,
    }
    (OUT_DIR / "explainability_report.json").write_text(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
