"""Fail closed when the v2 benchmark outputs are not submission-grade robust."""
from __future__ import annotations
import json
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "experiments/configs/v2_benchmark.json"
RESULTS = ROOT / "experiments/results"
OUT = RESULTS / "v2_robustness_preflight.json"

def main() -> int:
    config = json.loads(CONFIG.read_text())
    metadata = json.loads((RESULTS / "v2_run_metadata.json").read_text())
    forecasts = pd.read_csv(RESULTS / "v2_forecasts.csv")
    errors = []
    required_seeds = config["neural"]["seeds"]
    run_seeds = metadata.get("neural_seeds_run", [])
    if run_seeds != required_seeds:
        errors.append(f"configured {len(required_seeds)} neural seeds but results contain {len(run_seeds)}: {run_seeds}")
    if "seed" not in forecasts.columns:
        errors.append("forecast file has no seed column; seed-level metrics and DM aggregation cannot be computed")
    if metadata.get("epochs") != config["neural"]["epochs"]:
        errors.append(f"run used {metadata.get('epochs')} epochs but locked configuration requires {config['neural']['epochs']}")
    required_models = {"bayesian_var", "tcn", "gradient_boosting"}
    observed_models = set(forecasts["model"])
    missing_models = sorted(required_models - observed_models)
    if missing_models:
        errors.append(f"required stronger baselines are not implemented in current results: {missing_models}")
    report = {
        "passed": not errors,
        "configured_seed_count": len(required_seeds),
        "run_seed_count": len(run_seeds),
        "forecast_has_seed_column": "seed" in forecasts.columns,
        "locked_epochs": config["neural"]["epochs"],
        "run_epochs": metadata.get("epochs"),
        "errors": errors,
        "interpretation": "Do not use current results for a submission-grade conclusion if this preflight fails.",
    }
    OUT.write_text(json.dumps(report, indent=2) + "\n")
    if errors:
        print("V2 ROBUSTNESS PREFLIGHT FAILED")
        print("\n".join(f"- {error}" for error in errors))
        return 1
    print("V2 ROBUSTNESS PREFLIGHT PASSED")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
