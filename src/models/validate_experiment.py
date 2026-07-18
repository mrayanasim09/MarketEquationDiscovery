"""Fail-closed preflight validation for the locked v2 benchmark engine."""
from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.models.graphs.factory import GRAPH_VARIANTS
from src.models.provenance import git_commit, required_provenance_fields, sha256_file
from src.models.storage import FORECAST_COLUMNS
from src.models.training import quarterly_step_count
from src.transform.common import V2_PROCESSED, require_validated_raw

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "experiments/results/v2"
CONFIG = RESULTS / "configs/benchmark_engine.json"


def _git_errors() -> list[str]:
    try:
        inside = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--is-inside-work-tree"], check=True, capture_output=True, text=True).stdout.strip()
        if inside != "true":
            return ["canonical workspace is not inside a Git work tree"]
        git_commit()
        changed = subprocess.run(
            ["git", "-C", str(ROOT), "diff", "--name-only", "HEAD", "--", "src", "data/raw/v2", "data/processed/v2", "experiments/results/v2/configs", "docs", "requirements.txt"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.splitlines()
        if changed:
            return [f"benchmark paths have uncommitted changes: {changed}"]
    except (OSError, subprocess.CalledProcessError) as exc:
        return [f"Git commit provenance is unavailable: {exc}"]
    return []


def _quarter_horizon_errors(samples: pd.DataFrame) -> list[str]:
    """Regression guard for pandas-compatible quarterly Period arithmetic."""
    origin = pd.Period(str(samples["origin_quarter"].iloc[0]), freq="Q")
    probe = pd.PeriodIndex([origin, origin + 1, origin + 4], freq="Q")
    if quarterly_step_count(probe[0], probe[1]) != 1 or quarterly_step_count(probe[0], probe[2]) != 4:
        return ["quarterly PeriodIndex arithmetic does not return expected integer step counts"]
    parsed = samples.copy()
    for column in ("origin_quarter", "target_quarter", "macro_feature_quarter"):
        parsed[column] = pd.PeriodIndex(parsed[column], freq="Q")
    target_steps = parsed["target_quarter"].map(lambda value: value.ordinal) - parsed["origin_quarter"].map(lambda value: value.ordinal)
    input_steps = parsed["target_quarter"].map(lambda value: value.ordinal) - parsed["macro_feature_quarter"].map(lambda value: value.ordinal)
    errors: list[str] = []
    if not target_steps.eq(parsed["horizon_quarters"]).all():
        errors.append("forecast target quarters do not match their locked horizons")
    if not input_steps.eq(parsed["horizon_quarters"] + 1).all():
        errors.append("permitted macro-input-to-target steps must equal horizon plus the locked one-quarter availability lag")
    return errors


def _sequence_errors(samples: pd.DataFrame) -> list[str]:
    panel = pd.read_csv(V2_PROCESSED / "quarterly_feature_panel.csv")
    required = {"entity_id", "quarter", "cpi_yoy", "energy_cpi_yoy"}
    if missing := sorted(required - set(panel.columns)):
        return [f"quarterly feature panel lacks sequence columns: {missing}"]
    lookup = panel.set_index(["entity_id", "quarter"])[["cpi_yoy", "energy_cpi_yoy"]]
    errors: list[str] = []
    for row in samples.loc[samples["split"].eq("test")].itertuples(index=False):
        history = pd.period_range(end=pd.Period(row.macro_feature_quarter, freq="Q"), periods=4, freq="Q")
        keys = [(row.country, str(period)) for period in history]
        if not set(keys).issubset(lookup.index):
            errors.append("a test sequence model input lacks four registered quarterly observations")
            break
        if lookup.loc[keys].isna().any().any():
            errors.append("a test sequence model input has missing CPI or energy values")
            break
    return errors


def main() -> int:
    errors = _git_errors()
    try:
        require_validated_raw()
    except (FileNotFoundError, KeyError, RuntimeError, json.JSONDecodeError) as exc:
        errors.append(f"raw validation/hash gate failed: {exc}")

    cfg = json.loads(CONFIG.read_text())
    samples = pd.read_csv(V2_PROCESSED / "forecast_samples.csv")
    countries = json.loads((V2_PROCESSED / "countries.json").read_text())
    quarters = json.loads((V2_PROCESSED / "quarters.json").read_text())
    adjacency = np.load(V2_PROCESSED / "adjacency_directed_trade_eur.npy")

    if len(cfg["seeds"]) != 20 or len(set(cfg["seeds"])) != 20:
        errors.append("configuration must contain exactly 20 unique seeds")
    if cfg["epochs"] != 30:
        errors.append("locked benchmark configuration must use 30 epochs")
    if set(cfg["horizons"]) != {1, 2, 4}:
        errors.append("locked benchmark configuration must use horizons {1, 2, 4}")
    if len(countries) != 20:
        errors.append("processed benchmark panel must contain exactly 20 countries")
    if tuple(cfg["graph_variants"]) != tuple(GRAPH_VARIANTS):
        errors.append("configuration graph_variants do not exactly match the graph-factory runtime registry")
    if samples.duplicated(["country", "origin_quarter", "horizon_quarters"]).any():
        errors.append("processed samples contain duplicate forecasts")
    if samples.macro_feature_quarter.ge(samples.origin_quarter).any() or samples.trade_graph_quarter.ge(samples.origin_quarter).any():
        errors.append("future/same-origin feature or graph detected")
    if samples.target_quarter.le(samples.origin_quarter).any() or samples.target_cpi_yoy.isna().any():
        errors.append("invalid or missing target")
    if not set(samples.trade_graph_quarter).issubset(set(quarters)):
        errors.append("sample graph quarter missing from snapshots")
    if adjacency.shape != (len(quarters), len(countries), len(countries)):
        errors.append("graph tensor shape mismatches node/quarter contract")
    errors.extend(_quarter_horizon_errors(samples))
    errors.extend(_sequence_errors(samples))

    forecast_path = RESULTS / "forecasts.parquet"
    manifest_path = RESULTS / "run_manifest.json"
    if forecast_path.exists():
        forecasts = pd.read_parquet(forecast_path)
        missing = sorted(set(FORECAST_COLUMNS) - set(forecasts.columns))
        if missing:
            errors.append(f"stored forecasts lack required provenance/schema columns: {missing}")
        else:
            keys = ["run_id", "model_name", "model_variant", "seed", "horizon", "forecast_origin", "country"]
            if forecasts.duplicated(keys).any():
                errors.append("duplicate stored forecasts violate append-only run-level keys")
            if forecasts[required_provenance_fields()].isna().any().any():
                errors.append("stored forecasts have missing execution provenance")
            neural = forecasts[forecasts.model_name.isin(["mlp", "lstm", "tcn", "gcn", "temporal_graph"])]
            if not neural.empty and set(neural.seed.astype(int).unique()) != set(cfg["seeds"]):
                errors.append("stored neural results do not contain all configured seeds")
    if manifest_path.exists():
        manifests = json.loads(manifest_path.read_text())
        runs = manifests.get("runs") if isinstance(manifests, dict) else None
        if not isinstance(runs, list):
            errors.append("run_manifest.json must contain a runs list")
        elif len([run.get("run_id") for run in runs if isinstance(run, dict)]) != len({run.get("run_id") for run in runs if isinstance(run, dict)}):
            errors.append("run_manifest.json contains non-unique run_id values")

    report = {
        "passed": not errors,
        "mode": "post-run" if forecast_path.exists() else "pre-training",
        "validated_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit() if not _git_errors() else None,
        "configuration_id": f"sha256:{sha256_file(CONFIG)}",
        "required_forecast_schema": FORECAST_COLUMNS,
        "errors": errors,
        "seed_count": len(cfg["seeds"]),
        "countries": len(countries),
        "sample_rows": len(samples),
        "graph_shape": list(adjacency.shape),
    }
    (RESULTS / "metadata").mkdir(parents=True, exist_ok=True)
    (RESULTS / "metadata/validation.json").write_text(json.dumps(report, indent=2) + "\n")
    if errors:
        print("V2 BENCHMARK ENGINE VALIDATION FAILED")
        print("\n".join(f"- {error}" for error in errors))
        return 1
    print(f"V2 BENCHMARK ENGINE VALIDATION PASSED ({report['mode']}): {len(countries)} countries, {len(cfg['seeds'])} seeds")
    print("LOCKED BENCHMARK EXECUTION READY")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
