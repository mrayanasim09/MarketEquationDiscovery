"""Post-execution validation and hash registry for completed v2.1 benchmark outputs."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from src.models.storage_v2_1 import RESULTS  # noqa: E402
from src.models.validate_v2_1_contract import load_config  # noqa: E402

CONFIG_PATH = RESULTS / "configs/benchmark_engine_v2_1.json"
TUNING_MANIFEST = RESULTS / "tuning" / "tuning_manifest.json"

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()

def main() -> int:
    print("Starting post-execution validation of v2.1 results...")
    errors: list[str] = []

    # 1. File existence checks
    for filename in ("forecasts.parquet", "metrics.parquet", "dm_tests.parquet", "run_manifest.json", "environment_manifest.json"):
        if not (RESULTS / filename).exists():
            errors.append(f"Missing canonical output file: {filename}")

    if not (RESULTS / "checkpoints").is_dir():
        errors.append("Missing checkpoints directory")

    if errors:
        print("Validation FAILED due to missing files:")
        for err in errors:
            print(f"- {err}")
        return 1

    # Load data
    try:
        forecasts = pd.read_parquet(RESULTS / "forecasts.parquet")
        metrics = pd.read_parquet(RESULTS / "metrics.parquet")
        dm_tests_df = pd.read_parquet(RESULTS / "dm_tests.parquet")
    except Exception as exc:
        print(f"Error loading parquet files: {exc}")
        return 1

    cfg = load_config()
    samples = pd.read_csv(ROOT / "data/processed/v2/forecast_samples.csv")
    test_samples = samples[samples.split.eq("test")]

    # 2. Schema and Dimensions on forecasts
    expected_models = set(cfg["models"])
    expected_horizons = set(cfg["horizons"])
    expected_seeds = set(cfg["seeds"])
    expected_variants = set(cfg["graph_variants"])

    actual_models = set(forecasts.model.unique())
    actual_horizons = set(forecasts.horizon.unique())

    if actual_models != expected_models:
        errors.append(f"Model mismatch in forecasts: expected {expected_models}, got {actual_models}")
    if actual_horizons != expected_horizons:
        errors.append(f"Horizon mismatch in forecasts: expected {expected_horizons}, got {actual_horizons}")

    # Row count check
    # Deterministic models: 1 seed ("deterministic"), 1 graph variant ("none")
    # Non-graph neural models (mlp, lstm, tcn): 20 seeds, 1 graph variant ("none")
    # Graph neural models (gcn, temporal_graph): 20 seeds, 8 graph variants
    graph_neural = {"gcn", "temporal_graph"}
    non_graph_neural = {"mlp", "lstm", "tcn"}
    deterministic_models = set(cfg["models"]) - graph_neural - non_graph_neural
    n_det = len(deterministic_models)
    expected_rows = (
        len(test_samples) * n_det
        + len(test_samples) * len(non_graph_neural) * len(expected_seeds)
        + len(test_samples) * len(graph_neural) * len(expected_seeds) * len(expected_variants)
    )
    if len(forecasts) != expected_rows:
        errors.append(f"Forecast row count mismatch: expected {expected_rows}, got {len(forecasts)}")

    # Deterministic models validation
    det_models = {"persistence", "arima", "var", "ets", "dynamic_factor", "ridge", "gradient_boosting"}
    for model in det_models:
        model_df = forecasts[forecasts.model == model]
        if not model_df.empty:
            if set(model_df.seed.unique()) != {"deterministic"}:
                errors.append(f"Deterministic model {model} has non-deterministic seeds: {model_df.seed.unique()}")
            if set(model_df.graph_variant.unique()) != {"none"}:
                errors.append(f"Deterministic model {model} has graph variants: {model_df.graph_variant.unique()}")

    # Non-graph neural models validation
    ng_neural = {"mlp", "lstm", "tcn"}
    for model in ng_neural:
        model_df = forecasts[forecasts.model == model]
        if not model_df.empty:
            seeds = set(model_df.seed.unique().astype(int))
            if seeds != expected_seeds:
                errors.append(f"Non-graph neural model {model} has seed mismatch: expected {expected_seeds}, got {seeds}")
            if set(model_df.graph_variant.unique()) != {"none"}:
                errors.append(f"Non-graph neural model {model} has graph variants: {model_df.graph_variant.unique()}")

    # Graph neural models validation
    g_neural = {"gcn", "temporal_graph"}
    for model in g_neural:
        model_df = forecasts[forecasts.model == model]
        if not model_df.empty:
            seeds = set(model_df.seed.unique().astype(int))
            if seeds != expected_seeds:
                errors.append(f"Graph neural model {model} has seed mismatch: expected {expected_seeds}, got {seeds}")
            variants = set(model_df.graph_variant.unique())
            if variants != expected_variants:
                errors.append(f"Graph neural model {model} has variant mismatch: expected {expected_variants}, got {variants}")

    # Origin counts per horizon
    for horizon, expected_origins in {1: 35, 2: 34, 4: 32}.items():
        hor_df = forecasts[forecasts.horizon == horizon]
        if not hor_df.empty:
            origins = hor_df.forecast_origin.nunique()
            if origins != expected_origins:
                errors.append(f"Horizon {horizon} has unexpected unique origins: expected {expected_origins}, got {origins}")

    # 3. Quality checks
    # Duplicate keys check
    duplicate_keys = ["run_id", "country", "forecast_origin", "horizon", "model", "graph_variant", "seed"]
    if forecasts.duplicated(duplicate_keys).any():
        errors.append("Forecasts contain duplicate transaction keys")

    # Incoherent intervals check
    coherent = (forecasts.lower_95 <= forecasts.lower_80) & \
               (forecasts.lower_80 <= forecasts["mean"]) & \
               (forecasts["mean"] <= forecasts.upper_80) & \
               (forecasts.upper_80 <= forecasts.upper_95)
    if not coherent.all():
        num_incoherent = (~coherent).sum()
        errors.append(f"Incoherent predictive intervals found in {num_incoherent} rows")

    # Null values in forecasts check
    null_cols = forecasts.isnull().any()
    if null_cols.any():
        errors.append(f"Null values found in forecasts columns: {list(null_cols[null_cols].index)}")

    # 4. Metrics Parquet checks
    expected_metric_names = set(cfg["metrics"]["deterministic"]) | set(cfg["metrics"]["probabilistic"])
    actual_metrics = set(metrics.metric.unique())
    if actual_metrics != expected_metric_names:
        errors.append(f"Metrics parquet metric names mismatch: expected {expected_metric_names}, got {actual_metrics}")

    # 5. DM Tests Parquet checks
    comparators = set(cfg["statistical_testing"]["comparators"])
    graph_models = set(cfg["graph_models"])
    expected_dm_rows = len(graph_models) * len(expected_variants) * len(expected_seeds) * len(expected_horizons) * len(comparators)
    if len(dm_tests_df) != expected_dm_rows:
        errors.append(f"DM tests row count mismatch: expected {expected_dm_rows}, got {len(dm_tests_df)}")

    # Save output hashes registry if validation passes
    if not errors:
        print("Validation PASSED successfully.")
        hashes = {
            "run_id": str(forecasts.run_id.iloc[0]),
            "timestamp": pd.Timestamp.now(tz="UTC").isoformat(),
            "git_commit": sha256_file(CONFIG_PATH),
            "configuration_sha256": sha256_file(CONFIG_PATH),
            "tuning_manifest_sha256": sha256_file(TUNING_MANIFEST),
            "files": {
                "forecasts.parquet": {
                    "sha256": sha256_file(RESULTS / "forecasts.parquet"),
                    "rows": len(forecasts)
                },
                "metrics.parquet": {
                    "sha256": sha256_file(RESULTS / "metrics.parquet"),
                    "rows": len(metrics)
                },
                "dm_tests.parquet": {
                    "sha256": sha256_file(RESULTS / "dm_tests.parquet"),
                    "rows": len(dm_tests_df)
                }
            }
        }
        try:
            from src.models.provenance import git_commit
            hashes["git_commit"] = git_commit()
        except Exception:
            pass

        output_hashes_path = RESULTS / "metadata" / "output_hashes.json"
        output_hashes_path.parent.mkdir(parents=True, exist_ok=True)
        output_hashes_path.write_text(json.dumps(hashes, indent=2, sort_keys=True) + "\n")
        print(f"Output hashes registry written to {output_hashes_path}")
        return 0
    else:
        print("Validation FAILED with the following errors:")
        for err in errors:
            print(f"- {err}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
