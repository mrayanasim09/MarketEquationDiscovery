"""Machine-checkable journal-readiness contract for the prospective v2.1 run."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.models.graphs.factory import GRAPH_VARIANTS
from src.models.provenance import sha256_file

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "experiments/results/v2_1/configs/benchmark_engine_v2_1.json"
PROTOCOL = ROOT / "docs/research_protocol_v2_1.md"

REQUIRED_MODELS = {
    "persistence", "arima", "var", "ets", "dynamic_factor", "ridge",
    "gradient_boosting", "mlp", "lstm", "tcn", "gcn", "temporal_graph",
}
REQUIRED_DETERMINISTIC_METRICS = {"rmse", "mae", "smape"}
REQUIRED_PROBABILISTIC_METRICS = {
    "crps", "interval_coverage_80", "interval_width_80",
    "interval_coverage_95", "interval_width_95",
}
REQUIRED_COMPARATORS = {
    "arima", "ets", "dynamic_factor", "ridge", "gradient_boosting", "lstm", "tcn",
}
REQUIRED_PROTOCOL_HEADINGS = (
    "## 3. Pre-specified model registry",
    "## 4. Hyperparameter selection and tuning isolation",
    "## 5. Forecast outputs and deterministic evaluation",
    "## 6. Probabilistic evaluation",
    "## 7. Statistical inference",
    "## 8. Execution integrity and retention",
)


def load_config() -> dict[str, Any]:
    return json.loads(CONFIG.read_text())


def contract_errors(cfg: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if cfg.get("engine_version") != "2.1":
        errors.append("v2.1 configuration must declare engine_version 2.1")
    if not str(cfg.get("configuration_id", "")).strip():
        errors.append("v2.1 configuration requires a unique configuration_id")
    if set(cfg.get("models", [])) != REQUIRED_MODELS:
        errors.append("v2.1 model registry is incomplete or contains unapproved models")
    if tuple(cfg.get("graph_variants", [])) != tuple(GRAPH_VARIANTS):
        errors.append("v2.1 graph registry does not exactly match the graph factory registry")
    if set(cfg.get("graph_models", [])) != {"gcn", "temporal_graph"}:
        errors.append("v2.1 graph_models must be exactly gcn and temporal_graph")

    metrics = cfg.get("metrics", {})
    if set(metrics.get("deterministic", [])) != REQUIRED_DETERMINISTIC_METRICS:
        errors.append("v2.1 deterministic metrics must be exactly RMSE, MAE, and sMAPE")
    if set(metrics.get("probabilistic", [])) != REQUIRED_PROBABILISTIC_METRICS:
        errors.append("v2.1 probabilistic metrics must include CRPS and 80%/95% coverage and width")
    if metrics.get("origin_aggregation") != "mean across countries within forecast origin":
        errors.append("v2.1 metrics must aggregate countries within forecast origin")

    probabilistic = cfg.get("probabilistic_forecasts", {})
    if probabilistic.get("required") is not True or probabilistic.get("calibration_split") != "training_and_validation_only":
        errors.append("v2.1 probabilistic forecasts must be required and calibrated without test data")
    if probabilistic.get("nominal_coverages") != [0.8, 0.95]:
        errors.append("v2.1 probabilistic forecasts must report 80% and 95% intervals")

    split = cfg.get("split_rule", {})
    if split.get("selection_split") != "validation" or split.get("test_set_tuning_forbidden") is not True:
        errors.append("v2.1 split rule must isolate tuning to the validation period")
    if split.get("training_origins") != ["2011Q2", "2014Q4"] or split.get("validation_origins") != ["2015Q1", "2016Q4"] or split.get("test_origins") != ["2017Q1", "2025Q3"]:
        errors.append("v2.1 split boundaries differ from the prospective locked contract")

    testing = cfg.get("statistical_testing", {})
    if testing.get("primary_loss") != "absolute_error" or testing.get("unit") != "forecast_origin":
        errors.append("v2.1 primary DM inference must use origin-level absolute error")
    if set(testing.get("comparators", [])) != REQUIRED_COMPARATORS:
        errors.append("v2.1 DM comparator family is incomplete or altered")
    if testing.get("dm_variant") != "harvey_leybourne_newbold_bartlett_hac" or testing.get("hac_max_lag") != "horizon_minus_one":
        errors.append("v2.1 DM test must use pre-specified HLN Bartlett-HAC inference")
    bootstrap = testing.get("bootstrap", {})
    if bootstrap != {"method": "moving_block", "block_length": 4, "draws": 2000, "confidence_level": 0.95}:
        errors.append("v2.1 bootstrap configuration differs from the locked origin-block procedure")
    multiplicity = testing.get("multiplicity", {})
    if multiplicity != {"method": "benjamini_hochberg", "family": "all_primary_graph_comparator_horizon_tests", "q": 0.05}:
        errors.append("v2.1 multiplicity correction is not the locked BH primary-test family")

    integrity = cfg.get("execution_integrity", {})
    required_integrity = {
        "allow_partial_registry": False,
        "transactional_staging": True,
        "publish_only_after_completeness_validation": True,
        "preserve_failed_runs": True,
        "overwrite_existing_manifests": False,
    }
    if integrity != required_integrity:
        errors.append("v2.1 execution-integrity controls are incomplete or altered")
    return errors


def execution_errors(cfg: dict[str, Any]) -> list[str]:
    """Verify that the documented contract has an executable implementation.

    The tuning manifest is intentionally a hard gate: creating an empty manifest
    is not sufficient, and the validator will not create or mutate one.
    """
    errors: list[str] = []
    try:
        from src.models.evaluate_benchmark_engine_v2_1 import dm_tests, score_forecasts
        from src.models.run_benchmark_engine_v2_1 import RUNNER_OUTPUTS, SUPPORTED_MODELS
        from src.models.storage_v2_1 import FORECAST_COLUMNS
        from src.models.tuning.manifest import require_tuning_manifest
    except ImportError as exc:
        return [f"v2.1 execution implementation cannot be imported: {exc}"]
    if set(SUPPORTED_MODELS) != set(cfg["models"]):
        errors.append("v2.1 runner does not support every configured model")
    required_forecast = {"mean", "lower_80", "upper_80", "lower_95", "upper_95", "actual"}
    if not required_forecast.issubset(FORECAST_COLUMNS):
        errors.append("v2.1 forecast schema lacks required probabilistic outputs")
    required_metric_names = set(cfg["metrics"]["deterministic"]) | set(cfg["metrics"]["probabilistic"])
    if not {"score_forecasts", "dm_tests"}.issubset({score_forecasts.__name__, dm_tests.__name__}):
        errors.append("v2.1 journal metric or statistical-test implementation is unavailable")
    if not {"forecasts.parquet", "metrics.parquet", "dm_tests.parquet", "checkpoints", "environment_manifest.json"}.issubset(RUNNER_OUTPUTS):
        errors.append("v2.1 runner output contract is incomplete")
    if required_metric_names != REQUIRED_DETERMINISTIC_METRICS | REQUIRED_PROBABILISTIC_METRICS:
        errors.append("v2.1 configured metric registry is not journal-complete")
    try:
        require_tuning_manifest()
    except (FileNotFoundError, RuntimeError, KeyError, json.JSONDecodeError) as exc:
        errors.append(f"v2.1 tuning isolation gate failed: {exc}")
    return errors


def protocol_errors(cfg: dict[str, Any]) -> list[str]:
    if not PROTOCOL.is_file():
        return ["v2.1 protocol document is missing"]
    text = PROTOCOL.read_text()
    errors = [f"v2.1 protocol lacks required section: {heading}" for heading in REQUIRED_PROTOCOL_HEADINGS if heading not in text]
    if cfg.get("protocol_document") != str(PROTOCOL.relative_to(ROOT)):
        errors.append("v2.1 configuration does not point to the canonical protocol document")
    return errors


def validation_report() -> dict[str, Any]:
    cfg = load_config()
    errors = contract_errors(cfg) + protocol_errors(cfg) + execution_errors(cfg)
    return {
        "passed": not errors,
        "engine_version": cfg.get("engine_version"),
        "configuration_id": cfg.get("configuration_id"),
        "configuration_sha256": sha256_file(CONFIG),
        "protocol": str(PROTOCOL.relative_to(ROOT)),
        "protocol_sha256": sha256_file(PROTOCOL) if PROTOCOL.is_file() else None,
        "errors": errors,
    }


def main() -> int:
    report = validation_report()
    if report["passed"]:
        print("V2.1 CONTRACT VALIDATION PASSED")
        print(f"configuration_id={report['configuration_id']}")
        print(f"configuration_sha256={report['configuration_sha256']}")
        return 0
    print("V2.1 CONTRACT VALIDATION FAILED")
    print("\n".join(f"- {error}" for error in report["errors"]))
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
