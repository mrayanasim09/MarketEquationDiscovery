"""Append-only tuning-manifest persistence and provenance checks."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.models.provenance import sha256_file

ROOT = Path(__file__).resolve().parents[3]
RESULTS = ROOT / "experiments/results/v2_1"
CONFIG = RESULTS / "configs/benchmark_engine_v2_1.json"
TUNING_MANIFEST = RESULTS / "tuning" / "tuning_manifest.json"


def require_tuning_manifest() -> dict[str, Any]:
    """Return a completed immutable validation-only manifest or fail closed."""
    if not TUNING_MANIFEST.is_file():
        raise FileNotFoundError("v2.1 tuning manifest is required before final-test execution")
    manifest = json.loads(TUNING_MANIFEST.read_text())
    required = {"run_id", "git_commit", "configuration_id", "status", "configuration_sha256", "selection_split", "candidate_registry", "validation_losses", "selected_parameters", "calibration", "records", "written_at"}
    missing = sorted(required - set(manifest))
    if missing:
        raise RuntimeError(f"v2.1 tuning manifest lacks required fields: {missing}")
    if manifest["status"] != "completed" or manifest["selection_split"] != "validation":
        raise RuntimeError("v2.1 tuning manifest is not a completed validation-only selection")
    if manifest["configuration_sha256"] != sha256_file(CONFIG):
        raise RuntimeError("v2.1 tuning manifest does not match the locked configuration")
    if not manifest["validation_losses"] or not manifest["selected_parameters"] or not manifest["calibration"] or not manifest["records"]:
        raise RuntimeError("v2.1 tuning manifest has no validation results, selections, calibration, or per-model records")
    return manifest


def write_tuning_manifest(payload: dict[str, Any]) -> Path:
    """Write once. A completed selection is immutable and cannot be overwritten."""
    if TUNING_MANIFEST.exists():
        raise FileExistsError("refusing to overwrite immutable v2.1 tuning manifest")
    required = {"run_id", "git_commit", "configuration_id", "candidate_registry", "validation_losses", "selected_parameters", "calibration", "records"}
    missing = sorted(required - set(payload))
    if missing:
        raise ValueError(f"tuning payload lacks required fields: {missing}")
    RESULTS.mkdir(parents=True, exist_ok=True)
    record = {
        **payload,
        "status": "completed",
        "selection_split": "validation",
        "configuration_sha256": sha256_file(CONFIG),
        "written_at": datetime.now(timezone.utc).isoformat(),
    }
    TUNING_MANIFEST.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    return TUNING_MANIFEST
