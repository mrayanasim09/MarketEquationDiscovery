"""Immutable execution provenance for the locked v2 benchmark."""
from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUN_MANIFEST_REQUIRED_FIELDS = [
    "run_id", "git_commit", "dataset_version", "configuration_id", "execution_timestamp",
    "hardware_environment", "python_version", "dependency_versions", "models",
    "graph_variants", "seeds", "epochs", "forecast_horizon", "train_period",
    "validation_period", "test_period",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_commit() -> str:
    result = subprocess.run(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def build_execution_provenance(config_path: Path) -> dict[str, str]:
    """Create one provenance record shared by every forecast in one execution."""
    config_hash = sha256_file(config_path)
    dataset_hash = sha256_file(ROOT / "data" / "processed" / "v2" / "forecast_samples.csv")
    timestamp = datetime.now(timezone.utc).isoformat()
    commit = git_commit()
    run_id = f"v2-{commit[:12]}-{config_hash[:12]}-{timestamp.replace(':', '').replace('+00:00', 'Z')}"
    return {
        "run_id": run_id,
        "git_commit": commit,
        "dataset_version": f"forecast_samples_sha256:{dataset_hash}",
        "configuration_id": f"sha256:{config_hash}",
        "execution_timestamp": timestamp,
    }


def required_provenance_fields() -> list[str]:
    return ["run_id", "git_commit", "dataset_version", "configuration_id", "execution_timestamp"]
