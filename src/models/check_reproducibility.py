"""Read-only final hygiene check for a locked v2 benchmark execution."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

from src.models.graphs.factory import GRAPH_VARIANTS
from src.models.provenance import git_commit, required_provenance_fields, sha256_file
from src.models.storage import FORECAST_COLUMNS
from src.transform.common import require_validated_raw

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "experiments/results/v2"
CONFIG = RESULTS / "configs/benchmark_engine.json"
BENCHMARK_PATHS = [
    "src/models",
    "src/transform",
    "src/validate_v2_inputs.py",
    "data/raw/v2",
    "data/processed/v2",
    "experiments/results/v2/configs",
    "docs",
    "requirements.txt",
]
SCIENTIFIC_ARTIFACTS = [
    ROOT / "data/raw/v2/macro/macro_observations.csv",
    ROOT / "data/raw/v2/trade/trade_observations.csv",
    ROOT / "data/processed/v2/adjacency_directed_trade_eur.npy",
    ROOT / "data/processed/v2/adjacency_directed_observed_mask.npy",
    ROOT / "data/processed/v2/forecast_samples.csv",
    ROOT / "data/processed/v2/quarterly_feature_panel.csv",
    ROOT / "data/processed/v2/quarterly_trade_edges_directed.csv",
]
FORBIDDEN_RESULT_FILES = [
    RESULTS / "forecasts.parquet",
    RESULTS / "metrics.parquet",
    RESULTS / "dm_tests.parquet",
    RESULTS / "dm_tests_adjusted.parquet",
    RESULTS / "metrics_seed_summary.parquet",
]


def git_output(*args: str) -> str:
    return subprocess.run(["git", "-C", str(ROOT), *args], check=True, capture_output=True, text=True).stdout.strip()


def main() -> int:
    errors: list[str] = []
    try:
        if git_output("rev-parse", "--is-inside-work-tree") != "true":
            errors.append("canonical workspace is not a Git work tree")
        commit = git_commit()
        if git_output("diff", "--name-only", "HEAD", "--", *BENCHMARK_PATHS):
            errors.append("benchmark code, data, configuration, or documentation has uncommitted changes")
        if git_output("diff", "--name-only", "HEAD", "--", "data/raw/v2", "data/processed/v2"):
            errors.append("raw or processed scientific artifacts differ from the committed state")
    except (OSError, subprocess.CalledProcessError) as exc:
        errors.append(f"Git provenance check failed: {exc}")
        commit = "unavailable"

    if not CONFIG.is_file():
        errors.append("locked benchmark configuration is missing")
    else:
        config = json.loads(CONFIG.read_text())
        if tuple(config.get("graph_variants", [])) != tuple(GRAPH_VARIANTS):
            errors.append("locked configuration graph registry differs from graph factory")
    try:
        require_validated_raw()
    except (FileNotFoundError, KeyError, RuntimeError, json.JSONDecodeError) as exc:
        errors.append(f"raw validation hash gate failed: {exc}")

    missing_schema = sorted(set(required_provenance_fields()) - set(FORECAST_COLUMNS))
    if missing_schema:
        errors.append(f"forecast schema lacks required provenance fields: {missing_schema}")
    if "graph_variant" not in FORECAST_COLUMNS:
        errors.append("forecast schema lacks graph_variant")

    existing_results = [str(path.relative_to(ROOT)) for path in FORBIDDEN_RESULT_FILES if path.exists()]
    checkpoint_dir = RESULTS / "checkpoints"
    if checkpoint_dir.is_dir() and any(checkpoint_dir.iterdir()):
        existing_results.append(str(checkpoint_dir.relative_to(ROOT)))
    if existing_results:
        errors.append(f"benchmark outputs already exist: {existing_results}")
    manifest_path = RESULTS / "run_manifest.json"
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text())
            runs = manifest.get("runs") if isinstance(manifest, dict) else None
            run_ids = [run.get("run_id") for run in runs if isinstance(run, dict)] if isinstance(runs, list) else []
            if not isinstance(runs, list) or not run_ids or len(run_ids) != len(set(run_ids)):
                errors.append("failed-run manifest is malformed or has duplicate run_id values")
        except (OSError, json.JSONDecodeError) as exc:
            errors.append(f"failed-run manifest cannot be read: {exc}")

    artifacts = {str(path.relative_to(ROOT)): sha256_file(path) for path in SCIENTIFIC_ARTIFACTS if path.is_file()}
    missing_artifacts = [str(path.relative_to(ROOT)) for path in SCIENTIFIC_ARTIFACTS if not path.is_file()]
    if missing_artifacts:
        errors.append(f"scientific artifacts are missing: {missing_artifacts}")

    if errors:
        print("V2 REPRODUCIBILITY CHECK FAILED")
        print("\n".join(f"- {error}" for error in errors))
        return 1
    print("V2 REPRODUCIBILITY CHECK PASSED")
    print(f"git_commit={commit}")
    print("scientific_artifact_sha256=" + json.dumps(artifacts, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
