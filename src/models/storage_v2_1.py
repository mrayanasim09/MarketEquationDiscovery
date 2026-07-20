"""v2.1 transactional result schemas; isolated from archived v2 storage.

Defines the canonical column layouts for forecast, metric, and
Diebold-Mariano test parquet files. All writers enforce schema
completeness, key uniqueness, and provenance non-nullity before
writing to disk.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "experiments/results/v2_1"
PROVENANCE = ["run_id", "git_commit", "dataset_version", "configuration_id", "execution_timestamp"]
FORECAST_COLUMNS = PROVENANCE + [
    "country", "forecast_origin", "target_quarter", "horizon", "model", "graph_variant", "seed",
    "mean", "lower_80", "upper_80", "lower_95", "upper_95", "actual",
]
METRIC_COLUMNS = PROVENANCE + ["model", "graph_variant", "seed", "horizon", "metric", "value", "origin_count"]
DM_COLUMNS = PROVENANCE + [
    "model", "graph_variant", "seed", "horizon", "comparator", "loss", "dm_stat", "p_value",
    "p_value_bh", "loss_difference", "ci_low", "ci_high", "origin_count",
]


def write_parquet_exact(path: Path, frame: pd.DataFrame, columns: list[str], keys: list[str]) -> None:
    """Write *frame* to *path* as parquet, enforcing schema and key constraints.

    Raises ``ValueError`` if required columns are missing, transaction keys
    contain duplicates, or provenance fields contain null values.
    """
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{path.name} lacks required columns: {missing}")
    selected = frame.loc[:, columns].copy()
    if selected.duplicated(keys).any():
        raise ValueError(f"{path.name} has duplicate transaction keys: {keys}")
    if selected[PROVENANCE].isna().any().any():
        raise ValueError(f"{path.name} has missing provenance")
    path.parent.mkdir(parents=True, exist_ok=True)
    selected.to_parquet(path, index=False)


def append_failure_record(record: dict[str, Any]) -> None:
    """Append a JSON failure record to the failed transactions log."""
    RESULTS.mkdir(parents=True, exist_ok=True)
    with (RESULTS / "failed_transactions.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")
