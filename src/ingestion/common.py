"""Common guards for v2 canonical raw-observation ingestion."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable

from src.acquisition.common import ACQUISITION_RECORDS_DIR, V2_RAW, sha256
from src.config import ROOT
from src.validate_v2_inputs import MACRO_COLUMNS, TRADE_COLUMNS

MACRO_FILE = V2_RAW / "macro" / "macro_observations.csv"
TRADE_FILE = V2_RAW / "trade" / "trade_observations.csv"


def load_verified_record(source_identifier: str) -> dict[str, Any]:
    """Load a sidecar only when its recorded payload hash still matches."""
    record_path = ACQUISITION_RECORDS_DIR / f"{source_identifier}.json"
    if not record_path.is_file():
        raise FileNotFoundError(f"missing acquisition record: {record_path}")
    record = json.loads(record_path.read_text())
    if record.get("source_identifier") != source_identifier:
        raise ValueError(f"source identifier does not match sidecar: {record_path}")
    local_file = ROOT / str(record.get("local_file", ""))
    if not local_file.is_file():
        raise FileNotFoundError(f"archived source payload is missing: {local_file}")
    if sha256(local_file) != record.get("sha256"):
        raise ValueError(f"archived source payload hash does not match sidecar: {local_file}")
    for field in ("source_release_date", "source_url", "retrieved_at", "frequency"):
        if not str(record.get(field, "")).strip():
            raise ValueError(f"acquisition record has no {field}: {record_path}")
    return record


def existing_keys(path: Path, columns: list[str], key_columns: list[str]) -> set[tuple[str, ...]]:
    if not path.is_file():
        raise FileNotFoundError(f"canonical raw table is missing: {path}")
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != columns:
            raise ValueError(f"canonical raw table schema differs from documented contract: {path}")
        return {tuple(row[column] for column in key_columns) for row in reader}


def append_rows(
    *,
    path: Path,
    columns: list[str],
    key_columns: list[str],
    rows: Iterable[dict[str, str]],
) -> int:
    """Append strictly schema-conformant rows and reject duplicate source observations."""
    known = existing_keys(path, columns, key_columns)
    prepared = list(rows)
    if not prepared:
        raise ValueError("ingestion yielded no observed source rows")
    for row in prepared:
        if list(row) != columns:
            raise ValueError("ingestion row does not match the canonical schema and column order")
        key = tuple(row[column] for column in key_columns)
        if key in known:
            raise ValueError(f"duplicate canonical observation would be created: {key}")
        known.add(key)
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writerows(prepared)
    return len(prepared)


def macro_row(record: dict[str, Any], *, entity_id: str, observation_date: str, variable: str, value: object, unit: str) -> dict[str, str]:
    return {
        "entity_id": entity_id,
        "observation_date": str(observation_date),
        "variable": variable,
        "value": str(value),
        "unit": unit,
        "frequency": str(record["frequency"]),
        "source_release_date": str(record["source_release_date"]),
        "source_identifier": str(record["source_identifier"]),
        "source_url": str(record["source_url"]),
        "retrieved_at": str(record["retrieved_at"]),
        "observation_status": "observed",
        "missing_reason": "",
    }


def trade_row(record: dict[str, Any], *, exporter: str, importer: str, observation_period: str, trade_value: object, currency_unit: str) -> dict[str, str]:
    return {
        "exporter": exporter,
        "importer": importer,
        "observation_period": str(observation_period),
        "trade_value": str(trade_value),
        "currency_unit": currency_unit,
        "frequency": str(record["frequency"]),
        "source_release_date": str(record["source_release_date"]),
        "source_identifier": str(record["source_identifier"]),
        "source_url": str(record["source_url"]),
        "retrieved_at": str(record["retrieved_at"]),
        "observation_status": "observed",
        "missing_reason": "",
    }
