"""Build v2 raw-manifest and source-registry records from acquisition sidecars.

This command does not populate macro_observations.csv or trade_observations.csv;
those canonical files require a separately reviewed, source-specific ingestion
step. It refuses to include altered or missing archived source payloads.
"""

from __future__ import annotations

import argparse
import csv
import json
from typing import Any

from src.acquisition.common import ACQUISITION_RECORDS_DIR, V2_RAW, sha256
from src.validate_v2_inputs import REGISTRY_COLUMNS

MANIFEST_PATH = V2_RAW / "raw_manifest.json"
REGISTRY_PATH = V2_RAW / "metadata" / "source_registry.csv"
REQUIRED_RECORD_FIELDS = {
    "source_identifier", "provider", "dataset_name", "frequency", "coverage_description",
    "release_lag_description", "license_note", "source_url", "source_release_date",
    "retrieved_at", "local_file", "sha256",
}


def load_records() -> list[dict[str, Any]]:
    if not ACQUISITION_RECORDS_DIR.is_dir():
        raise FileNotFoundError(f"no acquisition records found: {ACQUISITION_RECORDS_DIR}")
    records: list[dict[str, Any]] = []
    for path in sorted(ACQUISITION_RECORDS_DIR.glob("*.json")):
        record = json.loads(path.read_text())
        missing = sorted(REQUIRED_RECORD_FIELDS - set(record))
        if missing:
            raise ValueError(f"{path} missing required fields: {missing}")
        local = V2_RAW.parents[2] / record["local_file"]
        if not local.is_file():
            raise FileNotFoundError(f"source payload recorded by {path} is missing: {record['local_file']}")
        if sha256(local) != record["sha256"]:
            raise ValueError(f"source payload hash changed after acquisition: {record['local_file']}")
        records.append(record)
    if not records:
        raise ValueError("no acquisition record JSON files found")
    identifiers = [record["source_identifier"] for record in records]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("acquisition records contain duplicate source_identifier values")
    return records


def write_registry(records: list[dict[str, Any]]) -> None:
    REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with REGISTRY_PATH.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=REGISTRY_COLUMNS)
        writer.writeheader()
        for record in records:
            writer.writerow({column: record[column] for column in REGISTRY_COLUMNS})


def write_manifest(records: list[dict[str, Any]]) -> None:
    downloads = [
        {
            "source_identifier": record["source_identifier"],
            "source_url": record["source_url"],
            "retrieved_at": record["retrieved_at"],
            "source_release_date": record["source_release_date"],
            "sha256": record["sha256"],
            "local_file": record["local_file"],
            "license_note": record["license_note"],
        }
        for record in records
    ]
    MANIFEST_PATH.write_text(json.dumps({"schema_version": "2.0", "status": "acquired_raw_payloads", "downloads": downloads}, indent=2) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Verify sidecars and hashes without writing registry/manifest")
    args = parser.parse_args()
    try:
        records = load_records()
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
        print(f"ACQUISITION MANIFEST BUILD FAILED: {exc}")
        return 1
    if args.dry_run:
        print(f"Verified {len(records)} acquisition records and archived source payload hashes")
        return 0
    write_registry(records)
    write_manifest(records)
    print(f"Wrote {len(records)} registered sources to {REGISTRY_PATH}")
    print(f"Wrote {len(records)} verified downloads to {MANIFEST_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
