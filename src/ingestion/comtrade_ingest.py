"""Ingest observed export rows from an archived UN Comtrade JSON response.

The input must already be an archived, hash-verified acquisition payload. This
module accepts exports only because the canonical schema defines exporter/importer;
imports are not silently relabeled. It performs no partner aggregation or period
conversion.
"""

from __future__ import annotations

import argparse
import json

from src.config import ROOT
from src.ingestion.common import TRADE_FILE, append_rows, load_verified_record, trade_row
from src.validate_v2_inputs import TRADE_COLUMNS


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-identifier", required=True)
    parser.add_argument("--exporter-field", default="reporterISO")
    parser.add_argument("--importer-field", default="partnerISO")
    parser.add_argument("--period-field", default="period")
    parser.add_argument("--value-field", default="primaryValue")
    parser.add_argument("--flow-field", default="flowCode")
    parser.add_argument("--currency-unit", required=True)
    args = parser.parse_args()
    record = load_verified_record(args.source_identifier)
    payload = json.loads((ROOT / record["local_file"]).read_text())
    observations = payload.get("data")
    if not isinstance(observations, list):
        raise ValueError("UN Comtrade payload has no data array")
    rows = []
    for observation in observations:
        if observation.get(args.flow_field) != "X":
            raise ValueError("UN Comtrade ingestion accepts export flow X only")
        fields = [args.exporter_field, args.importer_field, args.period_field, args.value_field]
        if any(observation.get(field) in (None, "") for field in fields):
            raise ValueError(f"UN Comtrade observation missing one of {fields}")
        rows.append(trade_row(record, exporter=str(observation[args.exporter_field]), importer=str(observation[args.importer_field]), observation_period=str(observation[args.period_field]), trade_value=observation[args.value_field], currency_unit=args.currency_unit))
    count = append_rows(path=TRADE_FILE, columns=TRADE_COLUMNS, key_columns=["exporter", "importer", "observation_period", "source_identifier"], rows=rows)
    print(f"Ingested {count:,} observed UN Comtrade export values into {TRADE_FILE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
