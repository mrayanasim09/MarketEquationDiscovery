"""Ingest observed IMF IMTS rows from a hash-verified archived CSV response.

Column names are explicit CLI arguments because IMF SDMX download layouts depend
on the selected dataflow. The caller must select a source slice containing only
exports if mapping reporter/partner directly into exporter/importer.
"""

from __future__ import annotations

import argparse
import csv

from src.config import ROOT
from src.ingestion.common import TRADE_FILE, append_rows, load_verified_record, trade_row
from src.validate_v2_inputs import TRADE_COLUMNS


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-identifier", required=True)
    parser.add_argument("--exporter-column", required=True)
    parser.add_argument("--importer-column", required=True)
    parser.add_argument("--period-column", required=True)
    parser.add_argument("--value-column", required=True)
    parser.add_argument("--currency-unit", required=True)
    args = parser.parse_args()
    record = load_verified_record(args.source_identifier)
    payload_path = ROOT / record["local_file"]
    with payload_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        required = {args.exporter_column, args.importer_column, args.period_column, args.value_column}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError(f"IMF payload does not contain required CSV columns: {sorted(required)}")
        rows = []
        for observation in reader:
            if any(not observation[column] for column in required):
                raise ValueError("IMF payload contains a missing required source value; do not fill it")
            rows.append(trade_row(record, exporter=observation[args.exporter_column], importer=observation[args.importer_column], observation_period=observation[args.period_column], trade_value=observation[args.value_column], currency_unit=args.currency_unit))
    count = append_rows(path=TRADE_FILE, columns=TRADE_COLUMNS, key_columns=["exporter", "importer", "observation_period", "source_identifier"], rows=rows)
    print(f"Ingested {count:,} observed IMF IMTS values into {TRADE_FILE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
