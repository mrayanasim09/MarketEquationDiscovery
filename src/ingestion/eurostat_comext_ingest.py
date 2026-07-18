"""Ingest observed Eurostat Comext monthly export values into canonical v2 trade."""

from __future__ import annotations

import argparse
import json
from itertools import product

from src.config import ROOT
from src.ingestion.common import TRADE_FILE, append_rows, load_verified_record, trade_row
from src.ingestion.macro_ingest import EUROSTAT_TO_ISO3, flat_index, ordered_categories
from src.validate_v2_inputs import TRADE_COLUMNS


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-identifier", required=True)
    args = parser.parse_args()
    record = load_verified_record(args.source_identifier)
    dataset = json.loads((ROOT / record["local_file"]).read_text())
    required = ["freq", "reporter", "partner", "product", "flow", "indicators", "time"]
    if dataset.get("id") != required:
        raise ValueError(f"unexpected Comext dimension order: {dataset.get('id')}")
    dimensions = dataset["dimension"]
    if dimensions["freq"]["category"]["index"] != {"M": 0}:
        raise ValueError("Comext source is not monthly")
    if dimensions["product"]["category"]["index"] != {"TOTAL": 0}:
        raise ValueError("Comext ingestion accepts total-goods product TOTAL only")
    if dimensions["flow"]["category"]["index"] != {"2": 0}:
        raise ValueError("Comext ingestion accepts export flow 2 only")
    if dimensions["indicators"]["category"]["index"] != {"VALUE_IN_EUROS": 0}:
        raise ValueError("Comext ingestion accepts VALUE_IN_EUROS only")
    reporters = ordered_categories(dataset, "reporter")
    partners = ordered_categories(dataset, "partner")
    periods = ordered_categories(dataset, "time")
    values = dataset.get("value", {})
    rows = []
    for reporter_index, partner_index, time_index in product(range(len(reporters)), range(len(partners)), range(len(periods))):
        index = flat_index([0, reporter_index, partner_index, 0, 0, 0, time_index], dataset["size"])
        value = values.get(str(index))
        if value is None:
            continue
        reporter, partner = reporters[reporter_index], partners[partner_index]
        if reporter not in EUROSTAT_TO_ISO3 or partner not in EUROSTAT_TO_ISO3:
            raise ValueError(f"Comext geography has no configured ISO3 mapping: {reporter}, {partner}")
        rows.append(trade_row(record, exporter=EUROSTAT_TO_ISO3[reporter], importer=EUROSTAT_TO_ISO3[partner], observation_period=periods[time_index], trade_value=value, currency_unit="EUR, current"))
    count = append_rows(path=TRADE_FILE, columns=TRADE_COLUMNS, key_columns=["exporter", "importer", "observation_period", "source_identifier"], rows=rows)
    print(f"Ingested {count:,} observed Eurostat Comext monthly export values into {TRADE_FILE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
