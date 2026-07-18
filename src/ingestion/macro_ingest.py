"""Ingest observed Eurostat HICP JSON values into canonical v2 macro observations.

The source payload remains archived unchanged. This mapper only decodes Eurostat's
published JSON-stat positions, maps Eurostat country codes to ISO3 identifiers,
and carries through its native monthly index values. Missing JSON-stat positions
are skipped; no missing value is created or filled.
"""

from __future__ import annotations

import argparse
import json
from itertools import product

from src.config import ROOT
from src.ingestion.common import MACRO_FILE, append_rows, load_verified_record, macro_row
from src.validate_v2_inputs import MACRO_COLUMNS

EUROSTAT_TO_ISO3 = {
    "AT": "AUT", "BE": "BEL", "BG": "BGR", "HR": "HRV", "CY": "CYP", "CZ": "CZE",
    "DK": "DNK", "EE": "EST", "FI": "FIN", "FR": "FRA", "DE": "DEU", "EL": "GRC", "GR": "GRC",
    "HU": "HUN", "IE": "IRL", "IT": "ITA", "LV": "LVA", "LT": "LTU", "LU": "LUX",
    "MT": "MLT", "NL": "NLD", "PL": "POL", "PT": "PRT", "RO": "ROU", "SK": "SVK",
    "SI": "SVN", "ES": "ESP", "SE": "SWE",
}


def ordered_categories(dataset: dict, dimension: str) -> list[str]:
    index = dataset["dimension"][dimension]["category"]["index"]
    return [code for code, _ in sorted(index.items(), key=lambda item: item[1])]


def flat_index(indices: list[int], sizes: list[int]) -> int:
    result = 0
    for index, size in zip(indices, sizes):
        result = result * size + index
    return result


def parse_eurostat_hicp(source_id: str) -> list[dict[str, str]]:
    record = load_verified_record(source_id)
    payload = ROOT / record["local_file"]
    dataset = json.loads(payload.read_text())
    required = ["freq", "unit", "coicop", "geo", "time"]
    if dataset.get("id") != required:
        raise ValueError(f"unexpected Eurostat dimension order: {dataset.get('id')}")
    if dataset["dimension"]["freq"]["category"]["index"] != {"M": 0}:
        raise ValueError("Eurostat source is not a monthly payload")
    coicop_index = dataset["dimension"]["coicop"]["category"]["index"]
    variable_by_coicop = {
        "CP00": "HICP_CP00_INDEX",
        "CP045": "HICP_CP045_ENERGY_INDEX",
    }
    if len(coicop_index) != 1 or next(iter(coicop_index)) not in variable_by_coicop:
        raise ValueError("macro ingestion accepts one official HICP component: CP00 or CP045")
    variable = variable_by_coicop[next(iter(coicop_index))]

    units = ordered_categories(dataset, "unit")
    geos = ordered_categories(dataset, "geo")
    periods = ordered_categories(dataset, "time")
    sizes = dataset["size"]
    values = dataset.get("value", {})
    unit_labels = dataset["dimension"]["unit"]["category"].get("label", {})
    rows: list[dict[str, str]] = []
    for unit_index, geo_index, time_index in product(range(len(units)), range(len(geos)), range(len(periods))):
        index = flat_index([0, unit_index, 0, geo_index, time_index], sizes)
        value = values.get(str(index))
        if value is None:
            continue
        geo = geos[geo_index]
        entity_id = EUROSTAT_TO_ISO3.get(geo)
        if entity_id is None:
            raise ValueError(f"no ISO3 mapping configured for Eurostat geography: {geo}")
        period = periods[time_index]
        rows.append(
            macro_row(
                record,
                entity_id=entity_id,
                observation_date=f"{period}-01",
                variable=variable,
                value=value,
                unit=unit_labels.get(units[unit_index], units[unit_index]),
            )
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-identifier", required=True)
    args = parser.parse_args()
    rows = parse_eurostat_hicp(args.source_identifier)
    count = append_rows(
        path=MACRO_FILE,
        columns=MACRO_COLUMNS,
        key_columns=["entity_id", "observation_date", "variable", "source_identifier"],
        rows=rows,
    )
    print(f"Ingested {count:,} observed Eurostat HICP values into {MACRO_FILE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
