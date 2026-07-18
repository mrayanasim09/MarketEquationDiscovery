"""Download an unmodified official Eurostat Comext response for v2 trade."""

from __future__ import annotations

import argparse

from src.acquisition.common import V2_RAW, download_raw, write_acquisition_record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True, help="Complete official filtered Comext API query URL")
    parser.add_argument("--source-release-date", required=True, help="Eurostat UPDATE_DATA date, YYYY-MM-DD")
    parser.add_argument("--source-identifier", required=True)
    parser.add_argument("--coverage-description", required=True)
    parser.add_argument("--release-lag-description", required=True)
    parser.add_argument("--license-note", required=True)
    args = parser.parse_args()
    destination = V2_RAW / "trade" / "comext" / f"{args.source_identifier}.json"
    response = download_raw(url=args.url, destination=destination)
    record = write_acquisition_record(
        source_identifier=args.source_identifier,
        provider="Eurostat",
        dataset_name="Comext EU trade since 1988 by HS2-4-6 and CN8 (DS-045409)",
        frequency="monthly",
        coverage_description=args.coverage_description,
        release_lag_description=args.release_lag_description,
        license_note=args.license_note,
        source_url=args.url,
        source_release_date=args.source_release_date,
        local_file=destination,
        response_metadata=response,
    )
    print(f"Saved unmodified Eurostat Comext payload: {destination}")
    print(f"Saved acquisition record: {record}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
