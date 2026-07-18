"""Download an unmodified official monthly/quarterly macro source response for v2.

Use with an exact official OECD, IMF, national-statistical-office, or other
admissible HTTPS query URL. This module intentionally does not parse SDMX, create
CPI inflation, average monthly values, or populate the canonical observation CSV.
It only archives the bytes and writes provenance for later controlled ingestion.
"""

from __future__ import annotations

import argparse
import os

from src.acquisition.common import V2_RAW, download_raw, write_acquisition_record


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--url", required=True, help="Complete official source/API query URL")
    p.add_argument("--provider", required=True, help="Official provider, e.g. OECD or national statistical office")
    p.add_argument("--dataset-name", required=True, help="Official dataset/table and variable definition")
    p.add_argument("--frequency", choices=["monthly", "quarterly"], required=True)
    p.add_argument("--coverage-description", required=True, help="Entities, variables, and reference periods requested")
    p.add_argument("--release-lag-description", required=True, help="Documented source release schedule or lag")
    p.add_argument("--license-note", required=True, help="Dataset licence/terms URL or short statement")
    p.add_argument("--source-release-date", required=True, help="Official release/vintage date, YYYY-MM-DD")
    p.add_argument("--source-identifier", required=True, help="Unique, release-specific source ID")
    p.add_argument("--extension", default="csv", choices=["csv", "json", "xml", "xlsx", "zip"])
    p.add_argument("--token-env", help="Optional environment variable holding a bearer token")
    return p


def main() -> int:
    args = parser().parse_args()
    token = os.environ.get(args.token_env) if args.token_env else None
    headers = {"Authorization": f"Bearer {token}"} if token else None
    destination = V2_RAW / "macro" / "official" / f"{args.source_identifier}.{args.extension}"
    response_metadata = download_raw(url=args.url, destination=destination, headers=headers)
    record = write_acquisition_record(
        source_identifier=args.source_identifier,
        provider=args.provider,
        dataset_name=args.dataset_name,
        frequency=args.frequency,
        coverage_description=args.coverage_description,
        release_lag_description=args.release_lag_description,
        license_note=args.license_note,
        source_url=args.url,
        source_release_date=args.source_release_date,
        local_file=destination,
        response_metadata=response_metadata,
    )
    print(f"Saved unmodified official macro payload: {destination}")
    print(f"Saved acquisition record: {record}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
