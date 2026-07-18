"""Download an unmodified official IMF SDMX/IMTS response for v2.

The caller supplies the official, release-specific IMF query URL because dataflow
keys and authenticated access rules can change. If required by the IMF endpoint,
set IMF_API_TOKEN (or select another environment variable with --token-env).
"""

from __future__ import annotations

import argparse
import os

from src.acquisition.common import V2_RAW, download_raw, write_acquisition_record


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--url", required=True, help="Complete official IMF SDMX or download URL")
    p.add_argument("--source-release-date", required=True, help="Official IMF release/vintage date, YYYY-MM-DD")
    p.add_argument("--source-identifier", required=True, help="Unique, release-specific source ID")
    p.add_argument("--dataset-name", default="IMF International Trade in Goods by partner country (IMTS)")
    p.add_argument("--frequency", choices=["monthly", "quarterly"], default="monthly")
    p.add_argument("--coverage-description", required=True)
    p.add_argument("--release-lag-description", required=True)
    p.add_argument("--license-note", required=True)
    p.add_argument("--token-env", default="IMF_API_TOKEN", help="Optional bearer-token environment variable")
    p.add_argument("--extension", default="csv", choices=["csv", "json", "xml", "zip"])
    return p


def main() -> int:
    args = parser().parse_args()
    token = os.environ.get(args.token_env)
    headers = {"Authorization": f"Bearer {token}"} if token else None
    destination = V2_RAW / "trade" / "imf" / f"{args.source_identifier}.{args.extension}"
    response_metadata = download_raw(url=args.url, destination=destination, headers=headers)
    record = write_acquisition_record(
        source_identifier=args.source_identifier,
        provider="International Monetary Fund",
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
    print(f"Saved unmodified IMF payload: {destination}")
    print(f"Saved acquisition record: {record}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
