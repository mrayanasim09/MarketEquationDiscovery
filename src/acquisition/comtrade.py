"""Download an unmodified official UN Comtrade monthly response for v2.

Requires COMTRADE_API_KEY for authenticated access. The public preview route can
be selected explicitly, but its record limits make it unsuitable for a full panel.
A release date is mandatory and must be taken from an official Comtrade release or
availability record; this script never substitutes its retrieval date.
"""

from __future__ import annotations

import argparse
import os
from urllib.parse import urlencode

from src.acquisition.common import V2_RAW, download_raw, write_acquisition_record

API_ROOT = "https://comtradeapi.un.org/public/v1/preview/C/M/HS"


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--period", required=True, help="Reported month, YYYY-MM")
    p.add_argument("--reporter-code", required=True, help="UN Comtrade reporter code")
    p.add_argument("--partner-code", required=True, help="UN Comtrade partner code")
    p.add_argument("--flow-code", default="X", help="Trade flow code, e.g. X or M")
    p.add_argument("--cmd-code", default="TOTAL", help="Commodity code; TOTAL retains total goods")
    p.add_argument("--source-release-date", required=True, help="Official release/vintage date, YYYY-MM-DD")
    p.add_argument("--source-identifier", required=True, help="Unique, release-specific source ID")
    p.add_argument("--license-note", required=True, help="UN Comtrade usage/licensing note or URL")
    p.add_argument("--release-lag-description", required=True, help="Documented Comtrade availability/release timing")
    p.add_argument("--api-url", default=API_ROOT, help="Official endpoint; use a licensed endpoint where applicable")
    p.add_argument("--api-key-env", default="COMTRADE_API_KEY", help="Environment variable holding an optional API key")
    return p


def main() -> int:
    args = parser().parse_args()
    params = {
        "period": args.period,
        "reporterCode": args.reporter_code,
        "flowCode": args.flow_code,
        "partnerCode": args.partner_code,
        "partner2Code": "0",
        "customsCode": "C00",
        "motCode": "0",
        "cmdCode": args.cmd_code,
        "maxRecords": "500",
    }
    url = f"{args.api_url}?{urlencode(params)}"
    api_key = os.environ.get(args.api_key_env)
    headers = {"Ocp-Apim-Subscription-Key": api_key} if api_key else None
    source_id = args.source_identifier
    destination = V2_RAW / "trade" / "comtrade" / f"{source_id}.json"
    response_metadata = download_raw(url=url, destination=destination, headers=headers)
    record = write_acquisition_record(
        source_identifier=source_id,
        provider="United Nations Statistics Division",
        dataset_name="UN Comtrade monthly merchandise trade",
        frequency="monthly",
        coverage_description=(
            f"reporter={args.reporter_code}; partner={args.partner_code}; "
            f"flow={args.flow_code}; commodity={args.cmd_code}; period={args.period}"
        ),
        release_lag_description=args.release_lag_description,
        license_note=args.license_note,
        source_url=url,
        source_release_date=args.source_release_date,
        local_file=destination,
        response_metadata=response_metadata,
    )
    print(f"Saved unmodified UN Comtrade payload: {destination}")
    print(f"Saved acquisition record: {record}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
