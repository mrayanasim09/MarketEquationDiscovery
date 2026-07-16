"""Validation gates for macro_quarterly_panel (CPI frequency, no look-ahead)."""

from __future__ import annotations

import json
import sys

import pandas as pd

from src.config import DATA_RAW


def validate_cpi_quarterly_variance(panel: pd.DataFrame) -> list[str]:
    errors = []
    for iso3, grp in panel.groupby("iso3"):
        grp = grp.dropna(subset=["cpi_yoy"])
        for year, ygrp in grp.groupby(grp["quarter"].str[:4]):
            if len(ygrp) >= 4:
                std = ygrp["cpi_yoy"].std()
                if std is not None and std < 1e-6:
                    errors.append(f"{iso3} {year}: cpi_yoy flat across quarters (std={std})")
    return errors


def validate_usa_2022(panel: pd.DataFrame) -> list[str]:
    errors = []
    usa = panel[(panel["iso3"] == "USA") & (panel["quarter"].str.startswith("2022"))]
    if usa["cpi_yoy"].notna().sum() == 0:
        errors.append("USA 2022: no CPI data — re-run download_macro when FRED is reachable")
        return errors
    if len(usa) < 4:
        errors.append("USA 2022: fewer than 4 quarters present")
        return errors
    vals = usa.set_index("quarter")["cpi_yoy"].dropna()
    if vals.nunique() <= 1:
        errors.append(f"USA 2022: cpi_yoy not varying across quarters: {vals.to_dict()}")
    return errors


def validate_no_annual_broadcast(panel: pd.DataFrame, exclude_iso3: set[str] | None = None) -> list[str]:
    """Detect identical cpi_yoy for all four quarters (annual broadcast signature)."""
    exclude_iso3 = exclude_iso3 or set()
    errors = []
    for iso3, grp in panel.groupby("iso3"):
        if iso3 in exclude_iso3:
            continue
        grp = grp.dropna(subset=["cpi_yoy"])
        for year, ygrp in grp.groupby(grp["quarter"].str[:4]):
            if len(ygrp) == 4:
                unique = ygrp["cpi_yoy"].nunique()
                if unique == 1:
                    errors.append(
                        f"{iso3} {year}: all 4 quarters identical cpi_yoy={ygrp['cpi_yoy'].iloc[0]:.4f}"
                    )
    return errors


def main() -> int:
    path = DATA_RAW / "macro_quarterly_panel.csv"
    if not path.exists():
        print(f"ERROR: {path} not found. Run python -m src.download_macro first.")
        return 1

    panel = pd.read_csv(path)
    all_errors: list[str] = []

    print("Validating CPI quarterly variance...")
    all_errors.extend(validate_cpi_quarterly_variance(panel))

    print("Spot-check USA 2022...")
    all_errors.extend(validate_usa_2022(panel))

    print("Checking for annual broadcast pattern...")
    wb_lagged = {"VNM", "SGP"}
    broadcast = validate_no_annual_broadcast(panel, exclude_iso3=wb_lagged)
    if len(broadcast) > len(panel["iso3"].unique()) * 2:
        all_errors.extend(broadcast[:10])
        all_errors.append(f"... and {len(broadcast) - 10} more broadcast-like years")

    report = {
        "passed": len(all_errors) == 0,
        "error_count": len(all_errors),
        "errors": all_errors[:20],
        "countries": int(panel["iso3"].nunique()),
        "rows": len(panel),
        "cpi_missing_pct": round(float(panel["cpi_yoy"].isna().mean()) * 100, 1),
    }
    report_path = DATA_RAW / "macro_validation.json"
    report_path.write_text(json.dumps(report, indent=2))

    if all_errors:
        print(f"FAILED: {len(all_errors)} validation issue(s)")
        for err in all_errors[:10]:
            print(f"  - {err}")
        return 1

    print("PASSED: macro panel validation")
    print(f"  countries={report['countries']} rows={report['rows']} cpi_missing={report['cpi_missing_pct']}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
