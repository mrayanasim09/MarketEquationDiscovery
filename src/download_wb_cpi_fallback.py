"""World Bank annual CPI YoY with lagged quarterly assignment (no look-ahead)."""

from __future__ import annotations

import pandas as pd
import wbgapi as wb


def download_wb_cpi_lagged_quarterly(
    countries: list[str],
    start: int = 2000,
    end: int = 2024,
) -> pd.DataFrame:
    """
    For countries without FRED CPI: use WB annual inflation (FP.CPI.TOTL.ZG).
    Q1-Q3 of year Y use inflation from Y-1; Q4 of year Y uses inflation from Y.
    Avoids broadcasting full-year information into early quarters of the same year.
    """
    df = wb.data.DataFrame(
        "FP.CPI.TOTL.ZG",
        economy=countries,
        time=range(start - 1, end + 1),
        skipBlanks=True,
        labels=False,
    )
    df = df.reset_index().melt(id_vars="economy", var_name="year", value_name="cpi_yoy")
    df["year"] = df["year"].astype(str).str.replace("YR", "", regex=False).astype(int)
    df = df.rename(columns={"economy": "iso3"})

    annual = df.pivot(index="iso3", columns="year", values="cpi_yoy")
    rows = []
    for iso3 in countries:
        for year in range(start, end + 1):
            lag_val = annual.loc[iso3, year - 1] if year - 1 in annual.columns else None
            cur_val = annual.loc[iso3, year] if year in annual.columns else None
            for q in (1, 2, 3):
                rows.append(
                    {
                        "iso3": iso3,
                        "quarter": f"{year}Q{q}",
                        "cpi_yoy": lag_val,
                        "cpi_index": None,
                        "source": "wb_lagged_yoy",
                    }
                )
            rows.append(
                {
                    "iso3": iso3,
                    "quarter": f"{year}Q4",
                    "cpi_yoy": cur_val,
                    "cpi_index": None,
                    "source": "wb_lagged_yoy",
                }
            )
    return pd.DataFrame(rows)
