"""Create quarterly inflation features from complete native HICP averages.

Inflation is the annual percent change in the quarterly average index:
100 * (index_q / index_(q-4) - 1). No missing quarter is filled, and a change is
emitted only if both required complete quarterly averages are available.
"""

from __future__ import annotations

import pandas as pd

from src.transform.common import V2_PROCESSED, ensure_output_dir, require_validated_raw

PANEL_FILE = V2_PROCESSED / "quarterly_hicp_panel.csv"
OUT_FILE = V2_PROCESSED / "quarterly_feature_panel.csv"


def yoy_from_complete(panel: pd.DataFrame, index_column: str, complete_column: str) -> pd.Series:
    out = pd.Series(float("nan"), index=panel.index, dtype=float)
    for _, group in panel.groupby("entity_id", sort=False):
        positions = group.index
        periods = pd.PeriodIndex(group.loc[positions, "quarter"], freq="Q")
        prior = group[index_column].shift(4)
        prior_periods = periods - 4
        valid = (
            group[complete_column].astype(bool)
            & group[complete_column].shift(4).fillna(False).astype(bool)
            & pd.Series(periods.to_numpy() == prior_periods.to_numpy() + 4, index=positions)
        )
        out.loc[positions] = (group[index_column] / prior - 1.0).where(valid) * 100.0
    return out


def main() -> int:
    require_validated_raw()
    ensure_output_dir()
    panel = pd.read_csv(PANEL_FILE).sort_values(["entity_id", "quarter"]).copy()
    panel["cpi_yoy"] = yoy_from_complete(panel, "hicp_all_items_quarterly_average", "hicp_all_items_complete")
    panel["energy_cpi_yoy"] = yoy_from_complete(panel, "hicp_energy_quarterly_average", "hicp_energy_complete")
    panel.to_csv(OUT_FILE, index=False)
    print(f"Wrote {OUT_FILE}: {len(panel):,} country-quarter rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
