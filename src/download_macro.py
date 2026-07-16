"""Download macro panel: FRED CPI, WB lagged fallback, World Bank controls, FRED energy."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path

import pandas as pd
import requests
import wbgapi as wb
import yaml

from src.config import DATA_RAW, ROOT, ensure_dirs, load_protocol
from src.download_wb_cpi_fallback import download_wb_cpi_lagged_quarterly

FRED_GRAPH = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
FRED_BRENT = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DCOILBRENTEU"

WB_INDICATORS = {
    "gdp_yoy": "NY.GDP.MKTP.KD.ZG",
    "policy_rate": "FR.INR.RINR",
    "neer_level": "PA.NUS.FCRF",
}


def load_cpi_config() -> tuple[dict[str, dict], list[str]]:
    path = ROOT / "config" / "fred_cpi_series.yaml"
    with path.open() as f:
        cfg = yaml.safe_load(f)
    return cfg.get("fred", {}), cfg.get("wb_lagged_yoy", [])


def download_fred_series(series_id: str) -> pd.DataFrame:
    url = FRED_GRAPH.format(series_id=series_id)
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    df = pd.read_csv(StringIO(resp.text))
    date_col = "observation_date" if "observation_date" in df.columns else df.columns[0]
    value_col = series_id if series_id in df.columns else df.columns[1]
    df = df.rename(columns={date_col: "date", value_col: "cpi_index"})
    df["date"] = pd.to_datetime(df["date"])
    df["cpi_index"] = pd.to_numeric(df["cpi_index"], errors="coerce")
    return df.dropna(subset=["date", "cpi_index"])


def load_cached_csv(path: Path, label: str) -> pd.DataFrame | None:
    if path.exists() and path.stat().st_size > 100:
        print(f"  using cached {label}: {path}")
        return pd.read_csv(path)
    return None


def download_fred_cpi(fred_map: dict[str, dict]) -> pd.DataFrame:
    records = []
    for iso3, meta in fred_map.items():
        series_id = meta["id"]
        freq = meta.get("freq", "M")
        try:
            series = download_fred_series(series_id)
            series["iso3"] = iso3
            series["source"] = "fred"
            series["freq"] = freq
            records.append(series[["iso3", "date", "cpi_index", "source", "freq"]])
            print(f"  FRED CPI {iso3}: {len(series)} obs ({series_id}, {freq})")
        except (requests.RequestException, ValueError, KeyError) as exc:
            print(f"  warning: FRED CPI failed for {iso3} ({series_id}): {exc}")
    if not records:
        return pd.DataFrame(columns=["iso3", "date", "cpi_index", "source", "freq"])
    return pd.concat(records, ignore_index=True)


def index_to_quarterly_yoy(cpi: pd.DataFrame) -> pd.DataFrame:
    """Convert monthly or quarterly CPI index series to quarterly YoY."""
    rows = []
    for iso3, grp in cpi.groupby("iso3"):
        freq = grp["freq"].iloc[0]
        work = grp.sort_values("date").copy()
        work["date"] = pd.to_datetime(work["date"])
        if freq == "Q":
            work["quarter"] = work["date"].dt.to_period("Q")
            q_index = work.groupby("quarter", as_index=False)["cpi_index"].last()
        else:
            work["quarter"] = work["date"].dt.to_period("Q")
            q_index = work.groupby("quarter", as_index=False)["cpi_index"].mean()
        q_index["quarter"] = q_index["quarter"].astype(str)
        q_index["iso3"] = iso3
        q_index = q_index.sort_values("quarter")
        q_index["cpi_yoy"] = q_index["cpi_index"].pct_change(4) * 100
        rows.append(q_index)
    if not rows:
        return pd.DataFrame(columns=["iso3", "quarter", "cpi_index", "cpi_yoy"])
    return pd.concat(rows, ignore_index=True)


def download_world_bank(countries: list[str], start: int = 2000, end: int = 2024) -> pd.DataFrame:
    frames = []
    for col, indicator in WB_INDICATORS.items():
        df = wb.data.DataFrame(
            indicator,
            economy=countries,
            time=range(start, end + 1),
            skipBlanks=True,
            labels=False,
        )
        df = df.reset_index().melt(id_vars="economy", var_name="year", value_name=col)
        df["year"] = df["year"].astype(str).str.replace("YR", "", regex=False).astype(int)
        frames.append(df.set_index(["economy", "year"]))

    out = pd.concat(frames, axis=1).reset_index()
    return out.rename(columns={"economy": "iso3"})


def download_energy_brent() -> pd.DataFrame:
    cached = load_cached_csv(DATA_RAW / "fred_brent_monthly.csv", "Brent")
    if cached is not None and "brent_usd" in cached.columns:
        cached["date"] = pd.to_datetime(cached["date"])
        return cached.dropna(subset=["date", "brent_usd"])
    resp = requests.get(FRED_BRENT, timeout=60)
    resp.raise_for_status()
    df = pd.read_csv(StringIO(resp.text))
    df = df.rename(columns={"observation_date": "date", "DCOILBRENTEU": "brent_usd"})
    df["date"] = pd.to_datetime(df["date"])
    df["brent_usd"] = pd.to_numeric(df["brent_usd"], errors="coerce")
    return df.dropna()


def annual_controls_to_quarterly(wb_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in wb_df.iterrows():
        for q in (1, 2, 3, 4):
            rows.append(
                {
                    "iso3": row["iso3"],
                    "quarter": f"{row['year']}Q{q}",
                    "year": row["year"],
                    "gdp_yoy": row["gdp_yoy"],
                    "policy_rate": row["policy_rate"],
                    "neer_level": row["neer_level"],
                }
            )
    panel = pd.DataFrame(rows)
    panel["neer_chg"] = panel.groupby("iso3")["neer_level"].pct_change(4) * 100
    return panel


def build_quarterly_panel(
    cpi_q: pd.DataFrame,
    wb_q: pd.DataFrame,
    energy_m: pd.DataFrame,
    policy_ffill_limit: int = 2,
) -> pd.DataFrame:
    panel = cpi_q.merge(wb_q, on=["iso3", "quarter"], how="outer")

    energy_m = energy_m.copy()
    energy_m["quarter"] = energy_m["date"].dt.to_period("Q").astype(str)
    energy_q = energy_m.groupby("quarter")["brent_usd"].mean().reset_index()
    energy_q["energy_idx"] = energy_q["brent_usd"].pct_change(4) * 100
    panel = panel.merge(energy_q[["quarter", "energy_idx"]], on="quarter", how="left")

    panel["covid"] = panel["quarter"].apply(
        lambda q: 1 if q.startswith(("2020Q", "2021Q")) else 0
    )
    panel = panel.sort_values(["iso3", "quarter"])
    panel["policy_rate"] = panel.groupby("iso3")["policy_rate"].ffill(limit=policy_ffill_limit)
    panel["quarter_period"] = panel["quarter"].apply(lambda x: pd.Period(x, freq="Q"))
    return panel.sort_values(["iso3", "quarter_period"])


def main() -> None:
    ensure_dirs()
    protocol = load_protocol()
    countries = protocol["countries"]
    fred_map, wb_lagged = load_cpi_config()

    print("Downloading FRED CPI indices...")
    fred_path = DATA_RAW / "fred_cpi_monthly.csv"
    cached_fred = load_cached_csv(fred_path, "FRED CPI")
    if cached_fred is not None and len(cached_fred) > 0 and "freq" not in cached_fred.columns:
        # Re-fetch if old cache lacks freq column
        cached_fred = None
    if cached_fred is not None and len(cached_fred) > 0:
        fred_cpi = cached_fred
    else:
        fred_cpi = download_fred_cpi(fred_map)
        if len(fred_cpi) > 0:
            fred_cpi.to_csv(fred_path, index=False)
        elif fred_path.exists() and fred_path.stat().st_size > 100:
            print("  warning: FRED download empty; keeping existing cache")
            fred_cpi = pd.read_csv(fred_path)
    print(f"  using {fred_path} ({len(fred_cpi)} rows)")

    print("Transforming FRED CPI to quarterly YoY...")
    cpi_q_fred = index_to_quarterly_yoy(fred_cpi)
    cpi_q_fred["source"] = "fred"

    cpi_parts = [cpi_q_fred[["iso3", "quarter", "cpi_index", "cpi_yoy", "source"]]]

    if wb_lagged:
        print(f"Downloading WB lagged CPI fallback for {wb_lagged}...")
        wb_cpi_q = download_wb_cpi_lagged_quarterly(wb_lagged)
        wb_path = DATA_RAW / "wb_cpi_lagged_quarterly.csv"
        wb_cpi_q.to_csv(wb_path, index=False)
        print(f"  saved {wb_path} ({len(wb_cpi_q)} rows)")
        cpi_parts.append(wb_cpi_q[["iso3", "quarter", "cpi_index", "cpi_yoy", "source"]])

    cpi_q = pd.concat(cpi_parts, ignore_index=True)
    cpi_q = cpi_q[cpi_q["iso3"].isin(countries)]
    cpi_q = cpi_q[(cpi_q["quarter"] >= "2000Q1") & (cpi_q["quarter"] <= "2024Q4")]

    print("Downloading World Bank controls (GDP, NEER, policy rate)...")
    wb_df = download_world_bank(countries)
    wb_path = DATA_RAW / "world_bank_annual.csv"
    wb_df.to_csv(wb_path, index=False)
    print(f"  saved {wb_path} ({len(wb_df)} rows)")

    print("Downloading Brent oil (FRED)...")
    energy = download_energy_brent()
    energy_path = DATA_RAW / "fred_brent_monthly.csv"
    energy.to_csv(energy_path, index=False)
    print(f"  saved {energy_path} ({len(energy)} rows)")

    wb_q = annual_controls_to_quarterly(wb_df)
    panel = build_quarterly_panel(cpi_q, wb_q, energy, policy_ffill_limit=2)
    macro_path = DATA_RAW / "macro_quarterly_panel.csv"
    panel.to_csv(macro_path, index=False)
    print(f"  saved {macro_path} ({len(panel)} rows)")

    cpi_sources = cpi_q.groupby("iso3")["source"].first().to_dict()
    manifest = {
        "downloaded_at": datetime.now(timezone.utc).isoformat(),
        "protocol_version": "1.1",
        "sources": {
            "cpi": "FRED index (primary); WB annual YoY lagged for VNM/SGP",
            "controls": "World Bank WDI annual",
            "energy": "FRED Brent (DCOILBRENTEU)",
        },
        "cpi_source_by_country": cpi_sources,
        "wb_lagged_yoy_countries": wb_lagged,
        "countries": countries,
        "policy_rate_missing_pct": round(float(panel["policy_rate"].isna().mean()) * 100, 1),
        "policy_rate_ffill_limit_quarters": 2,
        "files": {
            "fred_cpi_monthly": str(fred_path),
            "macro_quarterly_panel": str(macro_path),
        },
        "row_counts": {
            "fred_cpi_monthly": len(fred_cpi),
            "macro_quarterly_panel": len(panel),
        },
    }
    manifest_path = DATA_RAW / "macro_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"  saved {manifest_path}")


if __name__ == "__main__":
    main()
