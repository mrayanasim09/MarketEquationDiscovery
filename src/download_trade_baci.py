"""Process CEPII BACI bulk CSV into bilateral trade edges for the graph."""

from __future__ import annotations

import argparse
import json
import re
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.config import DATA_RAW, ROOT, ensure_dirs, load_protocol

BULK_DIR = DATA_RAW / "bulk"
BACI_CANDIDATES = [
    BULK_DIR / "baci_latest.csv",
    BULK_DIR / "baci_latest.zip",
    BULK_DIR / "baci.csv",
    BULK_DIR / "baci.zip",
]

CEPII_URLS = [
    "https://www.cepii.fr/DATA_DOWNLOAD/baci/baci_202501.zip",
    "https://www.cepii.fr/DATA_DOWNLOAD/baci/baci_202401.zip",
    "https://www.cepii.fr/DATA_DOWNLOAD/baci/baci_202301.zip",
]

YEARLY_FILE_PATTERN = re.compile(r"BACI_HS\d+_Y(\d{4})_V\d+\.csv$", re.I)

INSTRUCTIONS = """
CEPII BACI bulk file required.

Option A — yearly folder (recommended):
  Place CEPII folder e.g. BACI_HS96_V202601 at project root or data/raw/bulk/

Option B — single file:
  data/raw/bulk/baci_latest.csv or baci_latest.zip

Then run: python -m src.download_trade_baci
"""


def find_baci_folder() -> Path | None:
    search_roots = [ROOT, BULK_DIR, DATA_RAW / "bulk"]
    for base in search_roots:
        if not base.exists():
            continue
        for path in sorted(base.iterdir()):
            if path.is_dir() and path.name.upper().startswith("BACI_HS"):
                if list(path.glob("BACI_HS*_Y*.csv")):
                    return path
    return None


def find_baci_file() -> Path | None:
    for path in BACI_CANDIDATES:
        if path.exists():
            return path
    if BULK_DIR.exists():
        for path in sorted(BULK_DIR.glob("baci*")):
            if path.suffix in (".csv", ".zip"):
                return path
    return None


def find_country_codes_file(baci_folder: Path | None) -> Path | None:
    if baci_folder is None:
        return None
    candidates = list(baci_folder.glob("country_codes*.csv"))
    return candidates[0] if candidates else None


def load_baci_mapping(country_codes_path: Path | None, countries: list[str]) -> dict[int, str]:
    """Map CEPII country_code (i/j in BACI) -> ISO3."""
    if country_codes_path and country_codes_path.exists():
        df = pd.read_csv(country_codes_path)
        code_col = "country_code" if "country_code" in df.columns else df.columns[0]
        iso_col = "country_iso3" if "country_iso3" in df.columns else "iso3"
        df = df[[code_col, iso_col]].dropna()
        df[code_col] = df[code_col].astype(int)
        mapping = dict(zip(df[code_col], df[iso_col]))
        return {k: v for k, v in mapping.items() if v in countries}

    # Fallback: config/baci_iso_mapping.csv (ISO numeric — legacy)
    mapping_path = ROOT / "config" / "baci_iso_mapping.csv"
    df = pd.read_csv(mapping_path)
    return dict(zip(df["iso_num"].astype(int), df["iso3"]))


def panel_baci_codes(iso_map: dict[int, str], countries: list[str]) -> set[int]:
    country_set = set(countries)
    return {code for code, iso3 in iso_map.items() if iso3 in country_set}


def read_baci(path: Path) -> pd.DataFrame:
    if path.suffix == ".zip":
        with zipfile.ZipFile(path) as zf:
            csv_names = [n for n in zf.namelist() if n.endswith(".csv") and "baci" in n.lower()]
            if not csv_names:
                csv_names = [n for n in zf.namelist() if n.endswith(".csv")]
            if not csv_names:
                raise ValueError(f"No CSV inside zip: {path}")
            with zf.open(csv_names[0]) as f:
                return pd.read_csv(f)
    return pd.read_csv(path)


def normalize_baci_columns(df: pd.DataFrame) -> pd.DataFrame:
    lower = {c.lower(): c for c in df.columns}

    def pick(*names: str) -> str | None:
        for n in names:
            if n in lower:
                return lower[n]
        return None

    year_col = pick("t", "year")
    exp_col = pick("i", "exporter", "exporter_code")
    imp_col = pick("j", "importer", "importer_code")
    val_col = pick("v", "value", "trade_usd")

    if not all([year_col, exp_col, imp_col, val_col]):
        raise ValueError(f"Unrecognized BACI columns: {list(df.columns)}")

    out = df[[year_col, exp_col, imp_col, val_col]].copy()
    out.columns = ["year", "exporter_num", "importer_num", "trade_usd"]
    out["year"] = pd.to_numeric(out["year"], errors="coerce").astype("Int64")
    out["trade_usd"] = pd.to_numeric(out["trade_usd"], errors="coerce")
    return out.dropna(subset=["year", "exporter_num", "importer_num", "trade_usd"])


def process_yearly_file(
    path: Path,
    year: int,
    panel_codes: set[int],
    iso_map: dict[int, str],
) -> pd.DataFrame:
    """Read one BACI year file; keep only panel countries; sum across products."""
    chunks = []
    for chunk in pd.read_csv(path, usecols=["t", "i", "j", "v"], chunksize=2_000_000):
        chunk = chunk[(chunk["i"].isin(panel_codes)) & (chunk["j"].isin(panel_codes))]
        if chunk.empty:
            continue
        agg = chunk.groupby(["i", "j"], as_index=False)["v"].sum()
        chunks.append(agg)

    if not chunks:
        return pd.DataFrame(columns=["year", "exporter", "importer", "trade_usd"])

    year_df = pd.concat(chunks, ignore_index=True)
    year_df = year_df.groupby(["i", "j"], as_index=False)["v"].sum()
    year_df = year_df.rename(columns={"i": "exporter_num", "j": "importer_num", "v": "trade_usd"})
    year_df["year"] = year
    year_df["exporter"] = year_df["exporter_num"].map(iso_map)
    year_df["importer"] = year_df["importer_num"].map(iso_map)
    return year_df.dropna(subset=["exporter", "importer"])


def load_baci_from_folder(
    folder: Path,
    iso_map: dict[int, str],
    countries: list[str],
    start_year: int = 2000,
    end_year: int = 2024,
) -> pd.DataFrame:
    panel_codes = panel_baci_codes(iso_map, countries)
    yearly_files = []
    for path in sorted(folder.glob("BACI_HS*_Y*_V*.csv")):
        match = YEARLY_FILE_PATTERN.search(path.name)
        if not match:
            continue
        year = int(match.group(1))
        if start_year <= year <= end_year:
            yearly_files.append((year, path))

    if not yearly_files:
        raise ValueError(f"No yearly BACI files for {start_year}-{end_year} in {folder}")

    print(f"Processing {len(yearly_files)} yearly BACI files ({start_year}-{end_year})...")
    parts = []
    for year, path in yearly_files:
        print(f"  {path.name}...")
        parts.append(process_yearly_file(path, year, panel_codes, iso_map))

    return pd.concat(parts, ignore_index=True)


def build_annual_edges(
    baci: pd.DataFrame,
    countries: list[str],
    start_year: int = 2000,
    end_year: int = 2024,
) -> pd.DataFrame:
    country_set = set(countries)
    baci = baci[(baci["year"] >= start_year) & (baci["year"] <= end_year)]
    baci = baci[baci["exporter"].isin(country_set) & baci["importer"].isin(country_set)]

    pair = baci.groupby(["year", "exporter", "importer"], as_index=False)["trade_usd"].sum()

    rows = []
    for _, row in pair.iterrows():
        a, b = sorted([row["exporter"], row["importer"]])
        rows.append({"year": int(row["year"]), "iso3_a": a, "iso3_b": b, "trade_usd": row["trade_usd"]})

    annual = pd.DataFrame(rows)
    return annual.groupby(["year", "iso3_a", "iso3_b"], as_index=False)["trade_usd"].sum()


def annual_to_quarterly(annual: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in annual.iterrows():
        for q in (1, 2, 3, 4):
            rows.append(
                {
                    "quarter": f"{int(row['year'])}Q{q}",
                    "iso3_a": row["iso3_a"],
                    "iso3_b": row["iso3_b"],
                    "trade_usd": row["trade_usd"] / 4.0,
                }
            )
    out = pd.DataFrame(rows)
    out["quarter_period"] = out["quarter"].apply(lambda x: pd.Period(x, freq="Q"))
    return out


def try_download_baci() -> Path | None:
    import requests

    BULK_DIR.mkdir(parents=True, exist_ok=True)
    for url in CEPII_URLS:
        print(f"  trying {url}...")
        try:
            resp = requests.get(url, timeout=120, stream=True)
            if resp.status_code != 200:
                continue
            dest = BULK_DIR / "baci_latest.zip"
            with dest.open("wb") as f:
                for chunk in resp.iter_content(chunk_size=65536):
                    f.write(chunk)
            print(f"  downloaded {dest}")
            return dest
        except requests.RequestException as exc:
            print(f"  failed: {exc}")
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Build trade edges from CEPII BACI bulk file")
    parser.add_argument("--try-download", action="store_true", help="Attempt CEPII URL download")
    args = parser.parse_args()

    ensure_dirs()
    BULK_DIR.mkdir(parents=True, exist_ok=True)
    protocol = load_protocol()
    countries = protocol["countries"]
    start_year = int(protocol["splits"]["full_start"][:4])
    end_year = int(protocol["splits"]["full_end"][:4])

    baci_folder = find_baci_folder()
    baci_path = find_baci_file()
    country_codes_path = find_country_codes_file(baci_folder)

    if baci_folder is None and baci_path is None and args.try_download:
        print("Attempting CEPII BACI download...")
        baci_path = try_download_baci()

    if baci_folder is None and baci_path is None:
        print(INSTRUCTIONS)
        return 1

    iso_map = load_baci_mapping(country_codes_path, countries)
    panel_codes = panel_baci_codes(iso_map, countries)
    print(f"Panel: {len(countries)} countries, {len(panel_codes)} BACI country codes")

    if baci_folder is not None:
        print(f"Reading BACI yearly folder: {baci_folder}")
        baci = load_baci_from_folder(baci_folder, iso_map, countries, start_year, end_year)
        source_label = str(baci_folder)
    else:
        print(f"Reading BACI from {baci_path}...")
        raw = read_baci(baci_path)
        baci = normalize_baci_columns(raw)
        baci["exporter"] = baci["exporter_num"].map(iso_map)
        baci["importer"] = baci["importer_num"].map(iso_map)
        baci = baci.dropna(subset=["exporter", "importer"])
        source_label = str(baci_path)

    print("Building annual bilateral edges...")
    annual = build_annual_edges(baci, countries, start_year, end_year)
    annual_path = DATA_RAW / "trade_bilateral_annual.csv"
    annual.to_csv(annual_path, index=False)
    print(f"  saved {annual_path} ({len(annual)} rows)")

    print("Expanding to quarterly (constant within year)...")
    quarterly = annual_to_quarterly(annual)
    quarterly_path = DATA_RAW / "trade_bilateral_quarterly.csv"
    quarterly.to_csv(quarterly_path, index=False)
    print(f"  saved {quarterly_path} ({len(quarterly)} rows)")

    majors = annual[
        ((annual["iso3_a"] == "USA") & (annual["iso3_b"] == "CHN"))
        | ((annual["iso3_a"] == "DEU") & (annual["iso3_b"] == "FRA"))
    ]
    if majors.empty:
        print("  warning: no USA-CHN or DEU-FRA edges — check country code mapping")
    else:
        print(f"  sanity check: {len(majors)} major-pair year rows (USA-CHN / DEU-FRA)")

    manifest = {
        "downloaded_at": datetime.now(timezone.utc).isoformat(),
        "source": "CEPII BACI bulk CSV",
        "baci_source": source_label,
        "country_codes_file": str(country_codes_path) if country_codes_path else None,
        "countries": countries,
        "year_range": [start_year, end_year],
        "annual_rows": len(annual),
        "quarterly_rows": len(quarterly),
        "note": "Annual flows held constant within each quarter",
    }
    manifest_path = DATA_RAW / "trade_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"  saved {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
