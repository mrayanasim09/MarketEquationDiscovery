"""Shared guards and constants for the v2 transformation pipeline."""

from __future__ import annotations

import json
from pathlib import Path

from src.acquisition.common import sha256
from src.config import DATA_PROCESSED, DATA_RAW, ROOT

V2_RAW = DATA_RAW / "v2"
V2_PROCESSED = DATA_PROCESSED / "v2"
RAW_VALIDATION = V2_RAW / "validation_manifest.json"
MACRO_RAW = V2_RAW / "macro" / "macro_observations.csv"
TRADE_RAW = V2_RAW / "trade" / "trade_observations.csv"
SOURCE_REGISTRY = V2_RAW / "metadata" / "source_registry.csv"
COUNTRIES_FILE = V2_PROCESSED / "countries.json"

# Conservative pseudo-real-time reference-period lags. Raw source downloads are
# current-release snapshots, not historic vintages; these lags prevent use of
# same-quarter information in the forecasting design.
MACRO_LAG_QUARTERS = 1
TRADE_LAG_QUARTERS = 1
HORIZONS = (1, 2, 4)


def require_validated_raw() -> None:
    """Fail if raw inputs changed since their successful validation run."""
    if not RAW_VALIDATION.is_file():
        raise FileNotFoundError(f"missing raw validation manifest: {RAW_VALIDATION}")
    manifest = json.loads(RAW_VALIDATION.read_text())
    expected = {
        MACRO_RAW: manifest["raw_files"][str(MACRO_RAW.relative_to(ROOT))]["sha256"],
        TRADE_RAW: manifest["raw_files"][str(TRADE_RAW.relative_to(ROOT))]["sha256"],
        SOURCE_REGISTRY: manifest["raw_files"][str(SOURCE_REGISTRY.relative_to(ROOT))]["sha256"],
    }
    changed = [str(path.relative_to(ROOT)) for path, digest in expected.items() if not path.is_file() or sha256(path) != digest]
    if changed:
        raise RuntimeError(f"raw inputs differ from validation manifest; rerun raw validation before transformation: {changed}")


def ensure_output_dir() -> None:
    V2_PROCESSED.mkdir(parents=True, exist_ok=True)


def quarter_label(period) -> str:
    return str(period)
