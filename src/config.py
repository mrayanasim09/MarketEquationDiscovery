from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"


def ensure_dirs() -> None:
    for path in (DATA_RAW, DATA_PROCESSED, ROOT / "paper"):
        path.mkdir(parents=True, exist_ok=True)
