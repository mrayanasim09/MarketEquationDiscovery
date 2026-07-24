from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config" / "protocol.yaml"
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
RESULTS = ROOT / "results"


def load_protocol() -> dict[str, Any]:
    with CONFIG_PATH.open() as f:
        return yaml.safe_load(f)


def ensure_dirs() -> None:
    for path in (DATA_RAW, DATA_PROCESSED, RESULTS, ROOT / "paper"):
        path.mkdir(parents=True, exist_ok=True)
