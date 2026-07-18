"""Shared, raw-only download and provenance utilities."""

from __future__ import annotations

import hashlib
import json
import re
from datetime import date, datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

import requests

from src.config import DATA_RAW, ROOT

V2_RAW = DATA_RAW / "v2"
METADATA_DIR = V2_RAW / "metadata"
ACQUISITION_RECORDS_DIR = METADATA_DIR / "acquisition_records"


def validate_release_date(value: str) -> str:
    """Accept only an explicit ISO release/vintage date supplied by the user."""
    try:
        return date.fromisoformat(value).isoformat()
    except ValueError as exc:
        raise ValueError("source release date must be ISO YYYY-MM-DD and must not be inferred") from exc


def safe_source_id(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._-")
    if not safe:
        raise ValueError("source identifier must contain letters or digits")
    return safe


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_official_https(url: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme != "https" or not parsed.netloc:
        raise ValueError("source URL must be a complete HTTPS URL")


def download_raw(
    *,
    url: str,
    destination: Path,
    headers: dict[str, str] | None = None,
    timeout_seconds: int = 120,
) -> dict[str, str]:
    """Write the unmodified response bytes and return response metadata."""
    require_official_https(url)
    destination.parent.mkdir(parents=True, exist_ok=True)
    response = requests.get(url, headers=headers, timeout=timeout_seconds)
    response.raise_for_status()
    destination.write_bytes(response.content)
    return {
        "http_status": str(response.status_code),
        "content_type": response.headers.get("content-type", ""),
        "etag": response.headers.get("etag", ""),
        "last_modified": response.headers.get("last-modified", ""),
    }


def write_acquisition_record(
    *,
    source_identifier: str,
    provider: str,
    dataset_name: str,
    frequency: str,
    coverage_description: str,
    release_lag_description: str,
    license_note: str,
    source_url: str,
    source_release_date: str,
    local_file: Path,
    response_metadata: dict[str, str],
) -> Path:
    """Create an immutable-style sidecar that later feeds registry and manifest creation."""
    if frequency not in {"monthly", "quarterly"}:
        raise ValueError("frequency must be monthly or quarterly")
    source_id = safe_source_id(source_identifier)
    record = {
        "source_identifier": source_id,
        "provider": provider,
        "dataset_name": dataset_name,
        "frequency": frequency,
        "coverage_description": coverage_description,
        "release_lag_description": release_lag_description,
        "license_note": license_note,
        "source_url": source_url,
        "source_release_date": validate_release_date(source_release_date),
        "retrieved_at": datetime.now(timezone.utc).isoformat(),
        "local_file": str(local_file.relative_to(ROOT)),
        "sha256": sha256(local_file),
        "response_metadata": response_metadata,
    }
    ACQUISITION_RECORDS_DIR.mkdir(parents=True, exist_ok=True)
    path = ACQUISITION_RECORDS_DIR / f"{source_id}.json"
    if path.exists():
        raise FileExistsError(f"acquisition record already exists: {path}; use a new release-specific source identifier")
    path.write_text(json.dumps(record, indent=2) + "\n")
    return path
