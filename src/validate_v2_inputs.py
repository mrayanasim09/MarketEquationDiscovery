"""Hard validation gates for the submission-grade v2 raw-data package.

This module validates source observations only.  It does not aggregate monthly
trade, construct features, fill gaps, or create forecast-ready variables.  Those
operations belong in later, separately auditable v2 pipeline stages.
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import DATA_RAW, ROOT

V2_RAW = DATA_RAW / "v2"
MACRO_DIR = V2_RAW / "macro"
TRADE_DIR = V2_RAW / "trade"
METADATA_DIR = V2_RAW / "metadata"
MACRO_FILE = MACRO_DIR / "macro_observations.csv"
TRADE_FILE = TRADE_DIR / "trade_observations.csv"
REGISTRY_FILE = METADATA_DIR / "source_registry.csv"
SCHEMA_FILES = [
    METADATA_DIR / "macro_observations.schema.json",
    METADATA_DIR / "trade_observations.schema.json",
    METADATA_DIR / "source_registry.schema.json",
    METADATA_DIR / "raw_manifest.schema.json",
]
ACQUISITION_MANIFEST_FILE = V2_RAW / "raw_manifest.json"
VALIDATION_MANIFEST_FILE = V2_RAW / "validation_manifest.json"
V2_PROCESSED = ROOT / "data" / "processed" / "v2"

MACRO_COLUMNS = [
    "entity_id", "observation_date", "variable", "value", "unit", "frequency",
    "source_release_date", "source_identifier", "source_url", "retrieved_at",
    "observation_status", "missing_reason",
]
TRADE_COLUMNS = [
    "exporter", "importer", "observation_period", "trade_value", "currency_unit",
    "frequency", "source_release_date", "source_identifier", "source_url",
    "retrieved_at", "observation_status", "missing_reason",
]
REGISTRY_COLUMNS = [
    "source_identifier", "provider", "dataset_name", "frequency",
    "coverage_description", "release_lag_description", "license_note",
]
MANIFEST_DOWNLOAD_FIELDS = {
    "source_identifier", "source_url", "retrieved_at", "source_release_date",
    "sha256", "local_file", "license_note",
}
PERIOD_PATTERNS = {
    "monthly": re.compile(r"\d{4}-(0[1-9]|1[0-2])$"),
    "quarterly": re.compile(r"\d{4}Q[1-4]$"),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def nonempty(frame: pd.DataFrame, column: str) -> pd.Series:
    return frame[column].notna() & frame[column].astype(str).str.strip().ne("")


def validate_directories_and_files() -> list[str]:
    required_dirs = [MACRO_DIR, TRADE_DIR, METADATA_DIR]
    required_files = [MACRO_FILE, TRADE_FILE, REGISTRY_FILE, ACQUISITION_MANIFEST_FILE, *SCHEMA_FILES]
    errors = [f"required v2 directory is missing: {path.relative_to(ROOT)}" for path in required_dirs if not path.is_dir()]
    errors.extend(
        f"required v2 input is missing: {path.relative_to(ROOT)}" for path in required_files if not path.is_file()
    )
    return errors


def read_csv(path: Path, label: str, errors: list[str]) -> pd.DataFrame | None:
    try:
        return pd.read_csv(path, dtype=str, keep_default_na=False)
    except (OSError, pd.errors.ParserError, UnicodeDecodeError) as exc:
        errors.append(f"{label} cannot be read as UTF-8 CSV: {exc}")
        return None


def validate_columns(frame: pd.DataFrame, expected: list[str], label: str) -> list[str]:
    actual = list(frame.columns)
    missing = sorted(set(expected) - set(actual))
    unexpected = sorted(set(actual) - set(expected))
    errors: list[str] = []
    if missing:
        errors.append(f"{label} missing required columns: {missing}")
    if unexpected:
        errors.append(f"{label} has undocumented columns: {unexpected}")
    if not missing and not unexpected and actual != expected:
        errors.append(f"{label} columns must use the documented order: {expected}")
    return errors


def validate_common_metadata(frame: pd.DataFrame, label: str) -> list[str]:
    errors: list[str] = []
    for column in ("source_release_date", "source_identifier", "source_url", "retrieved_at", "observation_status"):
        missing = int((~nonempty(frame, column)).sum())
        if missing:
            errors.append(f"{label} has {missing} missing {column} values")

    for column in ("source_release_date", "retrieved_at"):
        invalid = int(pd.to_datetime(frame[column], errors="coerce", utc=True).isna().sum())
        if invalid:
            errors.append(f"{label} has {invalid} invalid {column} values")
    invalid_urls = int((~frame["source_url"].str.match(r"https?://", na=False)).sum())
    if invalid_urls:
        errors.append(f"{label} has {invalid_urls} non-HTTP source_url values")

    not_observed = int(frame["observation_status"].ne("observed").sum())
    if not_observed:
        errors.append(
            f"{label} has {not_observed} non-observed rows; interpolated, imputed, and filled values are prohibited"
        )
    explained_fill = int(nonempty(frame, "missing_reason").sum())
    if explained_fill:
        errors.append(f"{label} has {explained_fill} rows with missing_reason; raw inputs must contain observed values only")
    return errors


def validate_macro(frame: pd.DataFrame) -> list[str]:
    errors = validate_columns(frame, MACRO_COLUMNS, "macro_observations.csv")
    if errors or frame.empty:
        return errors + (["macro_observations.csv contains no observations (the placeholder template is not data)"] if frame.empty else [])

    errors.extend(validate_common_metadata(frame, "macro_observations.csv"))
    for column in ("entity_id", "observation_date", "variable", "unit", "frequency"):
        missing = int((~nonempty(frame, column)).sum())
        if missing:
            errors.append(f"macro_observations.csv has {missing} missing {column} values")
    invalid_dates = int(pd.to_datetime(frame["observation_date"], errors="coerce").isna().sum())
    if invalid_dates:
        errors.append(f"macro_observations.csv has {invalid_dates} invalid observation_date values")
    invalid_values = int(pd.to_numeric(frame["value"], errors="coerce").isna().sum())
    if invalid_values:
        errors.append(f"macro_observations.csv has {invalid_values} missing or non-numeric value fields")
    invalid_frequency = int((~frame["frequency"].isin(PERIOD_PATTERNS)).sum())
    if invalid_frequency:
        errors.append("macro_observations.csv frequency must be monthly or quarterly; annual observations are not admissible")
    keys = ["entity_id", "observation_date", "variable", "source_identifier"]
    duplicates = int(frame.duplicated(keys).sum())
    if duplicates:
        errors.append(f"macro_observations.csv has {duplicates} duplicate observations on {keys}")
    return errors


def validate_trade(frame: pd.DataFrame) -> list[str]:
    errors = validate_columns(frame, TRADE_COLUMNS, "trade_observations.csv")
    if errors or frame.empty:
        return errors + (["trade_observations.csv contains no observations (the placeholder template is not data)"] if frame.empty else [])

    errors.extend(validate_common_metadata(frame, "trade_observations.csv"))
    for column in ("exporter", "importer", "observation_period", "currency_unit", "frequency"):
        missing = int((~nonempty(frame, column)).sum())
        if missing:
            errors.append(f"trade_observations.csv has {missing} missing {column} values")
    values = pd.to_numeric(frame["trade_value"], errors="coerce")
    invalid_values = int(values.isna().sum() + (values < 0).sum())
    if invalid_values:
        errors.append(f"trade_observations.csv has {invalid_values} missing, non-numeric, or negative trade_value fields")
    frequency_invalid = ~frame["frequency"].isin(PERIOD_PATTERNS)
    if frequency_invalid.any():
        errors.append("trade_observations.csv frequency must be monthly or quarterly; annual observations are not admissible")
    period_invalid = pd.Series(False, index=frame.index)
    for frequency, pattern in PERIOD_PATTERNS.items():
        mask = frame["frequency"].eq(frequency)
        period_invalid |= mask & ~frame["observation_period"].str.fullmatch(pattern, na=False)
    if period_invalid.any():
        errors.append("trade_observations.csv observation_period must be YYYY-MM for monthly or YYYYQn for quarterly rows")
    keys = ["exporter", "importer", "observation_period", "source_identifier"]
    duplicates = int(frame.duplicated(keys).sum())
    if duplicates:
        errors.append(f"trade_observations.csv has {duplicates} duplicate observations on {keys}")

    # A value duplicated across at least three reported quarters for a directed
    # pair is the recognizable signature of annual trade divided or copied into quarters.
    quarterly = frame[frame["frequency"].eq("quarterly")].copy()
    if not quarterly.empty:
        quarterly["year"] = quarterly["observation_period"].str[:4]
        quarterly["numeric_value"] = pd.to_numeric(quarterly["trade_value"], errors="coerce")
        pattern = quarterly.groupby(["year", "exporter", "importer", "source_identifier"])["numeric_value"].agg(["count", "nunique"])
        repeated = pattern[(pattern["count"] >= 3) & (pattern["nunique"] == 1)]
        if not repeated.empty:
            errors.append(
                f"trade_observations.csv has {len(repeated)} constant within-year quarterly pair series; "
                "annual-to-quarterly conversion is prohibited"
            )
    return errors


def validate_registry(registry: pd.DataFrame) -> list[str]:
    errors = validate_columns(registry, REGISTRY_COLUMNS, "source_registry.csv")
    if errors or registry.empty:
        return errors + (["source_registry.csv contains no registered sources"] if registry.empty else [])
    for column in REGISTRY_COLUMNS:
        missing = int((~nonempty(registry, column)).sum())
        if missing:
            errors.append(f"source_registry.csv has {missing} missing {column} values")
    duplicates = int(registry.duplicated(["source_identifier"]).sum())
    if duplicates:
        errors.append(f"source_registry.csv has {duplicates} duplicate source_identifier values")
    return errors


def validate_processed_model_compatibility() -> list[str]:
    """Validate persisted model inputs without changing raw or processed artifacts."""
    errors: list[str] = []
    required = {
        "countries": V2_PROCESSED / "countries.json",
        "quarters": V2_PROCESSED / "quarters.json",
        "feature_panel": V2_PROCESSED / "quarterly_feature_panel.csv",
        "forecast_samples": V2_PROCESSED / "forecast_samples.csv",
        "graph_manifest": V2_PROCESSED / "graph_manifest.json",
        "adjacency": V2_PROCESSED / "adjacency_directed_trade_eur.npy",
        "observed_mask": V2_PROCESSED / "adjacency_directed_observed_mask.npy",
    }
    missing = [str(path.relative_to(ROOT)) for path in required.values() if not path.is_file()]
    if missing:
        return [f"processed/model compatibility inputs are missing: {missing}"]

    import numpy as np

    try:
        countries = json.loads(required["countries"].read_text())
        quarters = json.loads(required["quarters"].read_text())
        graph_manifest = json.loads(required["graph_manifest"].read_text())
        panel = pd.read_csv(required["feature_panel"])
        samples = pd.read_csv(required["forecast_samples"])
        adjacency = np.load(required["adjacency"])
        observed_mask = np.load(required["observed_mask"])
    except (OSError, ValueError, json.JSONDecodeError, pd.errors.ParserError) as exc:
        return [f"processed/model compatibility inputs cannot be read: {exc}"]

    required_sample_columns = {
        "country", "origin_quarter", "horizon_quarters", "target_quarter",
        "macro_feature_quarter", "trade_graph_quarter", "cpi_yoy_input",
        "energy_cpi_yoy_input", "target_cpi_yoy", "split",
    }
    required_panel_columns = {"entity_id", "quarter", "cpi_yoy", "energy_cpi_yoy"}
    missing_sample_columns = sorted(required_sample_columns - set(samples.columns))
    missing_panel_columns = sorted(required_panel_columns - set(panel.columns))
    if missing_sample_columns:
        errors.append(f"forecast_samples.csv missing required model-input columns: {missing_sample_columns}")
    if missing_panel_columns:
        errors.append(f"quarterly_feature_panel.csv missing required feature columns: {missing_panel_columns}")
    if errors:
        return errors

    if not isinstance(countries, list) or len(countries) != len(set(countries)):
        errors.append("countries.json must contain unique registered entities")
    if not isinstance(quarters, list) or len(quarters) != len(set(quarters)):
        errors.append("quarters.json must contain unique registered quarter labels")
    if graph_manifest.get("countries") != countries or graph_manifest.get("quarters") != quarters:
        errors.append("graph_manifest node or temporal registration differs from countries.json/quarters.json")
    expected_shape = (len(quarters), len(countries), len(countries))
    if adjacency.shape != expected_shape or observed_mask.shape != expected_shape:
        errors.append(f"graph tensors must have shape {expected_shape}; found {adjacency.shape} and {observed_mask.shape}")

    if samples.duplicated(["country", "origin_quarter", "horizon_quarters"]).any():
        errors.append("forecast_samples.csv contains duplicate country-origin-horizon records")
    if set(samples["country"]) != set(countries) or not set(panel["entity_id"]).issubset(set(countries)):
        errors.append("feature or forecast entities do not match the registered graph node set")
    if samples[["cpi_yoy_input", "energy_cpi_yoy_input", "target_cpi_yoy"]].isna().any().any():
        errors.append("forecast_samples.csv has missing model inputs or targets")

    parsed = samples.copy()
    for column in ("origin_quarter", "target_quarter", "macro_feature_quarter", "trade_graph_quarter"):
        parsed[column] = pd.PeriodIndex(parsed[column], freq="Q")
    if (parsed["macro_feature_quarter"] >= parsed["origin_quarter"]).any() or (parsed["trade_graph_quarter"] >= parsed["origin_quarter"]).any():
        errors.append("processed samples contain same-origin or future macro/graph inputs")
    if (parsed["target_quarter"] <= parsed["origin_quarter"]).any():
        errors.append("processed samples contain non-future forecast targets")
    if not set(samples["trade_graph_quarter"]).issubset(set(quarters)):
        errors.append("forecast samples reference graph quarters absent from persisted snapshots")

    # LSTM/TCN and temporal-graph test inputs require four published quarterly values and snapshots.
    panel_index = panel.set_index(["entity_id", "quarter"])[["cpi_yoy", "energy_cpi_yoy"]]
    for row in parsed.loc[parsed["split"].eq("test")].itertuples(index=False):
        macro_history = pd.period_range(end=row.macro_feature_quarter, periods=4, freq="Q")
        graph_history = pd.period_range(end=row.trade_graph_quarter, periods=4, freq="Q")
        if any((row.country, str(period)) not in panel_index.index for period in macro_history):
            errors.append("test sequence input lacks a four-quarter feature history")
            break
        values = panel_index.loc[[(row.country, str(period)) for period in macro_history]]
        if values.isna().any().any():
            errors.append("test sequence input contains missing CPI or energy values")
            break
        if any(str(period) not in quarters for period in graph_history):
            errors.append("test temporal graph input lacks a required lagged graph snapshot")
            break
    return errors


def load_manifest(errors: list[str]) -> dict[str, Any] | None:
    try:
        manifest = json.loads(ACQUISITION_MANIFEST_FILE.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        errors.append(f"raw_manifest.json is invalid JSON: {exc}")
        return None
    if manifest.get("status") == "placeholder_no_data":
        errors.append("raw_manifest.json is a placeholder; replace it with official acquisition metadata")
    downloads = manifest.get("downloads")
    if not isinstance(downloads, list) or not downloads:
        errors.append("raw_manifest.json must contain a non-empty downloads list")
        return manifest
    for index, download in enumerate(downloads):
        if not isinstance(download, dict):
            errors.append(f"raw_manifest.json downloads[{index}] must be an object")
            continue
        missing = sorted(field for field in MANIFEST_DOWNLOAD_FIELDS if not str(download.get(field, "")).strip())
        if missing:
            errors.append(f"raw_manifest.json downloads[{index}] missing required metadata: {missing}")
        for date_field in ("retrieved_at", "source_release_date"):
            if date_field in download and pd.isna(pd.to_datetime(download[date_field], errors="coerce", utc=True)):
                errors.append(f"raw_manifest.json downloads[{index}] has invalid {date_field}")
        local_file = download.get("local_file")
        if local_file:
            path = ROOT / local_file
            if not path.is_file():
                errors.append(f"raw_manifest.json downloads[{index}] local_file does not exist: {local_file}")
            elif download.get("sha256") and sha256(path) != download["sha256"]:
                errors.append(f"raw_manifest.json downloads[{index}] sha256 does not match {local_file}")
    return manifest


def main() -> int:
    errors = validate_directories_and_files()
    if errors:
        print("V2 RAW INPUT VALIDATION FAILED")
        print("\n".join(f"- {error}" for error in errors))
        return 1

    macro = read_csv(MACRO_FILE, "macro_observations.csv", errors)
    trade = read_csv(TRADE_FILE, "trade_observations.csv", errors)
    registry = read_csv(REGISTRY_FILE, "source_registry.csv", errors)
    manifest = load_manifest(errors)
    if macro is not None:
        errors.extend(validate_macro(macro))
    if trade is not None:
        errors.extend(validate_trade(trade))
    if registry is not None:
        errors.extend(validate_registry(registry))

    if macro is not None and trade is not None and registry is not None and manifest is not None:
        observed_ids = set(macro["source_identifier"]) | set(trade["source_identifier"])
        registered_ids = set(registry["source_identifier"])
        manifest_ids = {str(item.get("source_identifier", "")) for item in manifest.get("downloads", []) if isinstance(item, dict)}
        missing_registry = sorted(observed_ids - registered_ids)
        missing_manifest = sorted(observed_ids - manifest_ids)
        if missing_registry:
            errors.append(f"observations reference unregistered source_identifier values: {missing_registry}")
        if missing_manifest:
            errors.append(f"observations have no matching raw_manifest.json download: {missing_manifest}")

    errors.extend(validate_processed_model_compatibility())

    if errors:
        print("V2 RAW INPUT VALIDATION FAILED")
        print("\n".join(f"- {error}" for error in errors))
        return 1

    assert macro is not None and trade is not None and registry is not None and manifest is not None
    validation_manifest = {
        "validated_at": datetime.now(timezone.utc).isoformat(),
        "protocol_version": "v2",
        "raw_files": {
            str(MACRO_FILE.relative_to(ROOT)): {"sha256": sha256(MACRO_FILE), "rows": len(macro)},
            str(TRADE_FILE.relative_to(ROOT)): {"sha256": sha256(TRADE_FILE), "rows": len(trade)},
            str(REGISTRY_FILE.relative_to(ROOT)): {"sha256": sha256(REGISTRY_FILE), "rows": len(registry)},
        },
        "source_identifiers": sorted(set(macro["source_identifier"]) | set(trade["source_identifier"])),
        "manifest_downloads": len(manifest["downloads"]),
        "raw_only": True,
        "annual_upsampling_forbidden": True,
    }
    VALIDATION_MANIFEST_FILE.write_text(json.dumps(validation_manifest, indent=2) + "\n")
    print(
        "V2 RAW INPUT VALIDATION PASSED: "
        f"Validated {len(macro):,} macro observations and {len(trade):,} trade observations from official releases"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
