"""Schema-checked, append-only result storage for benchmark engine v2."""
from __future__ import annotations
import json
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
RESULTS=ROOT/"experiments/results/v2"
PROVENANCE_COLUMNS=["run_id","git_commit","dataset_version","configuration_id","execution_timestamp"]
FORECAST_COLUMNS=PROVENANCE_COLUMNS+["model_name","model_variant","graph_variant","seed","horizon","forecast_origin","country","target_quarter","prediction","actual","split","training_sample_count","earliest_training_quarter","latest_training_quarter","feature_set","graph_type"]
METRIC_COLUMNS=PROVENANCE_COLUMNS+["model_name","model_variant","graph_variant","seed","horizon","split","metric","value","origin_count","feature_set","graph_type"]
DM_COLUMNS=PROVENANCE_COLUMNS+["model_name","model_variant","graph_variant","seed","horizon","comparator_name","comparator_variant","dm_stat","p_value","origins","loss_difference"]

def path(name:str)->Path: return RESULTS/name

def append_parquet(name:str, frame:pd.DataFrame, columns:list[str], keys:list[str]) -> None:
    missing=set(columns)-set(frame.columns)
    if missing: raise ValueError(f"{name} missing schema columns: {sorted(missing)}")
    RESULTS.mkdir(parents=True, exist_ok=True)
    frame=frame[columns].copy()
    if any(frame[column].isna().any() or frame[column].astype(str).str.strip().eq("").any() for column in PROVENANCE_COLUMNS if column in columns):
        raise ValueError(f"{name} has missing execution provenance")
    target=path(name)
    if target.exists():
        old=pd.read_parquet(target)
        combined=pd.concat([old,frame],ignore_index=True)
    else: combined=frame
    if combined.duplicated(keys).any():
        raise ValueError(f"duplicate append rejected for {name} using keys {keys}")
    combined.to_parquet(target,index=False)

def write_run_manifest(payload:dict) -> None:
    """Append a distinct run record rather than overwriting prior seed/origin evidence."""
    RESULTS.mkdir(parents=True, exist_ok=True)
    target=path("run_manifest.json")
    existing=json.loads(target.read_text()) if target.exists() else {"engine_version":"2.0","runs":[]}
    if "runs" not in existing: raise ValueError("run_manifest.json has an invalid schema")
    record={**payload,"written_at":datetime.now(timezone.utc).isoformat()}
    run_id=record.get("run_id")
    if not run_id: raise ValueError("run manifest requires a unique run_id")
    if any(item.get("run_id")==run_id for item in existing["runs"]):
        raise ValueError(f"run manifest already contains run_id {run_id}")
    existing["runs"].append(record)
    target.write_text(json.dumps(existing,indent=2)+"\n")
