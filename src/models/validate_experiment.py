"""Preflight validation for benchmark-engine v2; does not train models."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from src.transform.common import V2_PROCESSED, require_validated_raw

ROOT=Path(__file__).resolve().parents[2]
RESULTS=ROOT/"experiments/results/v2"; CONFIG=RESULTS/"configs/benchmark_engine.json"
def main()->int:
    require_validated_raw(); cfg=json.loads(CONFIG.read_text()); errors=[]
    samples=pd.read_csv(V2_PROCESSED/"forecast_samples.csv"); countries=json.loads((V2_PROCESSED/"countries.json").read_text()); quarters=json.loads((V2_PROCESSED/"quarters.json").read_text()); adj=np.load(V2_PROCESSED/"adjacency_directed_trade_eur.npy")
    if len(cfg["seeds"])!=20 or len(set(cfg["seeds"]))!=20: errors.append("configuration must contain exactly 20 unique seeds")
    if cfg["epochs"]!=30: errors.append("locked benchmark configuration must use 30 epochs")
    if samples.duplicated(["country","origin_quarter","horizon_quarters"]).any(): errors.append("processed samples contain duplicate forecasts")
    if samples.macro_feature_quarter.ge(samples.origin_quarter).any() or samples.trade_graph_quarter.ge(samples.origin_quarter).any(): errors.append("future/same-origin feature or graph detected")
    if samples.target_quarter.le(samples.origin_quarter).any() or samples.target_cpi_yoy.isna().any(): errors.append("invalid or missing target")
    if not set(samples.trade_graph_quarter).issubset(set(quarters)): errors.append("sample graph quarter missing from snapshots")
    if adj.shape!=(len(quarters),len(countries),len(countries)): errors.append("graph tensor shape mismatches node/quarter contract")
    # Result checks activate only after experiments are stored.
    forecast_path=RESULTS/"forecasts.parquet"
    if forecast_path.exists():
        f=pd.read_parquet(forecast_path); keys=["model_name","model_variant","seed","horizon","forecast_origin","country"]
        if f.duplicated(keys).any(): errors.append("duplicate stored forecasts")
        neural=f[f.model_name.isin(["mlp","lstm","tcn","gcn","temporal_graph"])]
        if set(neural.seed.unique())!=set(cfg["seeds"]): errors.append("stored neural results do not contain all configured seeds")
    report={"passed":not errors,"mode":"post-run" if forecast_path.exists() else "pre-training","errors":errors,"seed_count":len(cfg["seeds"]),"countries":len(countries),"sample_rows":len(samples),"graph_shape":list(adj.shape)}
    (RESULTS/"metadata/validation.json").write_text(json.dumps(report,indent=2)+"\n")
    if errors:
        print("V2 BENCHMARK ENGINE VALIDATION FAILED"); print("\n".join(f"- {e}" for e in errors)); return 1
    print(f"V2 BENCHMARK ENGINE VALIDATION PASSED ({report['mode']}): {len(countries)} countries, {len(cfg['seeds'])} seeds")
    return 0
if __name__=="__main__": raise SystemExit(main())
