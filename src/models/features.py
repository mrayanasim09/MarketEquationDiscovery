"""Origin-safe feature and sequence construction; never writes processed data."""
from __future__ import annotations
import numpy as np
import pandas as pd

def feature_sequence(panel:pd.DataFrame,country:str,end_quarter:pd.Period,k:int=4)->np.ndarray:
    subset=panel[(panel.entity_id==country)&(panel.period<=end_quarter)].sort_values("period").tail(k)
    if len(subset)!=k or subset[["cpi_yoy","energy_cpi_yoy"]].isna().any().any(): raise ValueError("incomplete permitted sequence")
    return subset[["cpi_yoy","energy_cpi_yoy"]].to_numpy(float)

def volatility(panel:pd.DataFrame,country:str,end_quarter:pd.Period,k:int=4)->float:
    subset=panel[(panel.entity_id==country)&(panel.period<=end_quarter)].sort_values("period").tail(k)["cpi_yoy"]
    if len(subset)!=k: raise ValueError("incomplete permitted volatility history")
    return float(subset.std(ddof=0))
