"""Strict rolling-origin eligibility and reproducibility metadata."""
from __future__ import annotations
import random
import numpy as np
import pandas as pd
import torch

def seed_everything(seed:int)->dict:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    return {"seed":seed,"python_seed":seed,"numpy_seed":seed,"torch_seed":seed}

def eligible_training(samples:pd.DataFrame,horizon:int,origin:pd.Period)->pd.DataFrame:
    out=samples[(samples.horizon_quarters==horizon)&(samples.target_quarter.map(lambda x:pd.Period(x,freq="Q"))<=origin)].copy()
    if out.empty: raise ValueError("no labels available at this forecast origin")
    return out

def training_metadata(train:pd.DataFrame)->dict:
    origins=train.origin_quarter.map(lambda x:pd.Period(x,freq="Q"))
    return {"training_sample_count":len(train),"earliest_training_quarter":str(origins.min()),"latest_training_quarter":str(origins.max())}
