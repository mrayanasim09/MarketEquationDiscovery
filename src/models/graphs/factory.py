"""Graph variants computed solely from the persisted quarterly trade snapshots."""
from __future__ import annotations
import numpy as np

GRAPH_VARIANTS = (
    "directed_trade",
    "log_trade",
    "import_dependence",
    "top_k_incoming",
    "reversed",
    "undirected",
    "degree_preserving_random",
    "identity_no_trade",
)


def row_normalize(a:np.ndarray)->np.ndarray:
    total=a.sum(axis=1,keepdims=True)
    return a/np.where(total>0,total,1.0)

def top_k_incoming(a:np.ndarray,k:int=5)->np.ndarray:
    out=np.zeros_like(a)
    for importer in range(a.shape[1]):
        keep=np.argsort(a[:,importer])[-k:]
        out[keep,importer]=a[keep,importer]
    return out

def degree_preserving_random(a:np.ndarray,rng:np.random.Generator)->np.ndarray:
    """Shuffle positive weights among existing directed edges, preserving degree/support."""
    out=np.zeros_like(a); mask=a>0; values=a[mask].copy(); rng.shuffle(values); out[mask]=values
    return out

def build(raw_trade:np.ndarray, variant:str, rng:np.random.Generator, top_k:int=5)->np.ndarray:
    a=np.asarray(raw_trade,dtype=float).copy()
    if variant=="directed_trade": out=a
    elif variant=="log_trade": out=np.log1p(a)
    elif variant=="import_dependence":
        # exporter->importer share of the importer's observed trade exposure
        out=a/np.where(a.sum(axis=0,keepdims=True)>0,a.sum(axis=0,keepdims=True),1.0)
    elif variant=="top_k_incoming": out=top_k_incoming(a,top_k)
    elif variant=="reversed": out=a.T
    elif variant=="undirected": out=a+a.T
    elif variant=="degree_preserving_random": out=degree_preserving_random(a,rng)
    elif variant=="identity_no_trade": out=np.eye(a.shape[0], dtype=float)
    else: raise ValueError(f"unknown graph variant: {variant}")
    return row_normalize(out)
