"""Additional non-graph baselines for the reconstructed engine."""
from __future__ import annotations
import numpy as np

class BayesianShrinkageVAR:
    """Conjugate ridge/shrinkage VAR approximation; not silently labeled ordinary VAR."""
    def __init__(self,lags:int=1,prior_precision:float=1.0): self.lags=lags; self.prior_precision=prior_precision
    def fit(self,y:np.ndarray):
        if len(y)<=self.lags: raise ValueError("insufficient history")
        x=np.array([y[t-self.lags:t].ravel() for t in range(self.lags,len(y))]); target=y[self.lags:]
        self.coef=np.linalg.solve(x.T@x+self.prior_precision*np.eye(x.shape[1]),x.T@target); self.last=y[-self.lags:].copy(); return self
    def forecast(self,steps:int)->np.ndarray:
        history=list(self.last.copy()); out=[]
        for _ in range(steps):
            nxt=np.asarray(history[-self.lags:]).ravel()@self.coef; out.append(nxt); history.append(nxt)
        return np.asarray(out)

def gradient_boosting_regressor():
    """Lazy dependency to keep engine construction separate from training."""
    try:
        from sklearn.ensemble import HistGradientBoostingRegressor
    except ImportError as exc:
        raise RuntimeError("gradient boosting requires scikit-learn; install it before the final benchmark run") from exc
    return HistGradientBoostingRegressor(max_iter=100,learning_rate=0.05,l2_regularization=1.0,random_state=0)
