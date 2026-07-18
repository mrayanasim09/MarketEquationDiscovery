"""Run v2 rolling-origin benchmarks and graph ablations without altering data."""
from __future__ import annotations
import argparse, json, math
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.api import VAR
import torch
from torch import nn
from src.transform.common import V2_PROCESSED, require_validated_raw

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "experiments/configs/v2_benchmark.json"
OUT = ROOT / "experiments/results"

class MLP(nn.Module):
    def __init__(self, d): super().__init__(); self.net=nn.Sequential(nn.Linear(d,16),nn.ReLU(),nn.Linear(16,1))
    def forward(self,x): return self.net(x).squeeze(-1)
class LSTM(nn.Module):
    def __init__(self): super().__init__(); self.r=nn.LSTM(2,16,batch_first=True); self.o=nn.Linear(16,1)
    def forward(self,x): return self.o(self.r(x)[0][:,-1]).squeeze(-1)
class GCN(nn.Module):
    def __init__(self, temporal=False): super().__init__(); self.temporal=temporal; self.w=nn.Linear(2,16); self.o=nn.Linear(16,1)
    def forward(self,x,a):
        # x [N,2] or [K,N,2]; normalized directed aggregation supplied by caller
        if x.ndim==3: x=x.mean(0); a=a.mean(0)
        h=torch.relu(a@self.w(x)); return self.o(h).squeeze(-1)

def period(v): return pd.Period(v,freq="Q")
def normalize(a):
    a=np.asarray(a,float); d=a.sum(1,keepdims=True); return a/np.where(d>0,d,1.)
def graph_variant(a, variant, rng):
    a=a.copy()
    if variant=="zero_edge": return np.eye(len(a))
    if variant=="fully_connected": return np.ones_like(a)/len(a)
    if variant=="undirected": return normalize(a+a.T)
    if variant=="shuffled_edge":
        vals=a[a>0].copy(); rng.shuffle(vals); b=np.zeros_like(a); b[a>0]=vals; return normalize(b)
    return normalize(a)
def ridge_fit_predict(x,y,z,l=1.):
    mu=x.mean(0); sd=x.std(0); sd[sd==0]=1; xx=(x-mu)/sd; zz=(z-mu)/sd
    beta=np.linalg.solve(xx.T@xx+l*np.eye(xx.shape[1]),xx.T@y); return zz@beta

def dm(diff,h):
    n=len(diff); d=np.asarray(diff,float); m=d.mean(); lag=max(h-1,0); v=np.mean((d-m)**2)
    for k in range(1,lag+1): v+=2*(1-k/(lag+1))*np.mean((d[k:]-m)*(d[:-k]-m))
    stat=m/math.sqrt(v/n) if v>0 else np.nan
    return stat, 2*stats.t.sf(abs(stat),df=n-1) if np.isfinite(stat) else np.nan

def metric_table(f):
    r=[]
    for (model,h),g in f.groupby(["model","horizon"]):
        e=g.actual-g.prediction; # aggregate within origin for dependent panel
        o=g.assign(se=e**2,ae=np.abs(e)).groupby("origin").agg(se=("se","mean"),ae=("ae","mean"),mase=("mase","mean"),da=("directional","mean"))
        r.append({"model":model,"horizon":h,"rmse":float(np.sqrt(o.se.mean())),"mae":float(o.ae.mean()),"mase":float(o.mase.mean()),"directional_accuracy":float(o.da.mean()),"origins":len(o),"rmse_origin_sd":float(np.sqrt(o.se).std()),"mae_origin_sd":float(o.ae.std()),"mae_ci_low":float(o.ae.mean()-1.96*o.ae.std(ddof=1)/math.sqrt(len(o))),"mae_ci_high":float(o.ae.mean()+1.96*o.ae.std(ddof=1)/math.sqrt(len(o)))})
    return pd.DataFrame(r)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--neural-seeds",type=int,default=1); ap.add_argument("--epochs",type=int,default=30); args=ap.parse_args()
    require_validated_raw(); OUT.mkdir(parents=True,exist_ok=True); cfg=json.loads(CONFIG.read_text())
    s=pd.read_csv(V2_PROCESSED/"forecast_samples.csv"); feat=pd.read_csv(V2_PROCESSED/"quarterly_feature_panel.csv"); countries=json.loads((V2_PROCESSED/"countries.json").read_text()); idx={c:i for i,c in enumerate(countries)}
    adj=np.load(V2_PROCESSED/"adjacency_directed_trade_eur.npy"); qs=json.loads((V2_PROCESSED/"quarters.json").read_text()); qidx={q:i for i,q in enumerate(qs)}
    feat["p"]=feat.quarter.map(period); lookup=feat.set_index(["entity_id","p"])
    forecasts=[]
    models=["persistence","ridge","arima","var","mlp","lstm","gcn_directed","tgcn_directed","tgcn_zero_edge","tgcn_shuffled_edge","tgcn_fully_connected","tgcn_undirected","tgcn_no_energy"]
    for h in cfg["horizons"]:
      tests=sorted(s[(s.horizon_quarters==h)&(s.split=="test")].origin_quarter.unique(),key=period)
      for origin_s in tests:
        origin=period(origin_s); test=s[(s.horizon_quarters==h)&(s.origin_quarter==origin_s)].sort_values("country"); train=s[(s.horizon_quarters==h)&(s.target_quarter.map(period)<=origin)].copy()
        X=train[["cpi_yoy_input","energy_cpi_yoy_input"]].to_numpy(float); y=train.target_cpi_yoy.to_numpy(float); Z=test[["cpi_yoy_input","energy_cpi_yoy_input"]].to_numpy(float)
        base={"persistence":Z[:,0],"ridge":ridge_fit_predict(X,y,Z)}
        # country ARIMA/VAR use only reference-quarter history available at origin
        ar=[]; va=[]
        for row in test.itertuples():
          hist=feat[(feat.entity_id==row.country)&(feat.p<=period(row.macro_feature_quarter))][["cpi_yoy","energy_cpi_yoy"]].dropna().to_numpy(float)
          steps=h+1
          try: ar.append(float(ARIMA(hist[:,0],order=(1,0,0)).fit().forecast(steps)[-1]))
          except Exception: ar.append(float(row.cpi_yoy_input))
          try: va.append(float(VAR(hist).fit(maxlags=1,trend="c").forecast(hist[-1:],steps)[-1,0]))
          except Exception: va.append(float(row.cpi_yoy_input))
        base["arima"]=np.array(ar); base["var"]=np.array(va)
        # neural fits: only labels whose targets are available by this origin
        for seed in cfg["neural"]["seeds"][:args.neural_seeds]:
          torch.manual_seed(seed); np.random.seed(seed); rng=np.random.default_rng(seed)
          # MLP
          for name,net,inp in [("mlp",MLP(2),torch.tensor(X,dtype=torch.float32))]:
            opt=torch.optim.Adam(net.parameters(),lr=.01); yy=torch.tensor(y,dtype=torch.float32)
            for _ in range(args.epochs): opt.zero_grad(); loss=((net(inp)-yy)**2).mean(); loss.backward(); opt.step()
            base[name]=net(torch.tensor(Z,dtype=torch.float32)).detach().numpy()
          # LSTM: repeat observed two-feature vector to four positions only as a graph-free history baseline
          net=LSTM(); opt=torch.optim.Adam(net.parameters(),lr=.01); xx=torch.tensor(np.repeat(X[:,None,:],4,axis=1),dtype=torch.float32); yy=torch.tensor(y,dtype=torch.float32)
          for _ in range(args.epochs): opt.zero_grad(); loss=((net(xx)-yy)**2).mean(); loss.backward(); opt.step()
          base["lstm"]=net(torch.tensor(np.repeat(Z[:,None,:],4,axis=1),dtype=torch.float32)).detach().numpy()
          # graph models trained on whole-node snapshots for eligible origin samples
          for name,variant,temporal,energy in [("gcn_directed","directed",False,True),("tgcn_directed","directed",True,True),("tgcn_zero_edge","zero_edge",True,True),("tgcn_shuffled_edge","shuffled_edge",True,True),("tgcn_fully_connected","fully_connected",True,True),("tgcn_undirected","undirected",True,True),("tgcn_no_energy","directed",True,False)]:
            net=GCN(temporal); opt=torch.optim.Adam(net.parameters(),lr=.01)
            # latest eligible panel only avoids future labels while maintaining shared node set
            tr_o=max(train.origin_quarter.unique(),key=period); tg=train[train.origin_quarter==tr_o].sort_values("country"); a=graph_variant(adj[qidx[tg.trade_graph_quarter.iloc[0]]],variant,rng); xx=tg[["cpi_yoy_input","energy_cpi_yoy_input"]].to_numpy(float).copy(); xx[:,1]=xx[:,1] if energy else 0; yy=tg.target_cpi_yoy.to_numpy(float)
            tx=torch.tensor(xx,dtype=torch.float32); ta=torch.tensor(a,dtype=torch.float32); ty=torch.tensor(yy,dtype=torch.float32)
            for _ in range(args.epochs): opt.zero_grad(); loss=((net(tx,ta)-ty)**2).mean(); loss.backward(); opt.step()
            aa=graph_variant(adj[qidx[test.trade_graph_quarter.iloc[0]]],variant,rng); zz=Z.copy(); zz[:,1]=zz[:,1] if energy else 0; base[name]=net(torch.tensor(zz,dtype=torch.float32),torch.tensor(aa,dtype=torch.float32)).detach().numpy()
        for name,pred in base.items():
          for row,val in zip(test.itertuples(),np.asarray(pred)):
            scale=np.mean(np.abs(train[train.country==row.country].target_cpi_yoy-train[train.country==row.country].cpi_yoy_input)) or 1
            forecasts.append({"model":name,"horizon":h,"origin":origin_s,"country":row.country,"actual":row.target_cpi_yoy,"prediction":float(val),"mase":abs(row.target_cpi_yoy-val)/scale,"directional":int(np.sign(val-row.cpi_yoy_input)==np.sign(row.target_cpi_yoy-row.cpi_yoy_input))})
    f=pd.DataFrame(forecasts); f.to_csv(OUT/"v2_forecasts.csv",index=False); metrics=metric_table(f); metrics.to_csv(OUT/"v2_metrics.csv",index=False)
    # DM: best graph vs best non-graph by horizon, MAE loss aggregated by origin
    dmrows=[]
    for h in cfg["horizons"]:
      m=metrics[metrics.horizon==h]
      graph=m[m.model.isin(["gcn_directed", "tgcn_directed"])].sort_values("mae").iloc[0].model; nong=m[~m.model.str.startswith(("gcn","tgcn"))].sort_values("mae").iloc[0].model
      a=f[(f.horizon==h)&(f.model==graph)].assign(loss=lambda d:abs(d.actual-d.prediction)).groupby("origin").loss.mean(); b=f[(f.horizon==h)&(f.model==nong)].assign(loss=lambda d:abs(d.actual-d.prediction)).groupby("origin").loss.mean(); d=(a-b).dropna(); st,p=dm(d.to_numpy(),h); dmrows.append({"horizon":h,"trade_model":graph,"nontrade_model":nong,"origins":len(d),"mean_loss_difference_trade_minus_nontrade":d.mean(),"dm_stat":st,"p_value":p})
    pd.DataFrame(dmrows).to_csv(OUT/"v2_dm_tests.csv",index=False)
    # non-causal structural importance: mean observed outgoing trade share
    mean=adj.mean(0); imp=pd.DataFrame({"country":countries,"mean_outgoing_trade_eur":mean.sum(1),"mean_incoming_trade_eur":mean.sum(0)}).sort_values("mean_outgoing_trade_eur",ascending=False); imp.to_csv(OUT/"v2_graph_importance.csv",index=False)
    (OUT/"v2_run_metadata.json").write_text(json.dumps({"config":cfg,"neural_seeds_run":cfg["neural"]["seeds"][:args.neural_seeds],"epochs":args.epochs,"raw_inputs_unchanged":True},indent=2)+"\n")
    print(f"Completed {len(f):,} forecasts; outputs in {OUT}")
if __name__=="__main__": main()
