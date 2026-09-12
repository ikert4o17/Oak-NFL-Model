"""Validate V11 wind signal by holdout season and severity bucket."""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from oak_nfl.data.nflverse import load_schedules

TRAIN_END=2022; HOLDOUT_START=2023; HOLDOUT_END=2025

def metrics(a,p):
    return {"games":len(a),"mae":float(mean_absolute_error(a,p)),"rmse":float(np.sqrt(mean_squared_error(a,p)))}

def prepare():
    g=load_schedules(list(range(2000,HOLDOUT_END+1))).copy()
    g=g[g.game_type.isin(["REG","WC","DIV","CON","SB"])].copy()
    for c in ["home_score","away_score","total_line","wind"]: g[c]=pd.to_numeric(g[c],errors="coerce")
    g["actual_total"]=g.home_score+g.away_score
    roof=g.roof.fillna("").astype(str).str.lower(); g["outdoor"]=~roof.isin(["dome","closed"])
    wind=g.wind.fillna(0.0)
    g["wind_over_10"]=np.where(g.outdoor,np.maximum(wind-10,0),0)
    g["wind_over_15"]=np.where(g.outdoor,np.maximum(wind-15,0),0)
    return g.dropna(subset=["actual_total","total_line"])

def fit(train,test):
    features=["wind_over_10","wind_over_15"]
    model=Pipeline([("impute",SimpleImputer(strategy="median")),("scale",StandardScaler()),("ridge",Ridge(alpha=10.0))])
    model.fit(train[features],train.actual_total-train.total_line)
    return test.total_line.to_numpy()+model.predict(test[features])

def report(label,frame,pred):
    control=metrics(frame.actual_total,frame.total_line.to_numpy()); model=metrics(frame.actual_total,pred)
    print(label,{"games":len(frame),"control_mae":round(control["mae"],4),"wind_mae":round(model["mae"],4),"mae_delta":round(model["mae"]-control["mae"],4),"control_rmse":round(control["rmse"],4),"wind_rmse":round(model["rmse"],4),"rmse_delta":round(model["rmse"]-control["rmse"],4)})

def run():
    g=prepare(); train=g[g.season.le(TRAIN_END)].copy(); h=g[g.season.between(HOLDOUT_START,HOLDOUT_END)].copy(); h["wind_prediction"]=fit(train,h)
    print("=== OAK V11 WIND STABILITY ==="); report("ALL HOLDOUT",h,h.wind_prediction.to_numpy())
    print("\n=== BY SEASON ===")
    for season in range(HOLDOUT_START,HOLDOUT_END+1):
        x=h[h.season.eq(season)]; report(str(season),x,x.wind_prediction.to_numpy())
    print("\n=== BY WIND SEVERITY ===")
    outdoor=h[h.outdoor].copy()
    buckets=[("<10",outdoor.wind.lt(10)),("10-14",outdoor.wind.between(10,14.999)),("15-19",outdoor.wind.between(15,19.999)),("20+",outdoor.wind.ge(20))]
    for name,mask in buckets:
        x=outdoor[mask]; report(name,x,x.wind_prediction.to_numpy())
    print("\n=== HIGH-WIND SEASON CHECK ===")
    for season in range(HOLDOUT_START,HOLDOUT_END+1):
        x=h[h.season.eq(season)&h.outdoor&h.wind.ge(15)];
        if len(x): report(f"{season} wind15+",x,x.wind_prediction.to_numpy())
if __name__=="__main__": run()
