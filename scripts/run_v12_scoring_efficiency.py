"""Benchmark scoring conversion efficiency for Oak V12 totals modeling."""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from oak_nfl.data.nflverse import load_pbp
from oak_nfl.scoring_efficiency import build_scoring_efficiency

TRAIN_START=2014; TRAIN_END=2022; HOLDOUT_START=2023; HOLDOUT_END=2025

def metrics(a,p):
    return {"games":len(a),"mae":float(mean_absolute_error(a,p)),"rmse":float(np.sqrt(mean_squared_error(a,p)))}

def prepare_games():
    pbp=pd.concat([load_pbp(y) for y in range(TRAIN_START,HOLDOUT_END+1)],ignore_index=True)
    needed={"game_id","season","week","home_team","away_team","home_score","away_score","posteam","yards_gained","total_line"}
    missing=needed.difference(pbp.columns)
    if missing: raise ValueError(f"play-by-play missing V12 columns: {sorted(missing)}")
    base=(pbp[["game_id","season","week","home_team","away_team","home_score","away_score","total_line"]]
          .drop_duplicates("game_id",keep="last").copy())
    plays=pbp[pbp.posteam.notna()].copy(); plays["yards_gained"]=pd.to_numeric(plays.yards_gained,errors="coerce").fillna(0.0)
    yards=plays.groupby(["game_id","posteam"],as_index=False).yards_gained.sum()
    home=yards.rename(columns={"posteam":"home_team","yards_gained":"home_yards"})
    away=yards.rename(columns={"posteam":"away_team","yards_gained":"away_yards"})
    g=base.merge(home,on=["game_id","home_team"],how="left").merge(away,on=["game_id","away_team"],how="left")
    for c in ["home_score","away_score","total_line","home_yards","away_yards"]: g[c]=pd.to_numeric(g[c],errors="coerce")
    g["actual_total"]=g.home_score+g.away_score
    return g.dropna(subset=["actual_total","total_line","home_yards","away_yards"])

def fit(train,test,features):
    model=Pipeline([("impute",SimpleImputer(strategy="median")),("scale",StandardScaler()),("ridge",Ridge(alpha=10.0))])
    model.fit(train[features],train.actual_total-train.total_line)
    return test.total_line.to_numpy()+model.predict(test[features])

def run():
    g=prepare_games(); f=build_scoring_efficiency(g); g=g.merge(f,on="game_id",how="left")
    train=g[g.season.le(TRAIN_END)].copy(); holdout=g[g.season.between(HOLDOUT_START,HOLDOUT_END)].copy()
    print("=== OAK V12 SCORING EFFICIENCY ===")
    print(f"train games: {len(train)} | holdout games: {len(holdout)}")
    print("MARKET CONTROL",metrics(holdout.actual_total,holdout.total_line))
    sets={}
    for w in [4,8,"season"]:
        suffix=str(w)
        sets[f"dpa100_{suffix}"]=[f"home_dpa100_{suffix}",f"away_dpa100_{suffix}"]
        sets[f"ppa100_{suffix}"]=[f"home_ppa100_{suffix}",f"away_ppa100_{suffix}"]
        sets[f"ppa_dpa100_{suffix}"]=[f"home_ppa100_{suffix}",f"away_ppa100_{suffix}",f"home_dpa100_{suffix}",f"away_dpa100_{suffix}"]
    sets["all_windows"]=[c for c in g.columns if any(c.startswith(prefix) for prefix in ["home_ppa100_","away_ppa100_","home_dpa100_","away_dpa100_"])]
    rows=[]
    for name,features in sets.items(): rows.append({"model":name,**metrics(holdout.actual_total,fit(train,holdout,features))})
    result=pd.DataFrame(rows).sort_values(["mae","rmse"]); print(result.to_string(index=False))
    print("\n=== YEAR BY YEAR FOR TOP 3 ===")
    for name in result.head(3).model:
        features=sets[name]; pred=fit(train,holdout,features); temp=holdout[["season","actual_total","total_line"]].copy(); temp["pred"]=pred
        for season,x in temp.groupby("season"):
            control=metrics(x.actual_total,x.total_line); m=metrics(x.actual_total,x.pred)
            print(name,season,{"mae_delta":round(m["mae"]-control["mae"],4),"rmse_delta":round(m["rmse"]-control["rmse"],4)})
if __name__=="__main__": run()
