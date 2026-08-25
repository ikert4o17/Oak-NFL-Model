"""Test clean offensive scoring and defensive scoring suppression for Oak V12."""
from __future__ import annotations
from collections import defaultdict
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from oak_nfl.data.nflverse import load_pbp

START=2014; TRAIN_END=2022; HOLDOUT_START=2023; HOLDOUT_END=2025

def metrics(a,p): return {"games":len(a),"mae":float(mean_absolute_error(a,p)),"rmse":float(np.sqrt(mean_squared_error(a,p)))}

def prepare():
    pbp=pd.concat([load_pbp(y) for y in range(START,HOLDOUT_END+1)],ignore_index=True)
    base=pbp[["game_id","season","week","home_team","away_team","home_score","away_score","total_line"]].drop_duplicates("game_id",keep="last").copy()
    p=pbp[pbp.posteam.notna()].copy(); p["yards_gained"]=pd.to_numeric(p.yards_gained,errors="coerce").fillna(0)
    yards=p.groupby(["game_id","posteam"],as_index=False).yards_gained.sum()
    # Offensive scoring only: TDs credited to the offense plus field goals. Excludes return/defensive TDs.
    td_col="touchdown" if "touchdown" in p.columns else None
    fg_col="field_goal_result" if "field_goal_result" in p.columns else None
    if not td_col or not fg_col: raise ValueError("PBP missing touchdown/field_goal_result")
    p["off_td"]=(pd.to_numeric(p[td_col],errors="coerce").fillna(0).eq(1) & p.play_type.isin(["pass","run","qb_kneel"])).astype(int)
    p["fg_made"]=p[fg_col].fillna("").astype(str).str.lower().eq("made").astype(int)
    score=p.groupby(["game_id","posteam"],as_index=False).agg(off_td=("off_td","sum"),fg_made=("fg_made","sum"))
    score["off_points"]=7*score.off_td+3*score.fg_made
    team=yards.merge(score,on=["game_id","posteam"],how="left")
    h=team.rename(columns={"posteam":"home_team","yards_gained":"home_yards","off_points":"home_off_points"})[["game_id","home_team","home_yards","home_off_points"]]
    a=team.rename(columns={"posteam":"away_team","yards_gained":"away_yards","off_points":"away_off_points"})[["game_id","away_team","away_yards","away_off_points"]]
    g=base.merge(h,on=["game_id","home_team"]).merge(a,on=["game_id","away_team"])
    for c in ["home_score","away_score","total_line","home_yards","away_yards","home_off_points","away_off_points"]: g[c]=pd.to_numeric(g[c],errors="coerce")
    g["actual_total"]=g.home_score+g.away_score
    return g.dropna(subset=["actual_total","total_line","home_yards","away_yards"])

def features(g,windows=(3,4,5,6,8)):
    hist=defaultdict(list); rows=[]
    for r in g.sort_values(["season","week","game_id"]).itertuples(index=False):
        out={"game_id":r.game_id}
        for side,t in (("home",r.home_team),("away",r.away_team)):
            for w in windows:
                s=hist[t][-w:]; oy=sum(x[1] for x in s); dy=sum(x[3] for x in s)
                out[f"{side}_clean_ppa100_{w}"]=100*sum(x[0] for x in s)/oy if oy else np.nan
                out[f"{side}_clean_dpa100_{w}"]=100*sum(x[2] for x in s)/dy if dy else np.nan
        rows.append(out)
        hist[r.home_team].append((r.home_off_points,r.home_yards,r.away_off_points,r.away_yards))
        hist[r.away_team].append((r.away_off_points,r.away_yards,r.home_off_points,r.home_yards))
    return pd.DataFrame(rows)

def fit(tr,te,cols):
    m=Pipeline([("impute",SimpleImputer(strategy="median")),("scale",StandardScaler()),("ridge",Ridge(alpha=10))]); m.fit(tr[cols],tr.actual_total-tr.total_line); return te.total_line.to_numpy()+m.predict(te[cols])

def run():
    g=prepare(); g=g.merge(features(g),on="game_id"); tr=g[g.season.le(TRAIN_END)]; ho=g[g.season.between(HOLDOUT_START,HOLDOUT_END)]
    print("=== V12 CLEAN SCORING CONVERSION ==="); print("MARKET",metrics(ho.actual_total,ho.total_line))
    sets={}
    for w in [3,4,5,6,8]:
        sets[f"clean_ppa_{w}"]=[f"home_clean_ppa100_{w}",f"away_clean_ppa100_{w}"]
        sets[f"clean_dpa_{w}"]=[f"home_clean_dpa100_{w}",f"away_clean_dpa100_{w}"]
        sets[f"clean_both_{w}"]=sets[f"clean_ppa_{w}"]+sets[f"clean_dpa_{w}"]
    rows=[]
    for n,c in sets.items(): rows.append({"model":n,**metrics(ho.actual_total,fit(tr,ho,c))})
    res=pd.DataFrame(rows).sort_values(["mae","rmse"]); print(res.to_string(index=False))
    print("\n=== TOP 5 YEAR STABILITY ===")
    for n in res.head(5).model:
        pred=fit(tr,ho,sets[n]); x=ho[["season","actual_total","total_line"]].copy(); x["pred"]=pred
        for y,z in x.groupby("season"):
            a=metrics(z.actual_total,z.total_line); b=metrics(z.actual_total,z.pred); print(n,y,{"mae_delta":round(b['mae']-a['mae'],4),"rmse_delta":round(b['rmse']-a['rmse'],4)})
if __name__=="__main__": run()
