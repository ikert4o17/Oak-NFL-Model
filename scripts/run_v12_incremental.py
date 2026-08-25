"""Test whether V12 PPA5+DPA3 adds value beyond Oak core efficiency metrics."""
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
from oak_nfl.features import build_team_game_features, build_team_weekly_ratings

START=2014; TRAIN_END=2022; HOLDOUT_START=2023; HOLDOUT_END=2025

def metrics(a,p): return {"games":len(a),"mae":float(mean_absolute_error(a,p)),"rmse":float(np.sqrt(mean_squared_error(a,p)))}
def model(tr,te,cols):
    m=Pipeline([("impute",SimpleImputer(strategy="median")),("scale",StandardScaler()),("ridge",Ridge(alpha=10))]); m.fit(tr[cols],tr.actual_total-tr.total_line); return te.total_line.to_numpy()+m.predict(te[cols])

def prepare(pbp):
    base=pbp[["game_id","season","week","home_team","away_team","home_score","away_score","total_line"]].drop_duplicates("game_id",keep="last").copy()
    p=pbp[pbp.posteam.notna()].copy(); p["yards_gained"]=pd.to_numeric(p.yards_gained,errors="coerce").fillna(0)
    p["off_td"]=(pd.to_numeric(p.touchdown,errors="coerce").fillna(0).eq(1)&p.play_type.isin(["pass","run","qb_kneel"])).astype(int); p["fg_made"]=p.field_goal_result.fillna("").astype(str).str.lower().eq("made").astype(int)
    team=p.groupby(["game_id","posteam"],as_index=False).agg(yards=("yards_gained","sum"),off_td=("off_td","sum"),fg_made=("fg_made","sum")); team["off_points"]=7*team.off_td+3*team.fg_made
    h=team.rename(columns={"posteam":"home_team","yards":"home_yards","off_points":"home_off_points"})[["game_id","home_team","home_yards","home_off_points"]]; a=team.rename(columns={"posteam":"away_team","yards":"away_yards","off_points":"away_off_points"})[["game_id","away_team","away_yards","away_off_points"]]
    g=base.merge(h,on=["game_id","home_team"]).merge(a,on=["game_id","away_team"]); g["actual_total"]=pd.to_numeric(g.home_score)+pd.to_numeric(g.away_score)
    hist=defaultdict(list); rows=[]
    for r in g.sort_values(["season","week","game_id"]).itertuples(index=False):
        out={"game_id":r.game_id}
        for side,t in (("home",r.home_team),("away",r.away_team)):
            s5=hist[t][-5:]; s3=hist[t][-3:]; oy=sum(x[1] for x in s5); dy=sum(x[3] for x in s3)
            out[f"{side}_ppa5"]=100*sum(x[0] for x in s5)/oy if oy else np.nan; out[f"{side}_dpa3"]=100*sum(x[2] for x in s3)/dy if dy else np.nan
        rows.append(out); hist[r.home_team].append((r.home_off_points,r.home_yards,r.away_off_points,r.away_yards)); hist[r.away_team].append((r.away_off_points,r.away_yards,r.home_off_points,r.home_yards))
    return g.merge(pd.DataFrame(rows),on="game_id")

def run():
    pbp=pd.concat([load_pbp(y) for y in range(START,HOLDOUT_END+1)],ignore_index=True); g=prepare(pbp)
    ratings=build_team_weekly_ratings(build_team_game_features(pbp)); core=[c for c in ratings.columns if c.startswith("pregame_")]
    home=ratings.rename(columns={"team":"home_team",**{c:f"home_{c}" for c in core}}); away=ratings.rename(columns={"team":"away_team",**{c:f"away_{c}" for c in core}})
    g=g.merge(home[["game_id","home_team",*[f"home_{c}" for c in core]]],on=["game_id","home_team"],how="left").merge(away[["game_id","away_team",*[f"away_{c}" for c in core]]],on=["game_id","away_team"],how="left")
    core_cols=[f"{s}_{c}" for s in ["home","away"] for c in core]; score_cols=["home_ppa5","away_ppa5","home_dpa3","away_dpa3"]
    tr=g[g.season.le(TRAIN_END)].dropna(subset=["actual_total","total_line"]); ho=g[g.season.between(HOLDOUT_START,HOLDOUT_END)].dropna(subset=["actual_total","total_line"])
    sets={"MARKET":[],"CORE_EFFICIENCY":core_cols,"PPA5_DPA3":score_cols,"CORE_PLUS_PPA5_DPA3":core_cols+score_cols}
    print("=== V12 INCREMENTAL VALUE ===")
    for n,c in sets.items():
        pred=ho.total_line.to_numpy() if not c else model(tr,ho,c); print(n,metrics(ho.actual_total,pred))
        if n!="MARKET":
            x=ho[["season","actual_total","total_line"]].copy(); x["pred"]=pred
            for y,z in x.groupby("season"):
                a=metrics(z.actual_total,z.total_line); b=metrics(z.actual_total,z.pred); print(" ",y,{"mae_delta":round(b['mae']-a['mae'],4),"rmse_delta":round(b['rmse']-a['rmse'],4)})
if __name__=="__main__": run()
