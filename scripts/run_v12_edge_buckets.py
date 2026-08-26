"""Locked V12 edge-strength validation.

Tests whether larger absolute disagreements between the frozen V12 totals
candidate and the closing market correspond to stronger predictive value.
No features or thresholds are selected by this script.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from oak_nfl.data.nflverse import load_pbp
from oak_nfl.features import build_team_game_features, build_team_weekly_ratings
from run_v12_incremental import START, HOLDOUT_END, prepare, model, metrics
from run_v12_core_shootout import cols

TEST_SEASONS=range(2019,2026)
EDGE_BINS=[0,1,2,3,5,np.inf]
EDGE_LABELS=["<1","1-2","2-3","3-5","5+"]

def run():
    pbp=pd.concat([load_pbp(y) for y in range(START,HOLDOUT_END+1)],ignore_index=True)
    g=prepare(pbp)
    ratings=build_team_weekly_ratings(build_team_game_features(pbp))
    core=[c for c in ratings.columns if c.startswith("pregame_")]
    home=ratings.rename(columns={"team":"home_team",**{c:f"home_{c}" for c in core}})
    away=ratings.rename(columns={"team":"away_team",**{c:f"away_{c}" for c in core}})
    g=g.merge(home[["game_id","home_team",*[f"home_{c}" for c in core]]],on=["game_id","home_team"],how="left")
    g=g.merge(away[["game_id","away_team",*[f"away_{c}" for c in core]]],on=["game_id","away_team"],how="left")
    locked=["home_ppa5","away_ppa5","home_dpa3","away_dpa3"]+cols(core,["epa_per_play"])+cols(core,["explosive_rate"])
    if len(locked)!=len(set(locked)): raise ValueError("Duplicate locked features")

    rows=[]
    for y in TEST_SEASONS:
        tr=g[g.season.lt(y)].dropna(subset=["actual_total","total_line"])
        te=g[g.season.eq(y)].dropna(subset=["actual_total","total_line"]).copy()
        te["pred"]=model(tr,te,locked)
        te["edge"]=te.pred-te.total_line
        te["abs_edge"]=te.edge.abs()
        te["side_correct"]=np.where(te.edge>=0,te.actual_total>te.total_line,te.actual_total<te.total_line)
        te.loc[te.actual_total.eq(te.total_line),"side_correct"]=np.nan
        rows.append(te[["game_id","season","week","actual_total","total_line","pred","edge","abs_edge","side_correct"]])
    x=pd.concat(rows,ignore_index=True)
    x["edge_bucket"]=pd.cut(x.abs_edge,EDGE_BINS,labels=EDGE_LABELS,right=False,include_lowest=True)

    print("=== V12 LOCKED MARKET DISAGREEMENT BUCKETS ===")
    print("OVERALL",{"games":len(x),"market":metrics(x.actual_total,x.total_line),"model":metrics(x.actual_total,x.pred)})
    for b,z in x.groupby("edge_bucket",observed=True):
        mk=metrics(z.actual_total,z.total_line); md=metrics(z.actual_total,z.pred)
        decided=z.side_correct.dropna()
        print(str(b),{"games":len(z),"avg_abs_edge":round(z.abs_edge.mean(),3),"mae_delta":round(md['mae']-mk['mae'],4),"rmse_delta":round(md['rmse']-mk['rmse'],4),"direction_accuracy":round(float(decided.mean()),4) if len(decided) else None})

    print("\n=== OVER VS UNDER DISAGREEMENT ===")
    for side,z in [("OVER",x[x.edge>0]),("UNDER",x[x.edge<0])]:
        mk=metrics(z.actual_total,z.total_line); md=metrics(z.actual_total,z.pred); decided=z.side_correct.dropna()
        print(side,{"games":len(z),"avg_abs_edge":round(z.abs_edge.mean(),3),"mae_delta":round(md['mae']-mk['mae'],4),"rmse_delta":round(md['rmse']-mk['rmse'],4),"direction_accuracy":round(float(decided.mean()),4)})

    print("\n=== EDGE BUCKET YEAR STABILITY ===")
    for b,z in x.groupby("edge_bucket",observed=True):
        for y,q in z.groupby("season"):
            if len(q)<10: continue
            mk=metrics(q.actual_total,q.total_line); md=metrics(q.actual_total,q.pred); decided=q.side_correct.dropna()
            print(str(b),int(y),{"games":len(q),"mae_delta":round(md['mae']-mk['mae'],4),"direction_accuracy":round(float(decided.mean()),4) if len(decided) else None})
if __name__=="__main__": run()
