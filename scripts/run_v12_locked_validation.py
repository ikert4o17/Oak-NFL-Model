"""Locked validation for V12 candidate: PPA5+DPA3+EPA/play+explosive rate.

No feature selection occurs here. The candidate is frozen before evaluating
rolling historical test seasons and game-environment slices.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from oak_nfl.data.nflverse import load_pbp
from oak_nfl.features import build_team_game_features, build_team_weekly_ratings
from run_v12_incremental import START, HOLDOUT_END, prepare, model, metrics
from run_v12_core_shootout import cols

TEST_SEASONS=range(2019,2026)

def run():
    pbp=pd.concat([load_pbp(y) for y in range(START,HOLDOUT_END+1)],ignore_index=True)
    g=prepare(pbp)
    ratings=build_team_weekly_ratings(build_team_game_features(pbp))
    core=[c for c in ratings.columns if c.startswith("pregame_")]
    home=ratings.rename(columns={"team":"home_team",**{c:f"home_{c}" for c in core}})
    away=ratings.rename(columns={"team":"away_team",**{c:f"away_{c}" for c in core}})
    g=g.merge(home[["game_id","home_team",*[f"home_{c}" for c in core]]],on=["game_id","home_team"],how="left")
    g=g.merge(away[["game_id","away_team",*[f"away_{c}" for c in core]]],on=["game_id","away_team"],how="left")
    score=["home_ppa5","away_ppa5","home_dpa3","away_dpa3"]
    locked=score+cols(core,["epa_per_play"])+cols(core,["explosive_rate"])
    if len(locked)!=len(set(locked)): raise ValueError("Duplicate locked features")

    print("=== V12 LOCKED EXPANDING WALK-FORWARD ===")
    all_rows=[]
    for y in TEST_SEASONS:
        tr=g[g.season.lt(y)].dropna(subset=["actual_total","total_line"])
        te=g[g.season.eq(y)].dropna(subset=["actual_total","total_line"])
        pred=model(tr,te,locked)
        mk=metrics(te.actual_total,te.total_line); md=metrics(te.actual_total,pred)
        print(y,{"games":len(te),"market_mae":round(mk['mae'],4),"model_mae":round(md['mae'],4),"mae_delta":round(md['mae']-mk['mae'],4),"market_rmse":round(mk['rmse'],4),"model_rmse":round(md['rmse'],4),"rmse_delta":round(md['rmse']-mk['rmse'],4)})
        x=te[["game_id","season","week","actual_total","total_line"]].copy(); x["pred"]=pred; all_rows.append(x)
    x=pd.concat(all_rows,ignore_index=True)
    print("OVERALL",{"market":metrics(x.actual_total,x.total_line),"model":metrics(x.actual_total,x.pred)})

    print("\n=== LOCKED CANDIDATE ENVIRONMENT SLICES ===")
    x["total_bucket"]=pd.cut(x.total_line,[-np.inf,42,46,50,np.inf],labels=["<=42","42-46","46-50",">50"],include_lowest=True)
    x["season_phase"]=pd.cut(x.week,[0,4,9,14,np.inf],labels=["W1-4","W5-9","W10-14","W15+"])
    for field in ["total_bucket","season_phase"]:
        print("--",field)
        for key,z in x.groupby(field,observed=True):
            mk=metrics(z.actual_total,z.total_line); md=metrics(z.actual_total,z.pred)
            print(str(key),{"games":len(z),"mae_delta":round(md['mae']-mk['mae'],4),"rmse_delta":round(md['rmse']-mk['rmse'],4)})
if __name__=="__main__": run()
