"""Focused V12 core shootout: scoring conversion plus selected efficiency families."""
from __future__ import annotations
import pandas as pd
from oak_nfl.data.nflverse import load_pbp
from oak_nfl.features import build_team_game_features, build_team_weekly_ratings
from run_v12_incremental import START,HOLDOUT_END,TRAIN_END,HOLDOUT_START,prepare,model,metrics

def cols(core, names):
    return [f"{s}_{c}" for s in ["home","away"] for c in core if any(c.endswith(f"_{n}") for n in names)]

def run():
    pbp=pd.concat([load_pbp(y) for y in range(START,HOLDOUT_END+1)],ignore_index=True); g=prepare(pbp)
    ratings=build_team_weekly_ratings(build_team_game_features(pbp)); core=[c for c in ratings.columns if c.startswith("pregame_")]
    home=ratings.rename(columns={"team":"home_team",**{c:f"home_{c}" for c in core}}); away=ratings.rename(columns={"team":"away_team",**{c:f"away_{c}" for c in core}})
    g=g.merge(home[["game_id","home_team",*[f"home_{c}" for c in core]]],on=["game_id","home_team"],how="left").merge(away[["game_id","away_team",*[f"away_{c}" for c in core]]],on=["game_id","away_team"],how="left")
    score=["home_ppa5","away_ppa5","home_dpa3","away_dpa3"]
    epa=cols(core,["epa_per_play"]); pas=cols(core,["pass_epa_per_play"]); exp=cols(core,["explosive_rate"]); suc=cols(core,["success_rate"])
    sets={
      "SCORE":score,
      "SCORE_EPA":score+epa,
      "SCORE_PASS":score+pas,
      "SCORE_EXP":score+exp,
      "SCORE_EPA_PASS":score+epa+pas,
      "SCORE_EPA_EXP":score+epa+exp,
      "SCORE_PASS_EXP":score+pas+exp,
      "SCORE_EPA_PASS_EXP":score+epa+pas+exp,
      "SCORE_EPA_PASS_SUCCESS":score+epa+pas+suc,
    }
    tr=g[g.season.le(TRAIN_END)].dropna(subset=["actual_total","total_line"]); ho=g[g.season.between(HOLDOUT_START,HOLDOUT_END)].dropna(subset=["actual_total","total_line"])
    market=metrics(ho.actual_total,ho.total_line); print("=== V12 FOCUSED CORE SHOOTOUT ==="); print("MARKET",market)
    rows=[]
    for n,c in sets.items():
        pred=model(tr,ho,c); m=metrics(ho.actual_total,pred); rows.append({"model":n,**m})
    res=pd.DataFrame(rows).sort_values(["mae","rmse"]); print(res.to_string(index=False))
    print("\n=== YEAR-BY-YEAR ALL CANDIDATES ===")
    for n in res.model:
        pred=model(tr,ho,sets[n]); x=ho[["season","actual_total","total_line"]].copy(); x["pred"]=pred
        for y,z in x.groupby("season"):
            a=metrics(z.actual_total,z.total_line); b=metrics(z.actual_total,z.pred); print(n,y,{"mae_delta":round(b['mae']-a['mae'],4),"rmse_delta":round(b['rmse']-a['rmse'],4)})
if __name__=="__main__": run()
