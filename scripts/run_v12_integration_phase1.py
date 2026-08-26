"""Phase 1 totals integration: locked V12 plus QB and non-QB personnel signals.

Existing QB/personnel modules were validated for margin, not totals. This script
forces them to re-earn inclusion against the frozen V12 totals candidate.
"""
from __future__ import annotations
import pandas as pd
from oak_nfl.data.nflverse import load_pbp, load_injuries, load_players, load_snap_counts
from oak_nfl.features import build_team_game_features, build_team_weekly_ratings
from oak_nfl.qb import build_pregame_qb_ratings
from oak_nfl.data.injuries import latest_weekly_status, normalize_injury_feed
from oak_nfl.personnel import POSITION_POINT_CAPS, player_absence_points
from oak_nfl.personnel_value import attach_player_values, attach_snap_player_ids, build_pregame_player_values
from run_v12_incremental import START,HOLDOUT_END,prepare,model,metrics
from run_v12_core_shootout import cols

TEST_SEASONS=range(2023,2025)  # injury history is reliable through 2024

def canonical_injuries(raw):
    d=raw.loc[raw.report_status.notna() & raw.report_status.astype(str).str.strip().ne("")].copy()
    return latest_weekly_status(normalize_injury_feed(d,column_map={"gsis_id":"player_id","full_name":"player_name","position":"position_group","report_status":"status","date_modified":"report_date"},source="nflverse_injuries"))

def personnel_team_week(availability):
    f=availability[availability.position_group.ne("QB")].copy()
    f["absence_points"]=[0.5*player_absence_points(p,v,s) for p,v,s in zip(f.position_group,f.player_value,f.status)]
    out=f.groupby(["season","week","team"],as_index=False).absence_points.sum().rename(columns={"absence_points":"personnel_points"})
    out["personnel_points"]=out.personnel_points.clip(-2,2)
    return out

def run():
    pbp=pd.concat([load_pbp(y) for y in range(START,HOLDOUT_END+1)],ignore_index=True)
    g=prepare(pbp)
    ratings=build_team_weekly_ratings(build_team_game_features(pbp)); core=[c for c in ratings.columns if c.startswith("pregame_")]
    home=ratings.rename(columns={"team":"home_team",**{c:f"home_{c}" for c in core}}); away=ratings.rename(columns={"team":"away_team",**{c:f"away_{c}" for c in core}})
    g=g.merge(home[["game_id","home_team",*[f"home_{c}" for c in core]]],on=["game_id","home_team"],how="left").merge(away[["game_id","away_team",*[f"away_{c}" for c in core]]],on=["game_id","away_team"],how="left")
    locked=["home_ppa5","away_ppa5","home_dpa3","away_dpa3"]+cols(core,["epa_per_play"])+cols(core,["explosive_rate"])

    qb=build_pregame_qb_ratings(pbp)
    qcols=["pregame_qb_epa","pregame_qb_cpoe","pregame_qb_sack_rate"]
    qh=qb[["game_id","team",*qcols]].rename(columns={"team":"home_team",**{c:f"home_{c}" for c in qcols}})
    qa=qb[["game_id","team",*qcols]].rename(columns={"team":"away_team",**{c:f"away_{c}" for c in qcols}})
    g=g.merge(qh,on=["game_id","home_team"],how="left").merge(qa,on=["game_id","away_team"],how="left")
    qb_features=[f"{s}_{c}" for s in ["home","away"] for c in qcols]

    snaps=pd.concat([load_snap_counts(y) for y in range(START,2025)],ignore_index=True); injuries=pd.concat([load_injuries(y) for y in range(START,2025)],ignore_index=True); players=load_players()
    values=build_pregame_player_values(attach_snap_player_ids(snaps,players)); avail=attach_player_values(canonical_injuries(injuries),values); adj=personnel_team_week(avail)
    ph=adj.rename(columns={"team":"home_team","personnel_points":"home_personnel"}); pa=adj.rename(columns={"team":"away_team","personnel_points":"away_personnel"})
    g=g.merge(ph,on=["season","week","home_team"],how="left").merge(pa,on=["season","week","away_team"],how="left"); g[["home_personnel","away_personnel"]]=g[["home_personnel","away_personnel"]].fillna(0)
    personnel=["home_personnel","away_personnel"]

    sets={"V12_LOCKED":locked,"V12_PLUS_QB":locked+qb_features,"V12_PLUS_PERSONNEL":locked+personnel,"V12_PLUS_QB_PERSONNEL":locked+qb_features+personnel}
    print("=== V12 TOTALS INTEGRATION PHASE 1 ===")
    for name,features in sets.items():
        preds=[]
        for y in TEST_SEASONS:
            tr=g[g.season.lt(y)].dropna(subset=["actual_total","total_line"]); te=g[g.season.eq(y)].dropna(subset=["actual_total","total_line"]).copy(); te["pred"]=model(tr,te,features); preds.append(te[["season","actual_total","total_line","pred"]])
        x=pd.concat(preds); print(name,metrics(x.actual_total,x.pred))
        for y,z in x.groupby("season"):
            mk=metrics(z.actual_total,z.total_line); md=metrics(z.actual_total,z.pred); print(" ",y,{"mae_delta":round(md['mae']-mk['mae'],4),"rmse_delta":round(md['rmse']-mk['rmse'],4)})
if __name__=="__main__": run()
