"""Drive-level scoring efficiency experiment against locked V12.

Predetermined candidates: offensive/defensive points per drive, TD rate per
 drive, and empty-drive rate. All ratings are shifted one game and season-local.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from oak_nfl.data.nflverse import load_pbp
from oak_nfl.features import build_team_game_features, build_team_weekly_ratings
from run_v12_core_shootout import cols
from run_v12_incremental import HOLDOUT_END, START, metrics, model, prepare

TEST_SEASONS=range(2019,2026)


def build_drive_games(pbp):
    p=pbp.copy()
    need={"game_id","season","week","posteam","defteam","drive","drive_result","fixed_drive_result"}
    missing=need-set(p.columns)
    if missing: raise ValueError(f"PBP missing drive columns: {sorted(missing)}")
    p=p[p.posteam.notna() & p.defteam.notna() & p.drive.notna()].copy()
    # One row per offensive drive. Prefer nflverse fixed result when available.
    result=p["fixed_drive_result"].fillna(p["drive_result"]).astype(str)
    p["_result"]=result
    d=(p.sort_values(["game_id","posteam","drive"]).groupby(["game_id","season","week","posteam","defteam","drive"],as_index=False).agg(result=("_result","last")))
    r=d.result.str.lower()
    d["points"] = np.select([r.str.contains("touchdown",na=False),r.str.contains("field goal",na=False),r.str.contains("safety",na=False)],[7.0,3.0,2.0],default=0.0)
    d["td"] = r.str.contains("touchdown",na=False).astype(float)
    d["empty"] = d.points.eq(0).astype(float)
    return d.groupby(["game_id","season","week","posteam","defteam"],as_index=False).agg(drives=("drive","size"),points_per_drive=("points","mean"),td_per_drive=("td","mean"),empty_drive_rate=("empty","mean"))


def ratings(dg):
    ms=["points_per_drive","td_per_drive","empty_drive_rate"]
    o=dg.rename(columns={"posteam":"team"})[["game_id","season","week","team",*ms]].rename(columns={m:f"off_{m}" for m in ms})
    de=dg.rename(columns={"defteam":"team"})[["game_id","season","week","team",*ms]].rename(columns={m:f"def_{m}_allowed" for m in ms})
    c=o.merge(de,on=["game_id","season","week","team"],how="outer",validate="one_to_one").sort_values(["team","season","week","game_id"])
    for col in [x for x in c if x.startswith(("off_","def_"))]: c[f"pregame_{col}"]=c.groupby(["team","season"])[col].transform(lambda s:s.expanding().mean().shift(1))
    return c


def run():
    pbp=pd.concat([load_pbp(y) for y in range(START,HOLDOUT_END+1)],ignore_index=True); g=prepare(pbp)
    tg=build_team_game_features(pbp); base=build_team_weekly_ratings(tg); core=[c for c in base if c.startswith("pregame_")]
    h=base.rename(columns={"team":"home_team",**{c:f"home_{c}" for c in core}}); a=base.rename(columns={"team":"away_team",**{c:f"away_{c}" for c in core}})
    g=g.merge(h[["game_id","home_team",*[f"home_{c}" for c in core]]],on=["game_id","home_team"],how="left").merge(a[["game_id","away_team",*[f"away_{c}" for c in core]]],on=["game_id","away_team"],how="left")
    r=ratings(build_drive_games(pbp)); rc=[c for c in r if c.startswith("pregame_")]
    h=r.rename(columns={"team":"home_team",**{c:f"home_{c}" for c in rc}}); a=r.rename(columns={"team":"away_team",**{c:f"away_{c}" for c in rc}})
    g=g.merge(h[["game_id","home_team",*[f"home_{c}" for c in rc]]],on=["game_id","home_team"],how="left").merge(a[["game_id","away_team",*[f"away_{c}" for c in rc]]],on=["game_id","away_team"],how="left")
    score=["home_ppa5","away_ppa5","home_dpa3","away_dpa3"]; locked=score+cols(core,["epa_per_play"])+cols(core,["explosive_rate"])
    sets={"LOCKED_V12":locked}
    for key in ["points_per_drive","td_per_drive","empty_drive_rate"]:
        f=[]
        for side in ["home","away"]:
            f += [f"{side}_pregame_off_{key}",f"{side}_pregame_def_{key}_allowed"]
        sets[key.upper()]=locked+f
    allp={k:[] for k in sets}
    print("=== V12 DRIVE EFFICIENCY EXPERIMENT ===")
    for y in TEST_SEASONS:
        tr=g[g.season.lt(y)].dropna(subset=["actual_total","total_line"]); te=g[g.season.eq(y)].dropna(subset=["actual_total","total_line"]); control=None
        print(f"\n=== {y} ===")
        for name,fs in sets.items():
            p=model(tr,te,fs); m=metrics(te.actual_total,p)
            if name=="LOCKED_V12": control=m
            row=dict(m)
            if name!="LOCKED_V12": row.update(mae_vs_locked=round(m["mae"]-control["mae"],4),rmse_vs_locked=round(m["rmse"]-control["rmse"],4))
            print(name,row); z=te[["actual_total"]].copy(); z["pred"]=p; allp[name].append(z)
    print("\n=== OVERALL 2019-2025 WALK-FORWARD ==="); cm=None
    for name,zs in allp.items():
        z=pd.concat(zs); m=metrics(z.actual_total,z.pred)
        if name=="LOCKED_V12": cm=m
        row=dict(m)
        if name!="LOCKED_V12": row.update(mae_vs_locked=round(m["mae"]-cm["mae"],4),rmse_vs_locked=round(m["rmse"]-cm["rmse"],4))
        print(name,row)
if __name__=="__main__": run()
