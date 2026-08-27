"""Test leakage-safe team-strength recency weighting against locked V12.

The production feature builder remains untouched. This experiment reconstructs
pregame ratings from team-game metrics with three predetermined horizons:
expanding mean (control), last 4 games, and a 50/50 blend of last 4 with the
expanding mean. No horizon search or post-result tuning is performed.
"""
from __future__ import annotations

import pandas as pd

from oak_nfl.data.nflverse import load_pbp
from oak_nfl.features import build_team_game_features, build_team_weekly_ratings
from run_v12_core_shootout import cols
from run_v12_incremental import HOLDOUT_END, START, metrics, model, prepare

TEST_SEASONS = range(2019, 2026)
METRICS = ["epa_per_play", "success_rate", "explosive_rate", "pass_epa_per_play", "rush_epa_per_play"]


def recency_ratings(team_games: pd.DataFrame) -> pd.DataFrame:
    offense = team_games.rename(columns={"posteam":"team"})[["season","week","game_id","team",*METRICS]].rename(columns={m:f"off_{m}" for m in METRICS})
    defense = team_games.rename(columns={"defteam":"team"})[["season","week","game_id","team",*METRICS]].rename(columns={m:f"def_{m}_allowed" for m in METRICS})
    c = offense.merge(defense,on=["season","week","game_id","team"],how="outer",validate="one_to_one").sort_values(["team","season","week","game_id"])
    metric_cols=[x for x in c.columns if x.startswith(("off_","def_"))]
    grp=c.groupby(["team","season"],sort=False)
    for col in metric_cols:
        c[f"pregame_{col}_exp"] = grp[col].transform(lambda s:s.expanding().mean().shift(1))
        c[f"pregame_{col}_last4"] = grp[col].transform(lambda s:s.shift(1).rolling(4,min_periods=1).mean())
        c[f"pregame_{col}_blend"] = 0.5*c[f"pregame_{col}_exp"] + 0.5*c[f"pregame_{col}_last4"]
    return c


def attach(g: pd.DataFrame, r: pd.DataFrame, suffix: str) -> pd.DataFrame:
    fields=[c for c in r.columns if c.endswith(f"_{suffix}")]
    h=r.rename(columns={"team":"home_team",**{c:f"home_{c}" for c in fields}})
    a=r.rename(columns={"team":"away_team",**{c:f"away_{c}" for c in fields}})
    return g.merge(h[["game_id","home_team",*[f"home_{c}" for c in fields]]],on=["game_id","home_team"],how="left").merge(a[["game_id","away_team",*[f"away_{c}" for c in fields]]],on=["game_id","away_team"],how="left")


def run() -> None:
    pbp=pd.concat([load_pbp(y) for y in range(START,HOLDOUT_END+1)],ignore_index=True)
    g=prepare(pbp)
    tg=build_team_game_features(pbp)
    standard=build_team_weekly_ratings(tg)
    core=[c for c in standard.columns if c.startswith("pregame_")]
    h=standard.rename(columns={"team":"home_team",**{c:f"home_{c}" for c in core}})
    a=standard.rename(columns={"team":"away_team",**{c:f"away_{c}" for c in core}})
    g=g.merge(h[["game_id","home_team",*[f"home_{c}" for c in core]]],on=["game_id","home_team"],how="left").merge(a[["game_id","away_team",*[f"away_{c}" for c in core]]],on=["game_id","away_team"],how="left")
    rr=recency_ratings(tg)
    for s in ["last4","blend"]: g=attach(g,rr,s)

    score=["home_ppa5","away_ppa5","home_dpa3","away_dpa3"]
    locked=score+cols(core,["epa_per_play"])+cols(core,["explosive_rate"])
    base_metrics=["off_epa_per_play","def_epa_per_play_allowed","off_explosive_rate","def_explosive_rate_allowed"]
    sets={"LOCKED_V12":locked}
    for s in ["last4","blend"]:
        fs=score+sum(([f"home_pregame_{m}_{s}",f"away_pregame_{m}_{s}"] for m in base_metrics),[])
        sets[f"RECENCY_{s.upper()}"]=fs

    all_rows={k:[] for k in sets}
    print("=== V12 RECENCY WEIGHTING EXPERIMENT ===")
    print("Predetermined candidates: last-4 and 50/50 expanding+last-4 blend.")
    for y in TEST_SEASONS:
        tr=g[g.season.lt(y)].dropna(subset=["actual_total","total_line"]); te=g[g.season.eq(y)].dropna(subset=["actual_total","total_line"])
        print(f"\n=== {y} ===")
        control=None
        for name,fs in sets.items():
            p=model(tr,te,fs); m=metrics(te.actual_total,p)
            if name=="LOCKED_V12": control=m
            row={**m}
            if control is not None and name!="LOCKED_V12":
                row["mae_vs_locked"]=round(m["mae"]-control["mae"],4); row["rmse_vs_locked"]=round(m["rmse"]-control["rmse"],4)
            print(name,row)
            z=te[["game_id","season","actual_total"]].copy(); z["pred"]=p; all_rows[name].append(z)
    print("\n=== OVERALL 2019-2025 WALK-FORWARD ===")
    cm=None
    for name,pieces in all_rows.items():
        z=pd.concat(pieces,ignore_index=True); m=metrics(z.actual_total,z.pred)
        if name=="LOCKED_V12": cm=m
        row={**m}
        if cm is not None and name!="LOCKED_V12":
            row["mae_vs_locked"]=round(m["mae"]-cm["mae"],4); row["rmse_vs_locked"]=round(m["rmse"]-cm["rmse"],4)
        print(name,row)

if __name__=="__main__": run()
