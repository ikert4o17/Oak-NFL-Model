"""Frozen validation of the V12 explosive-play matchup interaction candidate.

Candidate selection is complete before this script: locked V12 plus the four
explosive matchup terms from run_v12_matchup_interactions. No thresholds or
feature selection are tuned here. Evaluation is expanding walk-forward and
reports season and broad environment stability against the locked V12 control.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from oak_nfl.data.nflverse import load_pbp
from oak_nfl.features import build_team_game_features, build_team_weekly_ratings
from run_v12_core_shootout import cols
from run_v12_incremental import HOLDOUT_END, START, metrics, model, prepare
from run_v12_matchup_interactions import add_matchup_interactions

TEST_SEASONS = range(2019, 2026)


def run() -> None:
    pbp = pd.concat([load_pbp(y) for y in range(START, HOLDOUT_END + 1)], ignore_index=True)
    g = prepare(pbp)
    ratings = build_team_weekly_ratings(build_team_game_features(pbp))
    core = [c for c in ratings.columns if c.startswith("pregame_")]
    home = ratings.rename(columns={"team": "home_team", **{c: f"home_{c}" for c in core}})
    away = ratings.rename(columns={"team": "away_team", **{c: f"away_{c}" for c in core}})
    g = g.merge(home[["game_id", "home_team", *[f"home_{c}" for c in core]]], on=["game_id", "home_team"], how="left")
    g = g.merge(away[["game_id", "away_team", *[f"away_{c}" for c in core]]], on=["game_id", "away_team"], how="left")
    g = add_matchup_interactions(g)

    score = ["home_ppa5", "away_ppa5", "home_dpa3", "away_dpa3"]
    locked = score + cols(core, ["epa_per_play"]) + cols(core, ["explosive_rate"])
    explosive = [
        "home_explosive_matchup_product", "away_explosive_matchup_product",
        "explosive_matchup_sum", "explosive_matchup_imbalance",
    ]
    candidate = locked + explosive
    if not all(c in g.columns for c in explosive):
        raise ValueError("Missing frozen explosive matchup terms")

    rows = []
    print("=== FROZEN V12 EXPLOSIVE MATCHUP VALIDATION ===")
    for y in TEST_SEASONS:
        tr = g[g.season.lt(y)].dropna(subset=["actual_total", "total_line"])
        te = g[g.season.eq(y)].dropna(subset=["actual_total", "total_line"])
        p0 = model(tr, te, locked)
        p1 = model(tr, te, candidate)
        m0 = metrics(te.actual_total, p0); m1 = metrics(te.actual_total, p1)
        print(y, {"games": len(te), "mae_delta_vs_locked": round(m1["mae"]-m0["mae"],4), "rmse_delta_vs_locked": round(m1["rmse"]-m0["rmse"],4), "candidate_mae": round(m1["mae"],4), "candidate_rmse": round(m1["rmse"],4)})
        x = te[["game_id","season","week","actual_total","total_line"]].copy()
        x["locked_pred"] = p0; x["candidate_pred"] = p1
        rows.append(x)

    x = pd.concat(rows, ignore_index=True)
    m0 = metrics(x.actual_total, x.locked_pred); m1 = metrics(x.actual_total, x.candidate_pred)
    print("OVERALL", {"games": len(x), "mae_delta_vs_locked": round(m1["mae"]-m0["mae"],4), "rmse_delta_vs_locked": round(m1["rmse"]-m0["rmse"],4)})

    x["total_bucket"] = pd.cut(x.total_line, [-np.inf,42,46,50,np.inf], labels=["<=42","42-46","46-50",">50"], include_lowest=True)
    x["season_phase"] = pd.cut(x.week, [0,4,9,14,np.inf], labels=["W1-4","W5-9","W10-14","W15+"])
    print("\n=== BROAD STABILITY SLICES ===")
    for field in ["total_bucket","season_phase"]:
        print("--", field)
        for key,z in x.groupby(field, observed=True):
            a=metrics(z.actual_total,z.locked_pred); b=metrics(z.actual_total,z.candidate_pred)
            print(str(key), {"games":len(z), "mae_delta_vs_locked":round(b["mae"]-a["mae"],4), "rmse_delta_vs_locked":round(b["rmse"]-a["rmse"],4)})

if __name__ == "__main__":
    run()
