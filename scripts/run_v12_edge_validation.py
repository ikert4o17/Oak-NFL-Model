"""Validate whether integrated V12 totals edge magnitude translates to betting performance."""
from __future__ import annotations

import numpy as np
import pandas as pd

from run_v12_integration_phase2 import (
    TEST_SEASONS,
    canonical_injuries,
    personnel_team_week,
)
from run_v12_core_shootout import cols
from run_v12_incremental import HOLDOUT_END, START, model, prepare
from run_v12_weather_validation import game_weather
from oak_nfl.data.nflverse import load_injuries, load_pbp, load_players, load_snap_counts
from oak_nfl.features import build_team_game_features, build_team_weekly_ratings
from oak_nfl.personnel_value import attach_player_values, attach_snap_player_ids, build_pregame_player_values
from oak_nfl.qb import build_pregame_qb_ratings


def build_games():
    pbp = pd.concat([load_pbp(y) for y in range(START, HOLDOUT_END + 1)], ignore_index=True)
    games = prepare(pbp).merge(game_weather(pbp), on="game_id", how="left")
    ratings = build_team_weekly_ratings(build_team_game_features(pbp))
    core = [c for c in ratings.columns if c.startswith("pregame_")]
    home = ratings.rename(columns={"team": "home_team", **{c: f"home_{c}" for c in core}})
    away = ratings.rename(columns={"team": "away_team", **{c: f"away_{c}" for c in core}})
    games = games.merge(home[["game_id", "home_team", *[f"home_{c}" for c in core]]], on=["game_id", "home_team"], how="left")
    games = games.merge(away[["game_id", "away_team", *[f"away_{c}" for c in core]]], on=["game_id", "away_team"], how="left")
    locked = ["home_ppa5", "away_ppa5", "home_dpa3", "away_dpa3"] + cols(core, ["epa_per_play"]) + cols(core, ["explosive_rate"])

    qb = build_pregame_qb_ratings(pbp)
    qcols = ["pregame_qb_epa", "pregame_qb_cpoe", "pregame_qb_sack_rate"]
    for side, team_col in [("home", "home_team"), ("away", "away_team")]:
        q = qb[["game_id", "team", *qcols]].rename(columns={"team": team_col, **{c: f"{side}_{c}" for c in qcols}})
        games = games.merge(q, on=["game_id", team_col], how="left")
    qb_features = [f"{side}_{c}" for side in ["home", "away"] for c in qcols]

    snaps = pd.concat([load_snap_counts(y) for y in range(START, 2025)], ignore_index=True)
    injuries = pd.concat([load_injuries(y) for y in range(START, 2025)], ignore_index=True)
    values = build_pregame_player_values(attach_snap_player_ids(snaps, load_players()))
    personnel = personnel_team_week(attach_player_values(canonical_injuries(injuries), values))
    games = games.merge(personnel.rename(columns={"team": "home_team", "personnel_points": "home_personnel"}), on=["season", "week", "home_team"], how="left")
    games = games.merge(personnel.rename(columns={"team": "away_team", "personnel_points": "away_personnel"}), on=["season", "week", "away_team"], how="left")
    games[["home_personnel", "away_personnel"]] = games[["home_personnel", "away_personnel"]].fillna(0)

    weather = ["temp", "wind", "outdoor", "temp_dev_65", "wind_outdoor", "extreme_temp_outdoor", "cold_40", "hot_85", "high_wind_15", "precip"]
    return games, locked + qb_features + ["home_personnel", "away_personnel"] + weather


def summarize(g, label):
    n = len(g)
    if not n:
        return
    decided = g[g.result.ne("push")]
    wins = int((decided.result == "win").sum())
    losses = int((decided.result == "loss").sum())
    win_rate = wins / (wins + losses) if wins + losses else np.nan
    roi_110 = (wins * (100 / 110) - losses) / (wins + losses) if wins + losses else np.nan
    print(label, {"games": n, "wins": wins, "losses": losses, "pushes": n - wins - losses, "win_rate": round(win_rate, 4), "roi_-110": round(roi_110, 4), "avg_abs_edge": round(g.abs_edge.mean(), 3)})


def run():
    games, features = build_games()
    preds = []
    for year in TEST_SEASONS:
        train = games[games.season.lt(year)].dropna(subset=["actual_total", "total_line"])
        test = games[games.season.eq(year)].dropna(subset=["actual_total", "total_line"]).copy()
        test["pred"] = model(train, test, features)
        preds.append(test[["season", "actual_total", "total_line", "pred"]])
    r = pd.concat(preds, ignore_index=True)
    r["edge"] = r.pred - r.total_line
    r["abs_edge"] = r.edge.abs()
    r["bet_over"] = r.edge.gt(0)
    r["result"] = np.where(r.actual_total.eq(r.total_line), "push", np.where((r.actual_total.gt(r.total_line) & r.bet_over) | (r.actual_total.lt(r.total_line) & ~r.bet_over), "win", "loss"))

    print("=== V12 INTEGRATED TOTALS EDGE VALIDATION ===")
    for lo, hi in [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, np.inf)]:
        summarize(r[(r.abs_edge >= lo) & (r.abs_edge < hi)], f"EDGE_{lo:g}_{hi if np.isfinite(hi) else 'PLUS'}")
    print("=== CUMULATIVE THRESHOLDS ===")
    for threshold in [1, 2, 3, 4, 5, 6, 7]:
        summarize(r[r.abs_edge >= threshold], f"EDGE_{threshold}_PLUS")
    print("=== DIRECTIONAL THRESHOLDS ===")
    for threshold in [2, 3, 4, 5]:
        summarize(r[(r.abs_edge >= threshold) & r.edge.gt(0)], f"OVER_{threshold}_PLUS")
        summarize(r[(r.abs_edge >= threshold) & r.edge.lt(0)], f"UNDER_{threshold}_PLUS")
    print("=== YEAR X THRESHOLD ===")
    for year in TEST_SEASONS:
        for threshold in [2, 3, 4, 5]:
            summarize(r[(r.season == year) & (r.abs_edge >= threshold)], f"{year}_EDGE_{threshold}_PLUS")


if __name__ == "__main__":
    run()
