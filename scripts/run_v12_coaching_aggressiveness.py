"""Coaching/aggressiveness challenger against promoted V12 + rest differential.

Predetermined, leakage-safe team-season behavior features:
1. neutral pass rate: pass share on early downs in competitive game states;
2. fourth-down go rate: share of eligible 4th-and-short/medium decisions where the
   offense goes for it rather than punts/attempts a field goal;
3. combined behavior package.

All pregame ratings are season-to-date expanding means shifted one game. This is
not a pace test and does not tune game-state windows or fourth-down thresholds.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from oak_nfl.data.nflverse import load_pbp
from oak_nfl.features import build_team_game_features, build_team_weekly_ratings
from run_v12_core_shootout import cols
from run_v12_incremental import HOLDOUT_END, START, metrics, model, prepare
from run_v12_schedule_context import build_schedule_context

TEST_SEASONS = range(2019, 2026)


def build_behavior_games(pbp: pd.DataFrame) -> pd.DataFrame:
    required = {"game_id", "season", "week", "posteam", "down", "ydstogo", "play_type"}
    missing = sorted(required.difference(pbp.columns))
    if missing:
        raise ValueError(f"PBP missing coaching-behavior columns: {missing}")

    p = pbp.copy()
    p["down"] = pd.to_numeric(p["down"], errors="coerce")
    p["ydstogo"] = pd.to_numeric(p["ydstogo"], errors="coerce")

    # Predetermined neutral-script proxy: first/second down, pass/run plays, and
    # score differential within 8 points when score fields are available.
    neutral = p.posteam.notna() & p.down.isin([1, 2]) & p.play_type.isin(["pass", "run"])
    if {"posteam_score", "defteam_score"}.issubset(p.columns):
        ps = pd.to_numeric(p.posteam_score, errors="coerce")
        ds = pd.to_numeric(p.defteam_score, errors="coerce")
        neutral &= (ps - ds).abs().le(8)
    n = p.loc[neutral, ["game_id", "season", "week", "posteam", "play_type"]].copy()
    n["is_pass"] = n.play_type.eq("pass").astype(float)
    neutral_game = n.groupby(["game_id", "season", "week", "posteam"], as_index=False).agg(
        neutral_pass_rate=("is_pass", "mean")
    ).rename(columns={"posteam": "team"})

    # Fixed eligible decision set: 4th down with <=5 yards to go. A pass/run is
    # a go; punts and field-goal attempts are non-go decisions. No threshold search.
    fourth = p.posteam.notna() & p.down.eq(4) & p.ydstogo.le(5) & p.play_type.isin(["pass", "run", "punt", "field_goal"])
    f = p.loc[fourth, ["game_id", "season", "week", "posteam", "play_type"]].copy()
    f["go"] = f.play_type.isin(["pass", "run"]).astype(float)
    fourth_game = f.groupby(["game_id", "season", "week", "posteam"], as_index=False).agg(
        fourth_down_go_rate=("go", "mean")
    ).rename(columns={"posteam": "team"})

    return neutral_game.merge(fourth_game, on=["game_id", "season", "week", "team"], how="outer")


def build_pregame_ratings(g: pd.DataFrame) -> pd.DataFrame:
    metrics_ = ["neutral_pass_rate", "fourth_down_go_rate"]
    g = g.sort_values(["team", "season", "week", "game_id"]).copy()
    for c in metrics_:
        g[f"pregame_{c}"] = g.groupby(["team", "season"])[c].transform(lambda s: s.expanding().mean().shift(1))
    return g[["game_id", "team", *[f"pregame_{c}" for c in metrics_]]]


def run() -> None:
    pbp = pd.concat([load_pbp(y) for y in range(START, HOLDOUT_END + 1)], ignore_index=True)
    g = prepare(pbp)

    core = build_team_weekly_ratings(build_team_game_features(pbp))
    core_cols = [c for c in core if c.startswith("pregame_")]
    h = core.rename(columns={"team": "home_team", **{c: f"home_{c}" for c in core_cols}})
    a = core.rename(columns={"team": "away_team", **{c: f"away_{c}" for c in core_cols}})
    g = g.merge(h[["game_id", "home_team", *[f"home_{c}" for c in core_cols]]], on=["game_id", "home_team"], how="left")
    g = g.merge(a[["game_id", "away_team", *[f"away_{c}" for c in core_cols]]], on=["game_id", "away_team"], how="left")
    g = g.merge(build_schedule_context(pbp)[["game_id", "rest_diff"]], on="game_id", how="left")

    br = build_pregame_ratings(build_behavior_games(pbp))
    behavior_cols = [c for c in br.columns if c.startswith("pregame_")]
    bh = br.rename(columns={"team": "home_team", **{c: f"home_{c}" for c in behavior_cols}})
    ba = br.rename(columns={"team": "away_team", **{c: f"away_{c}" for c in behavior_cols}})
    g = g.merge(bh[["game_id", "home_team", *[f"home_{c}" for c in behavior_cols]]], on=["game_id", "home_team"], how="left")
    g = g.merge(ba[["game_id", "away_team", *[f"away_{c}" for c in behavior_cols]]], on=["game_id", "away_team"], how="left")

    score = ["home_ppa5", "away_ppa5", "home_dpa3", "away_dpa3"]
    promoted = score + cols(core_cols, ["epa_per_play"]) + cols(core_cols, ["explosive_rate"]) + ["rest_diff"]
    pass_rate = [f"{s}_pregame_neutral_pass_rate" for s in ("home", "away")]
    fourth = [f"{s}_pregame_fourth_down_go_rate" for s in ("home", "away")]
    sets = {
        "PROMOTED_V12_REST": promoted,
        "NEUTRAL_PASS_RATE": promoted + pass_rate,
        "FOURTH_DOWN_AGGRESSIVENESS": promoted + fourth,
        "COMBINED_COACHING_BEHAVIOR": promoted + pass_rate + fourth,
    }

    allp = {k: [] for k in sets}
    print("=== V12 COACHING / AGGRESSIVENESS EXPERIMENT ===")
    for y in TEST_SEASONS:
        tr = g[g.season.lt(y)].dropna(subset=["actual_total", "total_line"])
        te = g[g.season.eq(y)].dropna(subset=["actual_total", "total_line"])
        control = None
        print(f"\n=== {y} ===")
        for name, fs in sets.items():
            pred = model(tr, te, fs)
            m = metrics(te.actual_total, pred)
            if name == "PROMOTED_V12_REST": control = m
            row = dict(m)
            if name != "PROMOTED_V12_REST":
                row.update(mae_vs_control=round(m["mae"] - control["mae"], 4), rmse_vs_control=round(m["rmse"] - control["rmse"], 4))
            print(name, row)
            z = te[["actual_total"]].copy(); z["pred"] = pred; allp[name].append(z)

    print("\n=== OVERALL 2019-2025 WALK-FORWARD ===")
    control = None
    for name, zs in allp.items():
        z = pd.concat(zs); m = metrics(z.actual_total, z.pred)
        if name == "PROMOTED_V12_REST": control = m
        row = dict(m)
        if name != "PROMOTED_V12_REST":
            row.update(mae_vs_control=round(m["mae"] - control["mae"], 4), rmse_vs_control=round(m["rmse"] - control["rmse"], 4))
        print(name, row)


if __name__ == "__main__":
    run()
