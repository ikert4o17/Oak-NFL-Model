"""Turnover-process experiment against locked V12.

Predetermined candidates: raw turnover rate, interception rate, fumble rate,
fumble recovery luck, and a combined turnover-process feature set. All ratings
are season-local expanding means shifted one game to remain pregame-only.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from oak_nfl.data.nflverse import load_pbp
from oak_nfl.features import build_team_game_features, build_team_weekly_ratings
from run_v12_core_shootout import cols
from run_v12_incremental import HOLDOUT_END, START, metrics, model, prepare

TEST_SEASONS = range(2019, 2026)


def build_turnover_games(pbp: pd.DataFrame) -> pd.DataFrame:
    need = {
        "game_id", "season", "week", "posteam", "defteam", "play",
        "pass_attempt", "interception", "fumble", "fumble_lost",
    }
    missing = need - set(pbp.columns)
    if missing:
        raise ValueError(f"PBP missing turnover columns: {sorted(missing)}")

    p = pbp[pbp.posteam.notna() & pbp.defteam.notna()].copy()
    p["off_play"] = pd.to_numeric(p["play"], errors="coerce").fillna(0).gt(0).astype(float)
    p["pass_attempt_n"] = pd.to_numeric(p["pass_attempt"], errors="coerce").fillna(0).clip(0, 1)
    p["interception_n"] = pd.to_numeric(p["interception"], errors="coerce").fillna(0).clip(0, 1)
    p["fumble_n"] = pd.to_numeric(p["fumble"], errors="coerce").fillna(0).clip(0, 1)
    p["fumble_lost_n"] = pd.to_numeric(p["fumble_lost"], errors="coerce").fillna(0).clip(0, 1)

    g = p.groupby(["game_id", "season", "week", "posteam", "defteam"], as_index=False).agg(
        plays=("off_play", "sum"),
        pass_attempts=("pass_attempt_n", "sum"),
        interceptions=("interception_n", "sum"),
        fumbles=("fumble_n", "sum"),
        fumbles_lost=("fumble_lost_n", "sum"),
    )
    g["turnover_rate"] = (g.interceptions + g.fumbles_lost) / g.plays.replace(0, np.nan)
    g["interception_rate"] = g.interceptions / g.pass_attempts.replace(0, np.nan)
    g["fumble_rate"] = g.fumbles / g.plays.replace(0, np.nan)
    g["fumble_recovery_luck"] = (
        (g.fumbles - g.fumbles_lost) / g.fumbles.replace(0, np.nan) - 0.5
    ).fillna(0.0)
    return g


def build_ratings(tg: pd.DataFrame) -> pd.DataFrame:
    ms = ["turnover_rate", "interception_rate", "fumble_rate", "fumble_recovery_luck"]
    off = tg.rename(columns={"posteam": "team"})[["game_id", "season", "week", "team", *ms]].rename(
        columns={m: f"off_{m}" for m in ms}
    )
    de = tg.rename(columns={"defteam": "team"})[["game_id", "season", "week", "team", *ms]].rename(
        columns={m: f"def_{m}_forced" for m in ms}
    )
    r = off.merge(de, on=["game_id", "season", "week", "team"], how="outer", validate="one_to_one")
    r = r.sort_values(["team", "season", "week", "game_id"])
    for c in [x for x in r.columns if x.startswith(("off_", "def_"))]:
        r[f"pregame_{c}"] = r.groupby(["team", "season"])[c].transform(
            lambda s: s.expanding().mean().shift(1)
        )
    return r


def run() -> None:
    pbp = pd.concat([load_pbp(y) for y in range(START, HOLDOUT_END + 1)], ignore_index=True)
    g = prepare(pbp)

    team_games = build_team_game_features(pbp)
    base = build_team_weekly_ratings(team_games)
    core = [c for c in base.columns if c.startswith("pregame_")]
    h = base.rename(columns={"team": "home_team", **{c: f"home_{c}" for c in core}})
    a = base.rename(columns={"team": "away_team", **{c: f"away_{c}" for c in core}})
    g = g.merge(h[["game_id", "home_team", *[f"home_{c}" for c in core]]], on=["game_id", "home_team"], how="left")
    g = g.merge(a[["game_id", "away_team", *[f"away_{c}" for c in core]]], on=["game_id", "away_team"], how="left")

    tr = build_ratings(build_turnover_games(pbp))
    tc = [c for c in tr.columns if c.startswith("pregame_")]
    h = tr.rename(columns={"team": "home_team", **{c: f"home_{c}" for c in tc}})
    a = tr.rename(columns={"team": "away_team", **{c: f"away_{c}" for c in tc}})
    g = g.merge(h[["game_id", "home_team", *[f"home_{c}" for c in tc]]], on=["game_id", "home_team"], how="left")
    g = g.merge(a[["game_id", "away_team", *[f"away_{c}" for c in tc]]], on=["game_id", "away_team"], how="left")

    score = ["home_ppa5", "away_ppa5", "home_dpa3", "away_dpa3"]
    locked = score + cols(core, ["epa_per_play"]) + cols(core, ["explosive_rate"])
    sets = {"LOCKED_V12": locked}

    def turnover_cols(key: str) -> list[str]:
        out = []
        for side in ["home", "away"]:
            out += [f"{side}_pregame_off_{key}", f"{side}_pregame_def_{key}_forced"]
        return out

    for key in ["turnover_rate", "interception_rate", "fumble_rate", "fumble_recovery_luck"]:
        sets[key.upper()] = locked + turnover_cols(key)
    sets["COMBINED_PROCESS"] = locked + sum(
        [turnover_cols(k) for k in ["interception_rate", "fumble_rate", "fumble_recovery_luck"]], []
    )

    all_preds = {k: [] for k in sets}
    print("=== V12 TURNOVER PROCESS EXPERIMENT ===")
    for y in TEST_SEASONS:
        train = g[g.season.lt(y)].dropna(subset=["actual_total", "total_line"])
        test = g[g.season.eq(y)].dropna(subset=["actual_total", "total_line"])
        control = None
        print(f"\n=== {y} ===")
        for name, fs in sets.items():
            pred = model(train, test, fs)
            m = metrics(test.actual_total, pred)
            if name == "LOCKED_V12":
                control = m
            row = dict(m)
            if name != "LOCKED_V12":
                row.update(
                    mae_vs_locked=round(m["mae"] - control["mae"], 4),
                    rmse_vs_locked=round(m["rmse"] - control["rmse"], 4),
                )
            print(name, row)
            z = test[["actual_total"]].copy()
            z["pred"] = pred
            all_preds[name].append(z)

    print("\n=== OVERALL 2019-2025 WALK-FORWARD ===")
    control = None
    for name, zs in all_preds.items():
        z = pd.concat(zs)
        m = metrics(z.actual_total, z.pred)
        if name == "LOCKED_V12":
            control = m
        row = dict(m)
        if name != "LOCKED_V12":
            row.update(
                mae_vs_locked=round(m["mae"] - control["mae"], 4),
                rmse_vs_locked=round(m["rmse"] - control["rmse"], 4),
            )
        print(name, row)


if __name__ == "__main__":
    run()
