"""Explicit starting-field-position experiment against promoted V12 + rest differential.

Predetermined challengers:
- average offensive starting field position
- average defensive/opponent starting field position
- short-field possession rate (drive starts at opponent 40 or better)
- combined field-position package

All ratings are season-to-date expanding means shifted one game, so only prior-game
information is used. No threshold search: opponent 40 is the single conventional
short-field definition fixed before the benchmark.
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


def build_field_position_games(pbp: pd.DataFrame) -> pd.DataFrame:
    required = {"game_id", "season", "week", "posteam", "defteam", "drive", "yardline_100"}
    missing = sorted(required.difference(pbp.columns))
    if missing:
        raise ValueError(f"PBP missing field-position columns: {missing}")

    p = pbp[list(required)].copy()
    p["yardline_100"] = pd.to_numeric(p["yardline_100"], errors="coerce")
    p = p[p.posteam.notna() & p.defteam.notna() & p.drive.notna() & p.yardline_100.notna()]
    p = p.sort_index()

    # First valid scrimmage location observed for each possession/drive. Convert
    # yardline_100 (distance to opponent goal) to yards from own goal line.
    d = p.groupby(["game_id", "season", "week", "posteam", "defteam", "drive"], as_index=False).first()
    d["start_field_pos"] = 100.0 - d["yardline_100"]
    d["short_field"] = d["yardline_100"].le(40).astype(float)

    off = d.groupby(["game_id", "season", "week", "posteam"], as_index=False).agg(
        start_field_pos=("start_field_pos", "mean"),
        short_field_rate=("short_field", "mean"),
    ).rename(columns={"posteam": "team"})

    against = d.groupby(["game_id", "season", "week", "defteam"], as_index=False).agg(
        opp_start_field_pos=("start_field_pos", "mean"),
        opp_short_field_rate=("short_field", "mean"),
    ).rename(columns={"defteam": "team"})

    return off.merge(against, on=["game_id", "season", "week", "team"], how="outer")


def build_ratings(g: pd.DataFrame) -> pd.DataFrame:
    metrics_ = ["start_field_pos", "opp_start_field_pos", "short_field_rate", "opp_short_field_rate"]
    g = g.sort_values(["team", "season", "week", "game_id"]).copy()
    for c in metrics_:
        g[f"pregame_{c}"] = g.groupby(["team", "season"])[c].transform(
            lambda s: s.expanding().mean().shift(1)
        )
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

    fp = build_ratings(build_field_position_games(pbp))
    fp_cols = [c for c in fp.columns if c.startswith("pregame_")]
    fh = fp.rename(columns={"team": "home_team", **{c: f"home_{c}" for c in fp_cols}})
    fa = fp.rename(columns={"team": "away_team", **{c: f"away_{c}" for c in fp_cols}})
    g = g.merge(fh[["game_id", "home_team", *[f"home_{c}" for c in fp_cols]]], on=["game_id", "home_team"], how="left")
    g = g.merge(fa[["game_id", "away_team", *[f"away_{c}" for c in fp_cols]]], on=["game_id", "away_team"], how="left")

    score = ["home_ppa5", "away_ppa5", "home_dpa3", "away_dpa3"]
    promoted = score + cols(core_cols, ["epa_per_play"]) + cols(core_cols, ["explosive_rate"]) + ["rest_diff"]
    start = [f"{s}_pregame_{c}" for s in ("home", "away") for c in ("start_field_pos", "opp_start_field_pos")]
    short = [f"{s}_pregame_{c}" for s in ("home", "away") for c in ("short_field_rate", "opp_short_field_rate")]
    sets = {
        "PROMOTED_V12_REST": promoted,
        "START_FIELD_POSITION": promoted + start,
        "SHORT_FIELD_RATE": promoted + short,
        "COMBINED_FIELD_POSITION": promoted + start + short,
    }

    allp = {k: [] for k in sets}
    print("=== V12 EXPLICIT FIELD POSITION EXPERIMENT ===")
    for y in TEST_SEASONS:
        tr = g[g.season.lt(y)].dropna(subset=["actual_total", "total_line"])
        te = g[g.season.eq(y)].dropna(subset=["actual_total", "total_line"])
        control = None
        print(f"\n=== {y} ===")
        for name, fs in sets.items():
            pred = model(tr, te, fs)
            m = metrics(te.actual_total, pred)
            if name == "PROMOTED_V12_REST":
                control = m
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
