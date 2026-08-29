"""Pressure/disruption challengers against promoted V12 + rest.

Frozen design: pregame season-to-date sack rate and QB-hit rate, with offense
measuring pressure allowed and defense measuring pressure generated. No tuning.
"""
from __future__ import annotations

import pandas as pd

from oak_nfl.data.nflverse import load_pbp
from oak_nfl.features import build_team_game_features, build_team_weekly_ratings
from run_v12_core_shootout import cols
from run_v12_incremental import HOLDOUT_END, START, metrics, model, prepare
from run_v12_schedule_context import build_schedule_context

TEST_SEASONS = range(2019, 2026)


def build_pressure_games(pbp: pd.DataFrame) -> pd.DataFrame:
    required = {"game_id", "season", "week", "posteam", "defteam", "pass", "sack", "qb_hit"}
    missing = required.difference(pbp.columns)
    if missing:
        raise ValueError(f"play-by-play data missing pressure columns: {sorted(missing)}")

    x = pbp[pbp["posteam"].notna() & pbp["defteam"].notna() & pbp["pass"].eq(1)].copy()
    x["sack_flag"] = pd.to_numeric(x["sack"], errors="coerce").fillna(0).clip(0, 1)
    x["qb_hit_flag"] = pd.to_numeric(x["qb_hit"], errors="coerce").fillna(0).clip(0, 1)
    return (
        x.groupby(["game_id", "season", "week", "posteam", "defteam"], sort=True)
        .agg(sack_rate=("sack_flag", "mean"), qb_hit_rate=("qb_hit_flag", "mean"))
        .reset_index()
    )


def build_pressure_ratings(pg: pd.DataFrame) -> pd.DataFrame:
    off = pg[["game_id", "season", "week", "posteam", "sack_rate", "qb_hit_rate"]].rename(
        columns={"posteam": "team", "sack_rate": "off_sack_rate_allowed", "qb_hit_rate": "off_qb_hit_rate_allowed"}
    )
    de = pg[["game_id", "season", "week", "defteam", "sack_rate", "qb_hit_rate"]].rename(
        columns={"defteam": "team", "sack_rate": "def_sack_rate_generated", "qb_hit_rate": "def_qb_hit_rate_generated"}
    )
    z = off.merge(de, on=["game_id", "season", "week", "team"], how="outer").sort_values(
        ["team", "season", "week", "game_id"]
    )
    metrics_ = ["off_sack_rate_allowed", "off_qb_hit_rate_allowed", "def_sack_rate_generated", "def_qb_hit_rate_generated"]
    for c in metrics_:
        z[f"pregame_{c}"] = z.groupby(["team", "season"])[c].transform(lambda s: s.expanding().mean().shift(1))
    return z[["game_id", "team", *[f"pregame_{c}" for c in metrics_]]]


def run() -> None:
    pbp = pd.concat([load_pbp(y) for y in range(START, HOLDOUT_END + 1)], ignore_index=True)
    g = prepare(pbp)
    core_games = build_team_game_features(pbp)
    core = build_team_weekly_ratings(core_games)
    pressure = build_pressure_ratings(build_pressure_games(pbp))

    core_cols = [c for c in core.columns if c.startswith("pregame_")]
    pcols = [c for c in pressure.columns if c.startswith("pregame_")]
    ratings = core.merge(pressure, on=["game_id", "team"], how="left")
    all_cols = core_cols + pcols
    h = ratings.rename(columns={"team": "home_team", **{c: f"home_{c}" for c in all_cols}})
    a = ratings.rename(columns={"team": "away_team", **{c: f"away_{c}" for c in all_cols}})
    g = g.merge(h[["game_id", "home_team", *[f"home_{c}" for c in all_cols]]], on=["game_id", "home_team"], how="left")
    g = g.merge(a[["game_id", "away_team", *[f"away_{c}" for c in all_cols]]], on=["game_id", "away_team"], how="left")
    g = g.merge(build_schedule_context(pbp)[["game_id", "rest_diff"]], on="game_id", how="left")

    score = ["home_ppa5", "away_ppa5", "home_dpa3", "away_dpa3"]
    promoted = score + cols(core_cols, ["epa_per_play"]) + cols(core_cols, ["explosive_rate"]) + ["rest_diff"]
    sack_cols = [f"{s}_pregame_{c}" for s in ("home", "away") for c in ("off_sack_rate_allowed", "def_sack_rate_generated")]
    hit_cols = [f"{s}_pregame_{c}" for s in ("home", "away") for c in ("off_qb_hit_rate_allowed", "def_qb_hit_rate_generated")]
    sets = {
        "PROMOTED_V12_REST": promoted,
        "SACK_RATE": promoted + sack_cols,
        "QB_HIT_RATE": promoted + hit_cols,
        "COMBINED_PRESSURE": promoted + sack_cols + hit_cols,
    }

    allp = {k: [] for k in sets}
    print("=== V12 PRESSURE / DISRUPTION ===")
    for y in TEST_SEASONS:
        tr = g[g.season.lt(y)].dropna(subset=["actual_total", "total_line"])
        te = g[g.season.eq(y)].dropna(subset=["actual_total", "total_line"])
        control = None
        print(f"\n=== {y} ===")
        for name, fs in sets.items():
            pred = model(tr, te, fs); m = metrics(te.actual_total, pred)
            if name == "PROMOTED_V12_REST": control = m
            row = dict(m)
            if name != "PROMOTED_V12_REST":
                row.update(mae_vs_control=round(m["mae"]-control["mae"],4), rmse_vs_control=round(m["rmse"]-control["rmse"],4))
            print(name, row)
            z = te[["actual_total"]].copy(); z["pred"] = pred; allp[name].append(z)

    print("\n=== OVERALL 2019-2025 WALK-FORWARD ===")
    control = None
    for name, zs in allp.items():
        z = pd.concat(zs); m = metrics(z.actual_total, z.pred)
        if name == "PROMOTED_V12_REST": control = m
        row = dict(m)
        if name != "PROMOTED_V12_REST":
            row.update(mae_vs_control=round(m["mae"]-control["mae"],4), rmse_vs_control=round(m["rmse"]-control["rmse"],4))
        print(name, row)


if __name__ == "__main__":
    run()
