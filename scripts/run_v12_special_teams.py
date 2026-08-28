"""Special-teams / field-position experiment against locked V12.

Predetermined challengers:
- overall special-teams EPA per special-teams play
- field-goal EPA per attempt
- punt EPA per punt
- kickoff EPA per kickoff
- combined special-teams components

All team ratings are season-local expanding means shifted one game so every
feature is strictly pregame. The locked V12 feature set remains the control.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from oak_nfl.data.nflverse import load_pbp
from oak_nfl.features import build_team_game_features, build_team_weekly_ratings
from run_v12_core_shootout import cols
from run_v12_incremental import HOLDOUT_END, START, metrics, model, prepare

TEST_SEASONS = range(2019, 2026)


def _num(p: pd.DataFrame, col: str) -> pd.Series:
    if col not in p.columns:
        return pd.Series(0.0, index=p.index)
    return pd.to_numeric(p[col], errors="coerce").fillna(0.0)


def build_special_team_games(pbp: pd.DataFrame) -> pd.DataFrame:
    p = pbp.copy()
    need = {"game_id", "season", "week", "posteam", "defteam", "epa"}
    missing = need - set(p.columns)
    if missing:
        raise ValueError(f"PBP missing required columns: {sorted(missing)}")

    p = p[p.posteam.notna() & p.defteam.notna()].copy()
    p["epa"] = pd.to_numeric(p["epa"], errors="coerce")

    fg = _num(p, "field_goal_attempt").eq(1)
    punt = _num(p, "punt_attempt").eq(1)
    kickoff = _num(p, "kickoff_attempt").eq(1)
    xp = _num(p, "extra_point_attempt").eq(1)
    st = fg | punt | kickoff | xp

    p["st_play"] = st.astype(float)
    p["fg_play"] = fg.astype(float)
    p["punt_play"] = punt.astype(float)
    p["kickoff_play"] = kickoff.astype(float)

    p["st_epa"] = np.where(st, p["epa"], np.nan)
    p["fg_epa"] = np.where(fg, p["epa"], np.nan)
    p["punt_epa"] = np.where(punt, p["epa"], np.nan)
    p["kickoff_epa"] = np.where(kickoff, p["epa"], np.nan)

    g = (
        p.groupby(["game_id", "season", "week", "posteam", "defteam"], as_index=False)
        .agg(
            st_plays=("st_play", "sum"),
            st_epa=("st_epa", "mean"),
            fg_epa=("fg_epa", "mean"),
            punt_epa=("punt_epa", "mean"),
            kickoff_epa=("kickoff_epa", "mean"),
        )
    )
    return g


def build_ratings(g: pd.DataFrame) -> pd.DataFrame:
    metrics_ = ["st_epa", "fg_epa", "punt_epa", "kickoff_epa"]
    off = g.rename(columns={"posteam": "team"})[
        ["game_id", "season", "week", "team", *metrics_]
    ].rename(columns={m: f"for_{m}" for m in metrics_})
    de = g.rename(columns={"defteam": "team"})[
        ["game_id", "season", "week", "team", *metrics_]
    ].rename(columns={m: f"against_{m}" for m in metrics_})

    c = off.merge(
        de,
        on=["game_id", "season", "week", "team"],
        how="outer",
        validate="one_to_one",
    ).sort_values(["team", "season", "week", "game_id"])

    value_cols = [x for x in c.columns if x.startswith(("for_", "against_"))]
    for col in value_cols:
        c[f"pregame_{col}"] = c.groupby(["team", "season"])[col].transform(
            lambda s: s.expanding().mean().shift(1)
        )
    return c


def run() -> None:
    pbp = pd.concat(
        [load_pbp(y) for y in range(START, HOLDOUT_END + 1)], ignore_index=True
    )
    g = prepare(pbp)

    tg = build_team_game_features(pbp)
    base = build_team_weekly_ratings(tg)
    core = [c for c in base if c.startswith("pregame_")]

    h = base.rename(
        columns={"team": "home_team", **{c: f"home_{c}" for c in core}}
    )
    a = base.rename(
        columns={"team": "away_team", **{c: f"away_{c}" for c in core}}
    )
    g = g.merge(
        h[["game_id", "home_team", *[f"home_{c}" for c in core]]],
        on=["game_id", "home_team"],
        how="left",
    ).merge(
        a[["game_id", "away_team", *[f"away_{c}" for c in core]]],
        on=["game_id", "away_team"],
        how="left",
    )

    r = build_ratings(build_special_team_games(pbp))
    rc = [c for c in r if c.startswith("pregame_")]
    h = r.rename(columns={"team": "home_team", **{c: f"home_{c}" for c in rc}})
    a = r.rename(columns={"team": "away_team", **{c: f"away_{c}" for c in rc}})
    g = g.merge(
        h[["game_id", "home_team", *[f"home_{c}" for c in rc]]],
        on=["game_id", "home_team"],
        how="left",
    ).merge(
        a[["game_id", "away_team", *[f"away_{c}" for c in rc]]],
        on=["game_id", "away_team"],
        how="left",
    )

    score = ["home_ppa5", "away_ppa5", "home_dpa3", "away_dpa3"]
    locked = score + cols(core, ["epa_per_play"]) + cols(core, ["explosive_rate"])
    sets = {"LOCKED_V12": locked}

    for key in ["st_epa", "fg_epa", "punt_epa", "kickoff_epa"]:
        f = []
        for side in ["home", "away"]:
            f += [
                f"{side}_pregame_for_{key}",
                f"{side}_pregame_against_{key}",
            ]
        sets[key.upper()] = locked + f

    combined = []
    for key in ["fg_epa", "punt_epa", "kickoff_epa"]:
        for side in ["home", "away"]:
            combined += [
                f"{side}_pregame_for_{key}",
                f"{side}_pregame_against_{key}",
            ]
    sets["COMBINED_SPECIAL_TEAMS"] = locked + combined

    allp = {k: [] for k in sets}
    print("=== V12 SPECIAL TEAMS EXPERIMENT ===")
    for y in TEST_SEASONS:
        tr = g[g.season.lt(y)].dropna(subset=["actual_total", "total_line"])
        te = g[g.season.eq(y)].dropna(subset=["actual_total", "total_line"])
        control = None
        print(f"\n=== {y} ===")
        for name, fs in sets.items():
            p = model(tr, te, fs)
            m = metrics(te.actual_total, p)
            if name == "LOCKED_V12":
                control = m
            row = dict(m)
            if name != "LOCKED_V12":
                row.update(
                    mae_vs_locked=round(m["mae"] - control["mae"], 4),
                    rmse_vs_locked=round(m["rmse"] - control["rmse"], 4),
                )
            print(name, row)
            z = te[["actual_total"]].copy()
            z["pred"] = p
            allp[name].append(z)

    print("\n=== OVERALL 2019-2025 WALK-FORWARD ===")
    cm = None
    for name, zs in allp.items():
        z = pd.concat(zs)
        m = metrics(z.actual_total, z.pred)
        if name == "LOCKED_V12":
            cm = m
        row = dict(m)
        if name != "LOCKED_V12":
            row.update(
                mae_vs_locked=round(m["mae"] - cm["mae"], 4),
                rmse_vs_locked=round(m["rmse"] - cm["rmse"], 4),
            )
        print(name, row)


if __name__ == "__main__":
    run()
