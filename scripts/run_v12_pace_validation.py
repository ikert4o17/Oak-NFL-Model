"""Test leakage-safe pace features against the frozen V12 totals core."""
from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd

from oak_nfl.data.nflverse import load_pbp
from oak_nfl.features import build_team_game_features, build_team_weekly_ratings
from run_v12_core_shootout import cols
from run_v12_incremental import HOLDOUT_END, START, metrics, model, prepare

TEST_SEASONS = range(2019, 2026)


def team_game_pace(pbp: pd.DataFrame) -> pd.DataFrame:
    """Estimate offensive pace from time elapsed between valid offensive snaps."""
    d = pbp.copy()
    d = d[d["posteam"].notna() & d["game_id"].notna()]
    if "play_type" in d:
        d = d[~d["play_type"].isin(["no_play", "kickoff", "extra_point", "field_goal"])]
    d = d[d["game_seconds_remaining"].notna()].sort_values(["game_id", "play_id"])
    d["next_gsr"] = d.groupby("game_id")["game_seconds_remaining"].shift(-1)
    d["next_team"] = d.groupby("game_id")["posteam"].shift(-1)
    d["snap_seconds"] = d["game_seconds_remaining"] - d["next_gsr"]
    valid = d["next_team"].eq(d["posteam"]) & d["snap_seconds"].between(5, 45)
    d["pace_seconds"] = d["snap_seconds"].where(valid)
    score_diff = d["score_differential"].abs() if "score_differential" in d else pd.Series(np.nan, index=d.index)
    neutral = valid & d["qtr"].le(3) & score_diff.le(8)
    d["neutral_pace_seconds"] = d["snap_seconds"].where(neutral)
    out = d.groupby(["game_id", "season", "week", "posteam"], as_index=False).agg(
        seconds_per_play=("pace_seconds", "mean"),
        neutral_seconds_per_play=("neutral_pace_seconds", "mean"),
        offensive_plays=("play_id", "count"),
    )
    return out.rename(columns={"posteam": "team"})


def pregame_pace(team_games: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """Emit rolling team pace using completed prior games only."""
    hist = defaultdict(list)
    rows = []
    for r in team_games.sort_values(["season", "week", "game_id", "team"]).itertuples(index=False):
        h = hist[r.team][-window:]
        rows.append({
            "game_id": r.game_id,
            "team": r.team,
            "pregame_seconds_per_play": np.nanmean([x[0] for x in h]) if h else np.nan,
            "pregame_neutral_seconds_per_play": np.nanmean([x[1] for x in h]) if h else np.nan,
            "pregame_offensive_plays": np.nanmean([x[2] for x in h]) if h else np.nan,
        })
        hist[r.team].append((r.seconds_per_play, r.neutral_seconds_per_play, r.offensive_plays))
    return pd.DataFrame(rows)


def run():
    pbp = pd.concat([load_pbp(y) for y in range(START, HOLDOUT_END + 1)], ignore_index=True)
    g = prepare(pbp)
    ratings = build_team_weekly_ratings(build_team_game_features(pbp))
    core = [c for c in ratings.columns if c.startswith("pregame_")]
    home = ratings.rename(columns={"team": "home_team", **{c: f"home_{c}" for c in core}})
    away = ratings.rename(columns={"team": "away_team", **{c: f"away_{c}" for c in core}})
    g = g.merge(home[["game_id", "home_team", *[f"home_{c}" for c in core]]], on=["game_id", "home_team"], how="left")
    g = g.merge(away[["game_id", "away_team", *[f"away_{c}" for c in core]]], on=["game_id", "away_team"], how="left")
    locked = ["home_ppa5", "away_ppa5", "home_dpa3", "away_dpa3"] + cols(core, ["epa_per_play"]) + cols(core, ["explosive_rate"])

    pace = pregame_pace(team_game_pace(pbp))
    pcols = ["pregame_seconds_per_play", "pregame_neutral_seconds_per_play", "pregame_offensive_plays"]
    ph = pace.rename(columns={"team": "home_team", **{c: f"home_{c}" for c in pcols}})
    pa = pace.rename(columns={"team": "away_team", **{c: f"away_{c}" for c in pcols}})
    g = g.merge(ph, on=["game_id", "home_team"], how="left").merge(pa, on=["game_id", "away_team"], how="left")
    all_pace = [f"{s}_{c}" for s in ["home", "away"] for c in pcols]
    sets = {
        "V12_LOCKED": locked,
        "V12_PLUS_RAW_PACE": locked + [f"{s}_pregame_seconds_per_play" for s in ["home", "away"]],
        "V12_PLUS_NEUTRAL_PACE": locked + [f"{s}_pregame_neutral_seconds_per_play" for s in ["home", "away"]],
        "V12_PLUS_PLAY_VOLUME": locked + [f"{s}_pregame_offensive_plays" for s in ["home", "away"]],
        "V12_PLUS_ALL_PACE": locked + all_pace,
    }
    print("=== V12 PACE VALIDATION ===")
    for name, features in sets.items():
        rows = []
        for y in TEST_SEASONS:
            tr = g[g.season.lt(y)].dropna(subset=["actual_total", "total_line"])
            te = g[g.season.eq(y)].dropna(subset=["actual_total", "total_line"]).copy()
            te["pred"] = model(tr, te, features)
            rows.append(te[["season", "actual_total", "total_line", "pred"]])
        x = pd.concat(rows)
        mk = metrics(x.actual_total, x.total_line)
        md = metrics(x.actual_total, x.pred)
        print(name, {"games": len(x), "mae": md["mae"], "rmse": md["rmse"], "mae_delta": md["mae"] - mk["mae"], "rmse_delta": md["rmse"] - mk["rmse"]})
        for y, z in x.groupby("season"):
            ym = metrics(z.actual_total, z.total_line)
            yd = metrics(z.actual_total, z.pred)
            print(" ", y, {"mae_delta": round(yd["mae"] - ym["mae"], 4), "rmse_delta": round(yd["rmse"] - ym["rmse"], 4)})


if __name__ == "__main__":
    run()
