"""Phase 2 totals integration: test weather on top of V12 + QB/personnel."""
from __future__ import annotations

import pandas as pd

from oak_nfl.data.injuries import latest_weekly_status, normalize_injury_feed
from oak_nfl.data.nflverse import load_injuries, load_pbp, load_players, load_snap_counts
from oak_nfl.features import build_team_game_features, build_team_weekly_ratings
from oak_nfl.personnel import player_absence_points
from oak_nfl.personnel_value import attach_player_values, attach_snap_player_ids, build_pregame_player_values
from oak_nfl.qb import build_pregame_qb_ratings
from run_v12_core_shootout import cols
from run_v12_incremental import HOLDOUT_END, START, metrics, model, prepare
from run_v12_weather_validation import game_weather

TEST_SEASONS = range(2023, 2025)  # reliable historical injury overlap


def canonical_injuries(raw):
    d = raw.loc[raw.report_status.notna() & raw.report_status.astype(str).str.strip().ne("")].copy()
    return latest_weekly_status(normalize_injury_feed(
        d,
        column_map={
            "gsis_id": "player_id",
            "full_name": "player_name",
            "position": "position_group",
            "report_status": "status",
            "date_modified": "report_date",
        },
        source="nflverse_injuries",
    ))


def personnel_team_week(availability):
    f = availability[availability.position_group.ne("QB")].copy()
    f["absence_points"] = [
        0.5 * player_absence_points(p, v, s)
        for p, v, s in zip(f.position_group, f.player_value, f.status)
    ]
    out = (
        f.groupby(["season", "week", "team"], as_index=False).absence_points.sum()
        .rename(columns={"absence_points": "personnel_points"})
    )
    out["personnel_points"] = out.personnel_points.clip(-2, 2)
    return out


def run():
    pbp = pd.concat([load_pbp(y) for y in range(START, HOLDOUT_END + 1)], ignore_index=True)
    games = prepare(pbp).merge(game_weather(pbp), on="game_id", how="left")

    ratings = build_team_weekly_ratings(build_team_game_features(pbp))
    core = [c for c in ratings.columns if c.startswith("pregame_")]
    home = ratings.rename(columns={"team": "home_team", **{c: f"home_{c}" for c in core}})
    away = ratings.rename(columns={"team": "away_team", **{c: f"away_{c}" for c in core}})
    games = games.merge(
        home[["game_id", "home_team", *[f"home_{c}" for c in core]]],
        on=["game_id", "home_team"], how="left",
    ).merge(
        away[["game_id", "away_team", *[f"away_{c}" for c in core]]],
        on=["game_id", "away_team"], how="left",
    )
    locked = ["home_ppa5", "away_ppa5", "home_dpa3", "away_dpa3"] + cols(
        core, ["epa_per_play"]
    ) + cols(core, ["explosive_rate"])

    qb = build_pregame_qb_ratings(pbp)
    qcols = ["pregame_qb_epa", "pregame_qb_cpoe", "pregame_qb_sack_rate"]
    qh = qb[["game_id", "team", *qcols]].rename(
        columns={"team": "home_team", **{c: f"home_{c}" for c in qcols}}
    )
    qa = qb[["game_id", "team", *qcols]].rename(
        columns={"team": "away_team", **{c: f"away_{c}" for c in qcols}}
    )
    games = games.merge(qh, on=["game_id", "home_team"], how="left").merge(
        qa, on=["game_id", "away_team"], how="left"
    )
    qb_features = [f"{side}_{c}" for side in ["home", "away"] for c in qcols]

    snaps = pd.concat([load_snap_counts(y) for y in range(START, 2025)], ignore_index=True)
    injuries = pd.concat([load_injuries(y) for y in range(START, 2025)], ignore_index=True)
    players = load_players()
    values = build_pregame_player_values(attach_snap_player_ids(snaps, players))
    availability = attach_player_values(canonical_injuries(injuries), values)
    personnel = personnel_team_week(availability)
    ph = personnel.rename(columns={"team": "home_team", "personnel_points": "home_personnel"})
    pa = personnel.rename(columns={"team": "away_team", "personnel_points": "away_personnel"})
    games = games.merge(ph, on=["season", "week", "home_team"], how="left").merge(
        pa, on=["season", "week", "away_team"], how="left"
    )
    games[["home_personnel", "away_personnel"]] = games[["home_personnel", "away_personnel"]].fillna(0)
    personnel_features = ["home_personnel", "away_personnel"]

    weather = [
        "temp", "wind", "outdoor", "temp_dev_65", "wind_outdoor", "extreme_temp_outdoor",
        "cold_40", "hot_85", "high_wind_15", "precip",
    ]
    integrated = locked + qb_features + personnel_features
    sets = {
        "V12_LOCKED": locked,
        "V12_PLUS_QB_PERSONNEL": integrated,
        "V12_PLUS_WEATHER": locked + weather,
        "V12_PLUS_QB_PERSONNEL_WEATHER": integrated + weather,
    }

    print("=== V12 TOTALS INTEGRATION PHASE 2 ===")
    for name, features in sets.items():
        preds = []
        for year in TEST_SEASONS:
            train = games[games.season.lt(year)].dropna(subset=["actual_total", "total_line"])
            test = games[games.season.eq(year)].dropna(subset=["actual_total", "total_line"]).copy()
            test["pred"] = model(train, test, features)
            preds.append(test[["season", "actual_total", "total_line", "pred"]])
        result = pd.concat(preds, ignore_index=True)
        market = metrics(result.actual_total, result.total_line)
        scored = metrics(result.actual_total, result.pred)
        print(name, {
            "games": len(result), "mae": scored["mae"], "rmse": scored["rmse"],
            "mae_delta": scored["mae"] - market["mae"],
            "rmse_delta": scored["rmse"] - market["rmse"],
        })
        for year, group in result.groupby("season"):
            mk = metrics(group.actual_total, group.total_line)
            md = metrics(group.actual_total, group.pred)
            print(" ", year, {
                "mae_delta": round(md["mae"] - mk["mae"], 4),
                "rmse_delta": round(md["rmse"] - mk["rmse"], 4),
            })


if __name__ == "__main__":
    run()
