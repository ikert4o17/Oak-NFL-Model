"""Validate weather/environment features against the locked V12 totals model."""
from __future__ import annotations

import numpy as np
import pandas as pd

from oak_nfl.data.nflverse import load_pbp
from oak_nfl.features import build_team_game_features, build_team_weekly_ratings
from run_v12_core_shootout import cols
from run_v12_incremental import HOLDOUT_END, START, metrics, model, prepare

TEST_SEASONS = range(2019, 2026)


def game_weather(pbp: pd.DataFrame) -> pd.DataFrame:
    base_cols = ["game_id"]
    optional = [c for c in ["temp", "wind", "roof", "weather"] if c in pbp.columns]
    if not optional:
        raise ValueError("nflverse PBP does not expose expected weather fields")

    frame = pbp[base_cols + optional].drop_duplicates("game_id", keep="last").copy()
    frame["temp"] = pd.to_numeric(frame.get("temp"), errors="coerce")
    frame["wind"] = pd.to_numeric(frame.get("wind"), errors="coerce")
    roof = frame.get("roof", pd.Series(index=frame.index, dtype=object)).fillna("").astype(str).str.lower()
    text = frame.get("weather", pd.Series(index=frame.index, dtype=object)).fillna("").astype(str).str.lower()

    frame["outdoor"] = (~roof.isin(["dome", "closed"])) .astype(float)
    frame["temp_dev_65"] = (frame["temp"] - 65.0).abs()
    frame["cold_40"] = (frame["temp"] <= 40).astype(float)
    frame["hot_85"] = (frame["temp"] >= 85).astype(float)
    frame["high_wind_15"] = (frame["wind"] >= 15).astype(float)
    frame["precip"] = text.str.contains("rain|snow|shower|storm|sleet", regex=True).astype(float)
    frame["wind_outdoor"] = frame["wind"] * frame["outdoor"]
    frame["extreme_temp_outdoor"] = frame["temp_dev_65"] * frame["outdoor"]
    return frame


def run():
    pbp = pd.concat([load_pbp(y) for y in range(START, HOLDOUT_END + 1)], ignore_index=True)
    games = prepare(pbp).merge(game_weather(pbp), on="game_id", how="left")

    ratings = build_team_weekly_ratings(build_team_game_features(pbp))
    core = [c for c in ratings.columns if c.startswith("pregame_")]
    home = ratings.rename(columns={"team": "home_team", **{c: f"home_{c}" for c in core}})
    away = ratings.rename(columns={"team": "away_team", **{c: f"away_{c}" for c in core}})
    games = games.merge(
        home[["game_id", "home_team", *[f"home_{c}" for c in core]]],
        on=["game_id", "home_team"],
        how="left",
    ).merge(
        away[["game_id", "away_team", *[f"away_{c}" for c in core]]],
        on=["game_id", "away_team"],
        how="left",
    )

    locked = ["home_ppa5", "away_ppa5", "home_dpa3", "away_dpa3"] + cols(
        core, ["epa_per_play"]
    ) + cols(core, ["explosive_rate"])
    continuous = ["temp", "wind", "outdoor", "temp_dev_65", "wind_outdoor", "extreme_temp_outdoor"]
    thresholds = ["cold_40", "hot_85", "high_wind_15", "precip", "outdoor"]
    sets = {
        "V12_LOCKED": locked,
        "V12_PLUS_CONTINUOUS_WEATHER": locked + continuous,
        "V12_PLUS_WEATHER_THRESHOLDS": locked + thresholds,
        "V12_PLUS_ALL_WEATHER": locked + list(dict.fromkeys(continuous + thresholds)),
    }

    print("=== V12 WEATHER / ENVIRONMENT VALIDATION ===")
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
            "games": len(result),
            "mae": scored["mae"],
            "rmse": scored["rmse"],
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

    print("\n=== EXTREME WEATHER COUNTS ===")
    sample = games[games.season.between(2019, 2025)]
    for field in ["cold_40", "hot_85", "high_wind_15", "precip"]:
        print(field, int(sample[field].fillna(0).sum()))


if __name__ == "__main__":
    run()
