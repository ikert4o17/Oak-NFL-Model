"""Benchmark situational efficiency features against frozen Oak V5."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from oak_nfl.backtest import evaluate_margin_predictions
from oak_nfl.data.games import build_game_results
from oak_nfl.data.nflverse import load_pbp
from oak_nfl.features import build_team_game_features
from oak_nfl.features_situational import build_situational_team_game_features
from oak_nfl.ratings.v5 import build_v5_game_predictions, build_v5_pregame_ratings

SITUATIONAL = [
    "early_down_epa_per_play", "early_down_success_rate",
    "third_down_epa_per_play", "third_down_success_rate",
    "red_zone_epa_per_play", "red_zone_success_rate", "sack_rate",
]


def _pregame(team_games: pd.DataFrame) -> pd.DataFrame:
    offense = team_games.rename(columns={"posteam": "team"})
    defense = team_games.rename(columns={"defteam": "team"})
    rows = []
    for side, frame in [("off", offense), ("def", defense)]:
        keep = ["season", "week", "game_id", "team", *SITUATIONAL]
        frame = frame[keep].sort_values(["team", "season", "week", "game_id"]).copy()
        for metric in SITUATIONAL:
            frame[f"pregame_{side}_{metric}"] = frame.groupby(["team", "season"])[metric].transform(
                lambda x: x.expanding().mean().shift(1)
            )
        rows.append(frame[["season", "week", "game_id", "team", *[f"pregame_{side}_{m}" for m in SITUATIONAL]]])
    return rows[0].merge(rows[1], on=["season", "week", "game_id", "team"], how="outer")


def _gaps(games: pd.DataFrame, ratings: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [c for c in ratings.columns if c.startswith("pregame_")]
    home = ratings.rename(columns={"team": "home_team", **{c: f"home_{c}" for c in metric_cols}})
    away = ratings.rename(columns={"team": "away_team", **{c: f"away_{c}" for c in metric_cols}})
    frame = games.merge(home, on=["season", "week", "game_id", "home_team"], how="left")
    frame = frame.merge(away, on=["season", "week", "game_id", "away_team"], how="left")
    for metric in SITUATIONAL:
        frame[f"{metric}_gap"] = (
            frame[f"home_pregame_off_{metric}"] - frame[f"home_pregame_def_{metric}"]
            - frame[f"away_pregame_off_{metric}"] + frame[f"away_pregame_def_{metric}"]
        )
    return frame


def run() -> None:
    pbp = pd.concat([load_pbp(y) for y in range(2014, 2026)], ignore_index=True)
    base_features = build_team_game_features(pbp)
    situational = build_situational_team_game_features(pbp)
    games = build_game_results(pbp)
    v5 = build_v5_game_predictions(games, build_v5_pregame_ratings(base_features))
    frame = _gaps(games, _pregame(situational))
    feature_sets = {
        "early_down": ["early_down_epa_per_play_gap", "early_down_success_rate_gap"],
        "third_down": ["third_down_epa_per_play_gap", "third_down_success_rate_gap"],
        "red_zone": ["red_zone_epa_per_play_gap", "red_zone_success_rate_gap"],
        "sacks": ["sack_rate_gap"],
        "all_situational": [f"{m}_gap" for m in SITUATIONAL],
    }
    train_ids = set(games.loc[games["season"].between(2015, 2022), "game_id"])
    holdout_ids = set(games.loc[games["season"].between(2023, 2025), "game_id"])
    print("=== OAK V7 SITUATIONAL TOURNAMENT ===")
    control = v5[v5["game_id"].isin(holdout_ids)].dropna(subset=["predicted_home_margin", "actual_home_margin"])
    print("V5 CONTROL", evaluate_margin_predictions(control))
    for name, features in feature_sets.items():
        merged = frame.merge(v5[["game_id", "predicted_home_margin"]], on="game_id", how="left", suffixes=("", "_v5"))
        model_features = ["predicted_home_margin", *features]
        usable = merged.dropna(subset=[*model_features, "actual_home_margin"])
        train = usable[usable["game_id"].isin(train_ids)]
        holdout = usable[usable["game_id"].isin(holdout_ids)].copy()
        if train.empty or holdout.empty:
            print(name, "SKIPPED insufficient data")
            continue
        model = Ridge(alpha=1.0).fit(train[model_features], train["actual_home_margin"])
        holdout["predicted_home_margin"] = model.predict(holdout[model_features])
        metrics = evaluate_margin_predictions(holdout)
        print(name, metrics, "coefficients", dict(zip(model_features, np.round(model.coef_, 4))))


if __name__ == "__main__":
    run()
