"""Test whether success and explosive rates add signal beyond Oak V4 EPA ratings."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from oak_nfl.backtest import evaluate_margin_predictions
from oak_nfl.data.games import build_game_results
from oak_nfl.data.nflverse import load_pbp
from oak_nfl.features import build_team_game_features
from oak_nfl.ratings.baseline import build_game_predictions, efficiency_rating_frame
from oak_nfl.ratings.v2 import build_v2_pregame_ratings
from oak_nfl.ratings.v5 import build_v5_pregame_ratings


def _v5_game_frame(games: pd.DataFrame, ratings: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [
        "pregame_off_epa_per_play",
        "pregame_def_epa_per_play_allowed",
        "pregame_off_success_rate",
        "pregame_def_success_rate_allowed",
        "pregame_off_explosive_rate",
        "pregame_def_explosive_rate_allowed",
    ]
    home = ratings[["game_id", "team", *metric_cols]].rename(
        columns={"team": "home_team", **{col: f"home_{col}" for col in metric_cols}}
    )
    away = ratings[["game_id", "team", *metric_cols]].rename(
        columns={"team": "away_team", **{col: f"away_{col}" for col in metric_cols}}
    )
    frame = games.merge(home, on=["game_id", "home_team"], how="left", validate="one_to_one")
    frame = frame.merge(away, on=["game_id", "away_team"], how="left", validate="one_to_one")

    frame["epa_gap"] = (
        frame["home_pregame_off_epa_per_play"]
        - frame["home_pregame_def_epa_per_play_allowed"]
        - frame["away_pregame_off_epa_per_play"]
        + frame["away_pregame_def_epa_per_play_allowed"]
    )
    frame["success_gap"] = (
        frame["home_pregame_off_success_rate"]
        - frame["home_pregame_def_success_rate_allowed"]
        - frame["away_pregame_off_success_rate"]
        + frame["away_pregame_def_success_rate_allowed"]
    )
    frame["explosive_gap"] = (
        frame["home_pregame_off_explosive_rate"]
        - frame["home_pregame_def_explosive_rate_allowed"]
        - frame["away_pregame_off_explosive_rate"]
        + frame["away_pregame_def_explosive_rate_allowed"]
    )
    return frame


def run() -> None:
    pbp = pd.concat([load_pbp(season) for season in range(2014, 2026)], ignore_index=True)
    team_games = build_team_game_features(pbp)
    games = build_game_results(pbp)

    # V4 control: promoted V2 ratings with the validated 26x EPA scale and +1.5 HFA.
    v2 = build_v2_pregame_ratings(team_games)
    v4_ratings = efficiency_rating_frame(v2)
    v4 = build_game_predictions(games, v4_ratings, epa_to_points=26.0, home_field_points=1.5)

    v5_ratings = build_v5_pregame_ratings(team_games)
    frame = _v5_game_frame(games, v5_ratings)
    features = ["epa_gap", "success_gap", "explosive_gap"]
    usable = frame.dropna(subset=[*features, "actual_home_margin"]).copy()

    train = usable[usable["season"].between(2015, 2022)].copy()
    holdout = usable[usable["season"].between(2023, 2025)].copy()

    # Ridge protects against correlated efficiency signals. Alpha is selected only
    # on training years; the 2023-2025 holdout remains untouched.
    alpha_rows = []
    for alpha in [0.0, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]:
        model = Ridge(alpha=alpha)
        model.fit(train[features], train["actual_home_margin"])
        pred = model.predict(train[features])
        alpha_rows.append((alpha, float(np.mean(np.abs(pred - train["actual_home_margin"])))))
    best_alpha = min(alpha_rows, key=lambda row: row[1])[0]

    model = Ridge(alpha=best_alpha)
    model.fit(train[features], train["actual_home_margin"])
    holdout["predicted_home_margin"] = model.predict(holdout[features])
    v5_metrics = evaluate_margin_predictions(holdout)

    holdout_ids = set(holdout["game_id"])
    v4_holdout = v4[v4["game_id"].isin(holdout_ids)].copy()
    v4_metrics = evaluate_margin_predictions(v4_holdout)

    print("=== OAK V5 TRAINING ===")
    print(f"selected_alpha: {best_alpha:.2f}")
    print(f"intercept: {model.intercept_:.4f}")
    for name, coef in zip(features, model.coef_):
        print(f"{name}: {coef:.4f}")
    print("\n=== HOLDOUT 2023-2025 ===")
    print("V4 CONTROL")
    for key, value in v4_metrics.items():
        print(f"{key}: {value:.4f}")
    print("V5 MULTI-METRIC")
    for key, value in v5_metrics.items():
        print(f"{key}: {value:.4f}")


if __name__ == "__main__":
    run()
