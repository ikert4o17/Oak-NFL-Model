"""Test whether splitting passing and rushing EPA improves on promoted Oak V5."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from oak_nfl.backtest import evaluate_margin_predictions
from oak_nfl.data.games import build_game_results
from oak_nfl.data.nflverse import load_pbp
from oak_nfl.features import build_team_game_features
from oak_nfl.ratings.v5 import build_v5_game_predictions, build_v5_pregame_ratings
from oak_nfl.ratings.v6 import build_v6_pregame_ratings


def _game_frame(games: pd.DataFrame, ratings: pd.DataFrame) -> pd.DataFrame:
    metrics = ["pass_epa_per_play", "rush_epa_per_play", "success_rate", "explosive_rate"]
    cols = [item for m in metrics for item in (f"pregame_off_{m}", f"pregame_def_{m}_allowed")]
    home = ratings[["game_id", "team", *cols]].rename(
        columns={"team": "home_team", **{c: f"home_{c}" for c in cols}}
    )
    away = ratings[["game_id", "team", *cols]].rename(
        columns={"team": "away_team", **{c: f"away_{c}" for c in cols}}
    )
    frame = games.merge(home, on=["game_id", "home_team"], how="left", validate="one_to_one")
    frame = frame.merge(away, on=["game_id", "away_team"], how="left", validate="one_to_one")
    for metric in metrics:
        frame[f"{metric}_gap"] = (
            frame[f"home_pregame_off_{metric}"]
            - frame[f"home_pregame_def_{metric}_allowed"]
            - frame[f"away_pregame_off_{metric}"]
            + frame[f"away_pregame_def_{metric}_allowed"]
        )
    return frame


def run() -> None:
    pbp = pd.concat([load_pbp(season) for season in range(2014, 2026)], ignore_index=True)
    team_games = build_team_game_features(pbp)
    games = build_game_results(pbp)

    # Frozen V5 control.
    v5 = build_v5_game_predictions(games, build_v5_pregame_ratings(team_games))

    frame = _game_frame(games, build_v6_pregame_ratings(team_games))
    features = [
        "pass_epa_per_play_gap",
        "rush_epa_per_play_gap",
        "success_rate_gap",
        "explosive_rate_gap",
    ]
    usable = frame.dropna(subset=[*features, "actual_home_margin"]).copy()
    train = usable[usable["season"].between(2015, 2022)].copy()
    holdout = usable[usable["season"].between(2023, 2025)].copy()

    alpha_rows = []
    for alpha in [0.0, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]:
        model = Ridge(alpha=alpha).fit(train[features], train["actual_home_margin"])
        pred = model.predict(train[features])
        alpha_rows.append((alpha, float(np.mean(np.abs(pred - train["actual_home_margin"])))))
    best_alpha = min(alpha_rows, key=lambda row: row[1])[0]
    model = Ridge(alpha=best_alpha).fit(train[features], train["actual_home_margin"])
    holdout["predicted_home_margin"] = model.predict(holdout[features])

    ids = set(holdout["game_id"])
    v5_holdout = v5[v5["game_id"].isin(ids)].dropna(subset=["predicted_home_margin", "actual_home_margin"])
    print("=== OAK V6 TRAINING ===")
    print(f"selected_alpha: {best_alpha:.2f}")
    print(f"intercept: {model.intercept_:.4f}")
    for name, coef in zip(features, model.coef_):
        print(f"{name}: {coef:.4f}")
    print("\n=== HOLDOUT 2023-2025 ===")
    print("V5 CONTROL")
    for key, value in evaluate_margin_predictions(v5_holdout).items():
        print(f"{key}: {value:.4f}")
    print("V6 PASS/RUSH SPLIT")
    for key, value in evaluate_margin_predictions(holdout).items():
        print(f"{key}: {value:.4f}")


if __name__ == "__main__":
    run()
