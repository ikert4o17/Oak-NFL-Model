"""Test whether passing/rushing EPA add signal beyond promoted Oak V5."""

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
    metrics = [
        "epa_per_play",
        "pass_epa_per_play",
        "rush_epa_per_play",
        "success_rate",
        "explosive_rate",
    ]
    cols = [item for metric in metrics for item in (
        f"pregame_off_{metric}", f"pregame_def_{metric}_allowed"
    )]
    home = ratings[["game_id", "team", *cols]].rename(
        columns={"team": "home_team", **{col: f"home_{col}" for col in cols}}
    )
    away = ratings[["game_id", "team", *cols]].rename(
        columns={"team": "away_team", **{col: f"away_{col}" for col in cols}}
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


def _fit_and_score(
    train: pd.DataFrame,
    holdout: pd.DataFrame,
    features: list[str],
) -> tuple[Ridge, dict[str, float]]:
    alpha_rows = []
    for alpha in [0.0, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]:
        model = Ridge(alpha=alpha).fit(train[features], train["actual_home_margin"])
        pred = model.predict(train[features])
        alpha_rows.append((alpha, float(np.mean(np.abs(pred - train["actual_home_margin"])))))
    best_alpha = min(alpha_rows, key=lambda row: row[1])[0]
    model = Ridge(alpha=best_alpha).fit(train[features], train["actual_home_margin"])
    scored = holdout.copy()
    scored["predicted_home_margin"] = model.predict(scored[features])
    return model, evaluate_margin_predictions(scored)


def _print_model(label: str, model: Ridge, features: list[str], metrics: dict[str, float]) -> None:
    print(f"\n{label}")
    print(f"intercept: {model.intercept_:.4f}")
    for name, coef in zip(features, model.coef_):
        print(f"{name}: {coef:.4f}")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")


def run() -> None:
    pbp = pd.concat([load_pbp(season) for season in range(2014, 2026)], ignore_index=True)
    team_games = build_team_game_features(pbp)
    games = build_game_results(pbp)
    v5 = build_v5_game_predictions(games, build_v5_pregame_ratings(team_games))

    frame = _game_frame(games, build_v6_pregame_ratings(team_games))
    all_features = [
        "epa_per_play_gap",
        "pass_epa_per_play_gap",
        "rush_epa_per_play_gap",
        "success_rate_gap",
        "explosive_rate_gap",
    ]
    usable = frame.dropna(subset=[*all_features, "actual_home_margin"]).copy()
    train = usable[usable["season"].between(2015, 2022)].copy()
    holdout = usable[usable["season"].between(2023, 2025)].copy()

    split_features = [
        "pass_epa_per_play_gap",
        "rush_epa_per_play_gap",
        "success_rate_gap",
        "explosive_rate_gap",
    ]
    pass_augmented_features = [
        "epa_per_play_gap",
        "pass_epa_per_play_gap",
        "success_rate_gap",
        "explosive_rate_gap",
    ]
    full_augmented_features = all_features

    split_model, split_metrics = _fit_and_score(train, holdout, split_features)
    pass_model, pass_metrics = _fit_and_score(train, holdout, pass_augmented_features)
    full_model, full_metrics = _fit_and_score(train, holdout, full_augmented_features)

    ids = set(holdout["game_id"])
    v5_holdout = v5[v5["game_id"].isin(ids)].dropna(
        subset=["predicted_home_margin", "actual_home_margin"]
    )
    print("=== HOLDOUT 2023-2025 ===")
    print("\nV5 CONTROL")
    for key, value in evaluate_margin_predictions(v5_holdout).items():
        print(f"{key}: {value:.4f}")
    _print_model("V6 PASS/RUSH SPLIT", split_model, split_features, split_metrics)
    _print_model("V6 V5 + PASS EPA", pass_model, pass_augmented_features, pass_metrics)
    _print_model("V6 V5 + PASS + RUSH EPA", full_model, full_augmented_features, full_metrics)


if __name__ == "__main__":
    run()
