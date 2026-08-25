"""End-to-end historical modeling pipelines."""

from __future__ import annotations

import pandas as pd

from oak_nfl.backtest import evaluate_margin_predictions
from oak_nfl.data.games import build_game_results
from oak_nfl.features import build_team_game_features, build_team_weekly_ratings
from oak_nfl.ratings.baseline import build_game_predictions, efficiency_rating_frame
from oak_nfl.ratings.v2 import build_v2_pregame_ratings


def run_baseline_backtest(
    pbp: pd.DataFrame,
    *,
    home_field_points: float = 1.5,
    epa_to_points: float = 20.0,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Run Oak's first leakage-safe historical margin baseline on one PBP frame."""
    team_games = build_team_game_features(pbp)
    pregame = build_team_weekly_ratings(team_games)
    ratings = efficiency_rating_frame(pregame)
    games = build_game_results(pbp)
    predictions = build_game_predictions(
        games,
        ratings,
        home_field_points=home_field_points,
        epa_to_points=epa_to_points,
    )
    metrics = evaluate_margin_predictions(predictions)
    return predictions, metrics


def run_v2_backtest(
    pbp: pd.DataFrame,
    *,
    prior_games: float = 4.0,
    prior_regression: float = 0.50,
    recency_decay: float = 0.85,
    home_field_points: float = 1.5,
    epa_to_points: float = 20.0,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Run Oak V2 using prior-season priors and recency-weighted current form."""
    team_games = build_team_game_features(pbp)
    pregame = build_v2_pregame_ratings(
        team_games,
        prior_games=prior_games,
        prior_regression=prior_regression,
        recency_decay=recency_decay,
    )
    ratings = efficiency_rating_frame(pregame)
    games = build_game_results(pbp)
    predictions = build_game_predictions(
        games,
        ratings,
        home_field_points=home_field_points,
        epa_to_points=epa_to_points,
    )
    metrics = evaluate_margin_predictions(predictions)
    return predictions, metrics
