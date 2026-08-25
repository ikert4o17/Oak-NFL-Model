"""Baseline team-strength and game-margin helpers."""

from __future__ import annotations

import pandas as pd


def efficiency_rating_frame(pregame_ratings: pd.DataFrame) -> pd.DataFrame:
    """Collapse pregame offense/defense EPA into a simple net team rating."""
    required = {
        "season",
        "week",
        "game_id",
        "team",
        "pregame_off_epa_per_play",
        "pregame_def_epa_per_play_allowed",
    }
    missing = required.difference(pregame_ratings.columns)
    if missing:
        raise ValueError(f"pregame ratings missing required columns: {sorted(missing)}")

    ratings = pregame_ratings.copy()
    ratings["offense_rating"] = ratings["pregame_off_epa_per_play"]
    ratings["defense_rating"] = -ratings["pregame_def_epa_per_play_allowed"]
    ratings["net_rating"] = ratings["offense_rating"] + ratings["defense_rating"]
    return ratings[
        [
            "season",
            "week",
            "game_id",
            "team",
            "offense_rating",
            "defense_rating",
            "net_rating",
        ]
    ]


def predict_margin(
    home_rating: float,
    away_rating: float,
    *,
    home_field_points: float = 1.5,
    epa_to_points: float = 20.0,
) -> float:
    """Convert a net EPA/play rating gap into an expected home scoring margin."""
    return (home_rating - away_rating) * epa_to_points + home_field_points


def build_game_predictions(
    games: pd.DataFrame,
    ratings: pd.DataFrame,
    *,
    home_field_points: float = 1.5,
    epa_to_points: float = 20.0,
) -> pd.DataFrame:
    """Attach point-in-time team ratings to games and predict home margin.

    ``games`` must contain one row per game with ``game_id``, ``season``, ``week``,
    ``home_team``, and ``away_team``. Ratings are matched on the same game/week,
    which ensures they represent the information set entering that game.
    """
    game_required = {"game_id", "season", "week", "home_team", "away_team"}
    rating_required = {"game_id", "season", "week", "team", "net_rating"}
    missing_games = game_required.difference(games.columns)
    missing_ratings = rating_required.difference(ratings.columns)
    if missing_games:
        raise ValueError(f"games missing required columns: {sorted(missing_games)}")
    if missing_ratings:
        raise ValueError(f"ratings missing required columns: {sorted(missing_ratings)}")

    home = ratings[["game_id", "season", "week", "team", "net_rating"]].rename(
        columns={"team": "home_team", "net_rating": "home_net_rating"}
    )
    away = ratings[["game_id", "season", "week", "team", "net_rating"]].rename(
        columns={"team": "away_team", "net_rating": "away_net_rating"}
    )

    output = games.merge(
        home,
        on=["game_id", "season", "week", "home_team"],
        how="left",
        validate="one_to_one",
    ).merge(
        away,
        on=["game_id", "season", "week", "away_team"],
        how="left",
        validate="one_to_one",
    )

    output["predicted_home_margin"] = (
        (output["home_net_rating"] - output["away_net_rating"]) * epa_to_points
        + home_field_points
    )
    return output
