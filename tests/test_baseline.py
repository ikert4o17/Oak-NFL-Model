import pandas as pd
import pytest

from oak_nfl.ratings.baseline import build_game_predictions, efficiency_rating_frame, predict_margin


def test_efficiency_rating_frame_builds_net_rating() -> None:
    pregame = pd.DataFrame(
        [
            {
                "season": 2025,
                "week": 2,
                "game_id": "g1",
                "team": "A",
                "pregame_off_epa_per_play": 0.10,
                "pregame_def_epa_per_play_allowed": -0.05,
            }
        ]
    )
    ratings = efficiency_rating_frame(pregame)
    assert ratings.loc[0, "offense_rating"] == pytest.approx(0.10)
    assert ratings.loc[0, "defense_rating"] == pytest.approx(0.05)
    assert ratings.loc[0, "net_rating"] == pytest.approx(0.15)


def test_predict_margin_applies_rating_gap_and_home_field() -> None:
    assert predict_margin(0.15, 0.05, home_field_points=1.5, epa_to_points=20) == pytest.approx(3.5)


def test_build_game_predictions_uses_same_game_pregame_ratings() -> None:
    games = pd.DataFrame(
        [
            {
                "game_id": "g1",
                "season": 2025,
                "week": 3,
                "home_team": "A",
                "away_team": "B",
            }
        ]
    )
    ratings = pd.DataFrame(
        [
            {"game_id": "g1", "season": 2025, "week": 3, "team": "A", "net_rating": 0.12},
            {"game_id": "g1", "season": 2025, "week": 3, "team": "B", "net_rating": 0.02},
        ]
    )

    predictions = build_game_predictions(games, ratings, home_field_points=1.5, epa_to_points=20)
    assert predictions.loc[0, "predicted_home_margin"] == pytest.approx(3.5)
