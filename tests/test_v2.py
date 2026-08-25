import pandas as pd
import pytest

from oak_nfl.ratings.v2 import build_v2_pregame_ratings


def _team_games() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"game_id": "2024_01_A_B", "season": 2024, "week": 1, "posteam": "A", "defteam": "B", "epa_per_play": 0.20},
            {"game_id": "2024_01_A_B", "season": 2024, "week": 1, "posteam": "B", "defteam": "A", "epa_per_play": -0.10},
            {"game_id": "2025_01_A_B", "season": 2025, "week": 1, "posteam": "A", "defteam": "B", "epa_per_play": 0.80},
            {"game_id": "2025_01_A_B", "season": 2025, "week": 1, "posteam": "B", "defteam": "A", "epa_per_play": -0.40},
            {"game_id": "2025_02_A_B", "season": 2025, "week": 2, "posteam": "A", "defteam": "B", "epa_per_play": 1.00},
            {"game_id": "2025_02_A_B", "season": 2025, "week": 2, "posteam": "B", "defteam": "A", "epa_per_play": -0.60},
        ]
    )


def test_v2_week_one_uses_previous_season_prior() -> None:
    ratings = build_v2_pregame_ratings(
        _team_games(), prior_games=4.0, prior_regression=0.50, recency_decay=0.85
    )
    a_week_one = ratings[
        (ratings["season"] == 2025) & (ratings["week"] == 1) & (ratings["team"] == "A")
    ].iloc[0]

    # 2024 league EPA is 0.05. A's 0.20 offense regresses halfway to 0.125.
    assert a_week_one["pregame_off_epa_per_play"] == pytest.approx(0.125)
    assert a_week_one["games_played"] == 0


def test_v2_week_two_excludes_week_two_result() -> None:
    ratings = build_v2_pregame_ratings(
        _team_games(), prior_games=4.0, prior_regression=0.50, recency_decay=0.85
    )
    a_week_two = ratings[
        (ratings["season"] == 2025) & (ratings["week"] == 2) & (ratings["team"] == "A")
    ].iloc[0]

    expected = (4.0 * 0.125 + 1.0 * 0.80) / 5.0
    assert a_week_two["pregame_off_epa_per_play"] == pytest.approx(expected)
    assert a_week_two["games_played"] == 1


def test_v2_rejects_invalid_parameters() -> None:
    games = _team_games()
    with pytest.raises(ValueError):
        build_v2_pregame_ratings(games, prior_games=-1)
    with pytest.raises(ValueError):
        build_v2_pregame_ratings(games, prior_regression=1.1)
    with pytest.raises(ValueError):
        build_v2_pregame_ratings(games, recency_decay=0)
