import pandas as pd
import pytest

from oak_nfl.data.games import build_game_results


def test_build_game_results_returns_one_row_and_margin() -> None:
    pbp = pd.DataFrame(
        [
            {
                "game_id": "g1",
                "season": 2025,
                "week": 1,
                "home_team": "A",
                "away_team": "B",
                "home_score": 7,
                "away_score": 3,
            },
            {
                "game_id": "g1",
                "season": 2025,
                "week": 1,
                "home_team": "A",
                "away_team": "B",
                "home_score": 24,
                "away_score": 17,
            },
        ]
    )
    games = build_game_results(pbp)
    assert len(games) == 1
    assert games.loc[0, "actual_home_margin"] == 7


def test_build_game_results_validates_schema() -> None:
    with pytest.raises(ValueError):
        build_game_results(pd.DataFrame({"game_id": ["g1"]}))
