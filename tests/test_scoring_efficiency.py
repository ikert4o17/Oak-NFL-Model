import pandas as pd
import pytest

from oak_nfl.scoring_efficiency import build_scoring_efficiency


def test_dpa100_is_pregame_and_uses_prior_game_only():
    games = pd.DataFrame(
        {
            "game_id": ["g1", "g2"],
            "season": [2024, 2024],
            "week": [1, 2],
            "home_team": ["KC", "KC"],
            "away_team": ["DEN", "LV"],
            "home_score": [21, 40],
            "away_score": [14, 10],
            "home_yards": [350, 500],
            "away_yards": [280, 200],
        }
    )
    features = build_scoring_efficiency(games)
    assert pd.isna(features.iloc[0].home_dpa100_4)
    assert features.iloc[1].home_dpa100_4 == pytest.approx(5.0)
    assert features.iloc[1].home_ppa100_4 == pytest.approx(6.0)


def test_current_game_cannot_change_own_feature():
    games = pd.DataFrame(
        {
            "game_id": ["g1", "g2"],
            "season": [2024, 2024],
            "week": [1, 2],
            "home_team": ["KC", "KC"],
            "away_team": ["DEN", "LV"],
            "home_score": [21, 20],
            "away_score": [14, 10],
            "home_yards": [350, 300],
            "away_yards": [280, 200],
        }
    )
    changed = games.copy()
    changed.loc[
        1,
        ["home_score", "away_score", "home_yards", "away_yards"],
    ] = [60, 50, 600, 550]

    original = build_scoring_efficiency(games).iloc[1]
    modified = build_scoring_efficiency(changed).iloc[1]
    assert original.home_dpa100_4 == modified.home_dpa100_4
    assert original.home_ppa100_4 == modified.home_ppa100_4
