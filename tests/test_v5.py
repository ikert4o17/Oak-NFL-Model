import pandas as pd
import pytest

from oak_nfl.ratings.v5 import build_v5_pregame_ratings


def _team_games() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "game_id": "2024g",
                "season": 2024,
                "week": 1,
                "posteam": "A",
                "defteam": "B",
                "epa_per_play": 0.20,
                "success_rate": 0.55,
                "explosive_rate": 0.12,
            },
            {
                "game_id": "2025w1",
                "season": 2025,
                "week": 1,
                "posteam": "A",
                "defteam": "B",
                "epa_per_play": -0.10,
                "success_rate": 0.35,
                "explosive_rate": 0.04,
            },
            {
                "game_id": "2025w2",
                "season": 2025,
                "week": 2,
                "posteam": "A",
                "defteam": "B",
                "epa_per_play": 0.40,
                "success_rate": 0.70,
                "explosive_rate": 0.20,
            },
        ]
    )


def test_v5_week_one_uses_previous_season_priors() -> None:
    ratings = build_v5_pregame_ratings(
        _team_games(), prior_games=4.0, prior_regression=1.0, recency_decay=0.85
    )
    row = ratings[(ratings["season"] == 2025) & (ratings["week"] == 1) & (ratings["team"] == "A")].iloc[0]
    assert row["pregame_off_epa_per_play"] == pytest.approx(0.20)
    assert row["pregame_off_success_rate"] == pytest.approx(0.55)
    assert row["pregame_off_explosive_rate"] == pytest.approx(0.12)


def test_v5_week_two_cannot_see_week_two_result() -> None:
    base = _team_games()
    first = build_v5_pregame_ratings(base)
    changed = base.copy()
    changed.loc[changed["game_id"].eq("2025w2"), ["epa_per_play", "success_rate", "explosive_rate"]] = [
        -0.90,
        0.05,
        0.00,
    ]
    second = build_v5_pregame_ratings(changed)

    cols = [
        "pregame_off_epa_per_play",
        "pregame_off_success_rate",
        "pregame_off_explosive_rate",
    ]
    first_row = first[(first["season"] == 2025) & (first["week"] == 2) & (first["team"] == "A")].iloc[0]
    second_row = second[(second["season"] == 2025) & (second["week"] == 2) & (second["team"] == "A")].iloc[0]
    for column in cols:
        assert first_row[column] == pytest.approx(second_row[column])
