import pandas as pd
import pytest

from oak_nfl.features import build_team_game_features, build_team_weekly_ratings, clean_scrimmage_plays


def sample_pbp() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "game_id": "2025_01_A_B",
                "season": 2025,
                "week": 1,
                "posteam": "A",
                "defteam": "B",
                "play_type": "pass",
                "epa": 0.4,
                "success": 1.0,
                "yards_gained": 12,
                "pass": 1,
                "rush": 0,
                "qb_kneel": 0,
                "qb_spike": 0,
            },
            {
                "game_id": "2025_01_A_B",
                "season": 2025,
                "week": 1,
                "posteam": "A",
                "defteam": "B",
                "play_type": "run",
                "epa": -0.2,
                "success": 0.0,
                "yards_gained": 4,
                "pass": 0,
                "rush": 1,
                "qb_kneel": 0,
                "qb_spike": 0,
            },
            {
                "game_id": "2025_01_A_B",
                "season": 2025,
                "week": 1,
                "posteam": "B",
                "defteam": "A",
                "play_type": "pass",
                "epa": -0.1,
                "success": 0.0,
                "yards_gained": 25,
                "pass": 1,
                "rush": 0,
                "qb_kneel": 0,
                "qb_spike": 0,
            },
            {
                "game_id": "2025_02_A_C",
                "season": 2025,
                "week": 2,
                "posteam": "A",
                "defteam": "C",
                "play_type": "pass",
                "epa": 0.8,
                "success": 1.0,
                "yards_gained": 30,
                "pass": 1,
                "rush": 0,
                "qb_kneel": 0,
                "qb_spike": 0,
            },
            {
                "game_id": "2025_02_A_C",
                "season": 2025,
                "week": 2,
                "posteam": "C",
                "defteam": "A",
                "play_type": "run",
                "epa": -0.5,
                "success": 0.0,
                "yards_gained": 2,
                "pass": 0,
                "rush": 1,
                "qb_kneel": 0,
                "qb_spike": 0,
            },
        ]
    )


def test_clean_scrimmage_plays_requires_expected_schema() -> None:
    with pytest.raises(ValueError):
        clean_scrimmage_plays(pd.DataFrame({"game_id": ["x"]}))


def test_build_team_game_features_aggregates_epa() -> None:
    features = build_team_game_features(sample_pbp())
    row = features[(features["game_id"] == "2025_01_A_B") & (features["posteam"] == "A")].iloc[0]
    assert row["plays"] == 2
    assert row["epa_per_play"] == pytest.approx(0.1)
    assert row["success_rate"] == pytest.approx(0.5)


def test_pregame_ratings_exclude_current_game() -> None:
    features = build_team_game_features(sample_pbp())
    ratings = build_team_weekly_ratings(features)

    week_one_a = ratings[(ratings["week"] == 1) & (ratings["team"] == "A")].iloc[0]
    week_two_a = ratings[(ratings["week"] == 2) & (ratings["team"] == "A")].iloc[0]

    assert pd.isna(week_one_a["pregame_off_epa_per_play"])
    assert week_two_a["pregame_off_epa_per_play"] == pytest.approx(0.1)
    assert week_two_a["pregame_def_epa_per_play_allowed"] == pytest.approx(-0.1)
