import pandas as pd
import pytest

from oak_nfl.weather import (
    build_temperature_acclimation,
    build_weather_features,
    normalize_weather,
)


def test_normalize_weather_requires_canonical_columns():
    with pytest.raises(ValueError, match="missing required columns"):
        normalize_weather(pd.DataFrame({"game_id": ["g1"]}))


def test_build_weather_features_zeroes_weather_for_domes():
    weather = pd.DataFrame(
        {
            "game_id": ["g1"],
            "temperature_f": [20],
            "wind_mph": [25],
            "precipitation_in": [0.5],
            "is_dome": [True],
        }
    )
    row = build_weather_features(weather).iloc[0]
    assert row["outdoor"] == 0
    assert row["wind_over_10"] == 0
    assert row["wind_over_15"] == 0
    assert row["precipitation"] == 0
    assert row["cold_below_32"] == 0


def test_build_weather_features_captures_adverse_outdoor_conditions():
    weather = pd.DataFrame(
        {
            "game_id": ["g1"],
            "temperature_f": [25],
            "wind_mph": [18],
            "precipitation_in": [0.2],
            "is_dome": [False],
        }
    )
    row = build_weather_features(weather).iloc[0]
    assert row["outdoor"] == 1
    assert row["wind_over_10"] == pytest.approx(8.0)
    assert row["wind_over_15"] == pytest.approx(3.0)
    assert row["precipitation"] == pytest.approx(0.2)
    assert row["cold_below_32"] == pytest.approx(7.0)
    assert row["heat_above_85"] == 0


def test_weather_numeric_inputs_are_coerced_and_nonnegative():
    weather = pd.DataFrame(
        {
            "game_id": ["g1"],
            "temperature_f": ["90"],
            "wind_mph": ["-3"],
            "precipitation_in": ["-1"],
            "is_dome": [False],
        }
    )
    row = build_weather_features(weather).iloc[0]
    assert row["wind_mph"] == 0
    assert row["precipitation_in"] == 0
    assert row["heat_above_85"] == pytest.approx(5.0)


def test_temperature_acclimation_uses_only_prior_home_weather():
    games = pd.DataFrame(
        {
            "game_id": ["g1", "g2", "g3"],
            "season": [2023, 2023, 2023],
            "week": [1, 2, 3],
            "home_team": ["MIA", "GB", "GB"],
            "away_team": ["NE", "DET", "MIA"],
            "temperature_f": [90, 45, 25],
            "is_dome": [False, False, False],
        }
    )
    features = build_temperature_acclimation(games)
    week3 = features.loc[features["game_id"].eq("g3")].iloc[0]
    assert week3["home_climate_temp"] == pytest.approx(45.0)
    assert week3["away_climate_temp"] == pytest.approx(90.0)
    assert week3["home_cold_shock"] == pytest.approx(20.0)
    assert week3["away_cold_shock"] == pytest.approx(65.0)
    assert week3["cold_shock_difference"] == pytest.approx(45.0)


def test_indoor_team_outdoor_exposure_is_flagged_from_prior_home_games():
    games = pd.DataFrame(
        {
            "game_id": ["g1", "g2", "g3"],
            "season": [2023, 2023, 2023],
            "week": [1, 2, 3],
            "home_team": ["DET", "CHI", "CHI"],
            "away_team": ["ATL", "GB", "DET"],
            "temperature_f": [70, 50, 30],
            "is_dome": [True, False, False],
        }
    )
    features = build_temperature_acclimation(games)
    week3 = features.loc[features["game_id"].eq("g3")].iloc[0]
    assert week3["away_indoor_home_share"] == pytest.approx(1.0)
    assert week3["away_indoor_to_outdoor"] == pytest.approx(1.0)
