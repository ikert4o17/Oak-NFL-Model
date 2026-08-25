import pandas as pd
import pytest

from oak_nfl.weather import build_weather_features, normalize_weather


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
