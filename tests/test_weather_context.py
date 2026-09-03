import pandas as pd

from oak_nfl.data.weather import fetch_slate_weather, kickoff_utc
from oak_nfl.weather_context import build_game_weather_context


class FakeResponse:
    def raise_for_status(self):
        return None

    def json(self):
        return {
            "hourly": {
                "time": ["2026-09-11T00:00", "2026-09-11T01:00"],
                "temperature_2m": [72.0, 71.0],
                "precipitation_probability": [20, 30],
                "precipitation": [0.0, 0.01],
                "weather_code": [1, 2],
                "wind_speed_10m": [8.0, 9.0],
                "wind_gusts_10m": [13.0, 14.0],
            }
        }


class FakeSession:
    def get(self, url, params=None, timeout=None):
        assert params["temperature_unit"] == "fahrenheit"
        assert params["wind_speed_unit"] == "mph"
        assert params["timezone"] == "UTC"
        return FakeResponse()


def test_kickoff_utc_uses_nflverse_eastern_time():
    game = pd.Series({"gameday": "2026-09-10", "gametime": "20:20"})
    assert kickoff_utc(game) == pd.Timestamp("2026-09-11T00:20:00Z")


def test_fetch_slate_weather_is_informational_zero_point():
    slate = pd.DataFrame(
        [
            {
                "game_id": "2026_01_DAL_PHI",
                "gameday": "2026-09-10",
                "gametime": "20:20",
                "home_team": "PHI",
                "away_team": "DAL",
                "location": "Home",
            }
        ]
    )
    row = fetch_slate_weather(slate, session=FakeSession()).iloc[0]
    assert bool(row["weather_available"])
    assert row["weather_source"] == "open-meteo"
    assert row["temperature_f"] == 72.0
    assert row["wind_mph"] == 8.0
    assert row["weather_auto_points"] == 0.0


def test_neutral_site_fails_safe_without_wrong_coordinates():
    slate = pd.DataFrame(
        [
            {
                "game_id": "neutral",
                "gameday": "2026-10-01",
                "gametime": "09:30",
                "home_team": "JAX",
                "away_team": "CHI",
                "location": "Neutral",
            }
        ]
    )
    row = fetch_slate_weather(slate, session=FakeSession()).iloc[0]
    assert not bool(row["weather_available"])
    assert row["weather_source"] == "missing"
    assert row["weather_auto_points"] == 0.0


def test_game_weather_context_preserves_environment_metadata():
    slate = pd.DataFrame(
        [
            {
                "game_id": "g1",
                "home_team": "DET",
                "away_team": "GB",
                "roof": "dome",
                "surface": "fieldturf",
                "stadium": "Ford Field",
            }
        ]
    )
    weather = pd.DataFrame(
        [
            {
                "game_id": "g1",
                "weather_available": False,
                "weather_source": "missing",
                "weather_auto_points": 0.0,
            }
        ]
    )
    row = build_game_weather_context(slate, weather=weather).iloc[0]
    assert row["roof"] == "dome"
    assert row["surface"] == "fieldturf"
    assert row["stadium"] == "Ford Field"
    assert row["weather_auto_points"] == 0.0
