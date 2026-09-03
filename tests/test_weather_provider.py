import pandas as pd

from oak_nfl.data.weather import fetch_slate_weather


class BrokenSession:
    def get(self, url, params=None, timeout=None):
        raise RuntimeError("provider unavailable")


def test_provider_failure_does_not_create_weather_points():
    slate = pd.DataFrame(
        [
            {
                "game_id": "g1",
                "gameday": "2026-09-13",
                "gametime": "13:00",
                "home_team": "BUF",
                "away_team": "NYJ",
                "location": "Home",
            }
        ]
    )
    row = fetch_slate_weather(slate, session=BrokenSession()).iloc[0]
    assert not bool(row["weather_available"])
    assert row["weather_source"] == "missing"
    assert row["weather_auto_points"] == 0.0
