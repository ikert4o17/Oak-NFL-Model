"""Live pregame weather provider for Oak's informational context layer."""

from __future__ import annotations

from datetime import UTC
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd
import requests

OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"
EASTERN = ZoneInfo("America/New_York")
HOURLY_FIELDS = (
    "temperature_2m,precipitation_probability,precipitation,weather_code,"
    "wind_speed_10m,wind_gusts_10m"
)
WEATHER_FETCH_ERRORS = (
    requests.RequestException,
    ValueError,
    KeyError,
    TypeError,
    IndexError,
)

# Approximate home-stadium coordinates. Neutral-site games intentionally do not
# fall back to these coordinates because a wrong forecast is worse than missing
# context. The schedule's venue/roof fields remain preserved separately.
TEAM_COORDINATES: dict[str, tuple[float, float]] = {
    "ARI": (33.5276, -112.2626),
    "ATL": (33.7554, -84.4008),
    "BAL": (39.2780, -76.6227),
    "BUF": (42.7738, -78.7870),
    "CAR": (35.2258, -80.8528),
    "CHI": (41.8623, -87.6167),
    "CIN": (39.0954, -84.5160),
    "CLE": (41.5061, -81.6995),
    "DAL": (32.7473, -97.0945),
    "DEN": (39.7439, -105.0201),
    "DET": (42.3400, -83.0456),
    "GB": (44.5013, -88.0622),
    "HOU": (29.6847, -95.4107),
    "IND": (39.7601, -86.1639),
    "JAX": (30.3239, -81.6373),
    "KC": (39.0489, -94.4839),
    "LA": (33.9535, -118.3392),
    "LAC": (33.9535, -118.3392),
    "LV": (36.0908, -115.1830),
    "MIA": (25.9580, -80.2389),
    "MIN": (44.9736, -93.2575),
    "NE": (42.0909, -71.2643),
    "NO": (29.9511, -90.0812),
    "NYG": (40.8135, -74.0745),
    "NYJ": (40.8135, -74.0745),
    "PHI": (39.9008, -75.1675),
    "PIT": (40.4468, -80.0158),
    "SEA": (47.5952, -122.3316),
    "SF": (37.4030, -121.9700),
    "TB": (27.9759, -82.5033),
    "TEN": (36.1665, -86.7713),
    "WAS": (38.9078, -76.8645),
}

WEATHER_COLUMNS = [
    "game_id",
    "weather_available",
    "weather_source",
    "forecast_time_utc",
    "temperature_f",
    "wind_mph",
    "wind_gust_mph",
    "precipitation_probability_pct",
    "precipitation_in",
    "weather_code",
    "weather_auto_points",
]


def kickoff_utc(game: pd.Series) -> pd.Timestamp:
    """Return scheduled kickoff in UTC; nflverse gametime is Eastern."""
    stamp = pd.Timestamp(f"{game['gameday']} {game['gametime']}")
    if stamp.tzinfo is None:
        stamp = stamp.tz_localize(EASTERN)
    return stamp.tz_convert(UTC)


def _nearest_hour(payload: dict[str, Any], kickoff: pd.Timestamp) -> dict[str, object]:
    hourly = payload.get("hourly") or {}
    times = pd.to_datetime(hourly.get("time", []), errors="coerce", utc=True)
    if len(times) == 0 or times.isna().all():
        raise ValueError("weather provider returned no hourly timestamps")
    index = int(abs(times - kickoff).argmin())

    def value(name: str) -> object:
        values = hourly.get(name) or []
        return values[index] if index < len(values) else None

    return {
        "forecast_time_utc": times[index],
        "temperature_f": value("temperature_2m"),
        "wind_mph": value("wind_speed_10m"),
        "wind_gust_mph": value("wind_gusts_10m"),
        "precipitation_probability_pct": value("precipitation_probability"),
        "precipitation_in": value("precipitation"),
        "weather_code": value("weather_code"),
    }


def fetch_game_weather(
    game: pd.Series,
    *,
    timeout: float = 20.0,
    session: requests.Session | None = None,
) -> dict[str, object]:
    """Fetch the hourly forecast nearest kickoff for one home-site game."""
    if str(game.get("location", "Home")).strip().lower() == "neutral":
        raise ValueError("neutral-site coordinates are not configured")
    home_team = str(game["home_team"]).strip().upper()
    if home_team not in TEAM_COORDINATES:
        raise ValueError(f"no weather coordinates configured for {home_team}")
    latitude, longitude = TEAM_COORDINATES[home_team]
    kickoff = kickoff_utc(game)
    client = session or requests.Session()
    response = client.get(
        OPEN_METEO_URL,
        params={
            "latitude": latitude,
            "longitude": longitude,
            "hourly": HOURLY_FIELDS,
            "temperature_unit": "fahrenheit",
            "wind_speed_unit": "mph",
            "precipitation_unit": "inch",
            "timezone": "UTC",
            "forecast_days": 16,
        },
        timeout=timeout,
    )
    response.raise_for_status()
    row = _nearest_hour(response.json(), kickoff)
    return {
        "game_id": game["game_id"],
        "weather_available": True,
        "weather_source": "open-meteo",
        **row,
        "weather_auto_points": 0.0,
    }


def fetch_slate_weather(
    slate: pd.DataFrame,
    *,
    timeout: float = 20.0,
    session: requests.Session | None = None,
) -> pd.DataFrame:
    """Fetch safe informational weather context for every game in a slate."""
    required = {"game_id", "gameday", "gametime", "home_team"}
    missing = required.difference(slate.columns)
    if missing:
        raise ValueError(f"slate missing weather columns: {sorted(missing)}")

    rows: list[dict[str, object]] = []
    for _, game in slate.iterrows():
        try:
            rows.append(fetch_game_weather(game, timeout=timeout, session=session))
        except WEATHER_FETCH_ERRORS:
            rows.append(
                {
                    "game_id": game["game_id"],
                    "weather_available": False,
                    "weather_source": "missing",
                    "forecast_time_utc": pd.NaT,
                    "temperature_f": pd.NA,
                    "wind_mph": pd.NA,
                    "wind_gust_mph": pd.NA,
                    "precipitation_probability_pct": pd.NA,
                    "precipitation_in": pd.NA,
                    "weather_code": pd.NA,
                    "weather_auto_points": 0.0,
                }
            )
    return pd.DataFrame(rows, columns=WEATHER_COLUMNS)
