"""Pregame weather and acclimation features for Oak V11 totals modeling."""

from __future__ import annotations

from collections import defaultdict, deque

import numpy as np
import pandas as pd

WEATHER_COLUMNS = [
    "game_id",
    "temperature_f",
    "wind_mph",
    "precipitation_in",
    "is_dome",
]


def normalize_weather(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize a game-level weather feed into Oak's canonical schema."""
    missing = set(WEATHER_COLUMNS).difference(frame.columns)
    if missing:
        raise ValueError(f"weather feed missing required columns: {sorted(missing)}")

    out = frame[WEATHER_COLUMNS].copy()
    out["temperature_f"] = pd.to_numeric(out["temperature_f"], errors="coerce")
    out["wind_mph"] = pd.to_numeric(out["wind_mph"], errors="coerce").clip(lower=0.0)
    out["precipitation_in"] = (
        pd.to_numeric(out["precipitation_in"], errors="coerce").clip(lower=0.0)
    )
    out["is_dome"] = out["is_dome"].fillna(False).astype(bool)
    return out.drop_duplicates("game_id", keep="last").reset_index(drop=True)


def build_weather_features(weather: pd.DataFrame) -> pd.DataFrame:
    """Create continuous weather severity features without imposing point values."""
    out = normalize_weather(weather)
    outdoor = ~out["is_dome"]

    wind = out["wind_mph"].fillna(0.0)
    temp = out["temperature_f"]
    precip = out["precipitation_in"].fillna(0.0)

    out["wind_over_10"] = np.where(outdoor, np.maximum(wind - 10.0, 0.0), 0.0)
    out["wind_over_15"] = np.where(outdoor, np.maximum(wind - 15.0, 0.0), 0.0)
    out["precipitation"] = np.where(outdoor, precip, 0.0)
    out["cold_below_32"] = np.where(
        outdoor & temp.notna(), np.maximum(32.0 - temp.fillna(32.0), 0.0), 0.0
    )
    out["heat_above_85"] = np.where(
        outdoor & temp.notna(), np.maximum(temp.fillna(85.0) - 85.0, 0.0), 0.0
    )
    out["outdoor"] = outdoor.astype(int)

    return out[
        [
            "game_id",
            "temperature_f",
            "wind_mph",
            "precipitation_in",
            "is_dome",
            "outdoor",
            "wind_over_10",
            "wind_over_15",
            "precipitation",
            "cold_below_32",
            "heat_above_85",
        ]
    ]


def build_temperature_acclimation(
    games: pd.DataFrame,
    *,
    history_games: int = 8,
) -> pd.DataFrame:
    """Build leakage-safe team climate-mismatch features.

    Each team's baseline comes only from its *previous home games*. Outdoor/open
    home games contribute temperature; all prior home games contribute to the
    indoor-home share. The current game's weather is never inserted into either
    team's baseline until after its features have been emitted.
    """
    required = {
        "game_id",
        "season",
        "week",
        "home_team",
        "away_team",
        "temperature_f",
        "is_dome",
    }
    missing = required.difference(games.columns)
    if missing:
        raise ValueError(f"games missing acclimation columns: {sorted(missing)}")
    if history_games < 1:
        raise ValueError("history_games must be at least 1")

    frame = games.copy()
    frame["temperature_f"] = pd.to_numeric(frame["temperature_f"], errors="coerce")
    frame["is_dome"] = frame["is_dome"].fillna(False).astype(bool)
    sort_cols = ["season", "week", "game_id"]
    frame = frame.sort_values(sort_cols).reset_index(drop=True)

    home_temps: dict[str, deque[float]] = defaultdict(lambda: deque(maxlen=history_games))
    home_indoor: dict[str, deque[float]] = defaultdict(lambda: deque(maxlen=history_games))
    rows: list[dict[str, object]] = []

    for row in frame.itertuples(index=False):
        temp = float(row.temperature_f) if pd.notna(row.temperature_f) else np.nan
        is_dome = bool(row.is_dome)
        outdoor_with_temp = (not is_dome) and pd.notna(temp)

        def profile(team: str) -> tuple[float, float, int]:
            temps = list(home_temps[team])
            indoors = list(home_indoor[team])
            mean_temp = float(np.mean(temps)) if temps else np.nan
            indoor_share = float(np.mean(indoors)) if indoors else np.nan
            return mean_temp, indoor_share, len(temps)

        home_mean, home_indoor_share, home_temp_games = profile(row.home_team)
        away_mean, away_indoor_share, away_temp_games = profile(row.away_team)

        if outdoor_with_temp:
            home_cold_shock = max(home_mean - temp, 0.0) if pd.notna(home_mean) else 0.0
            away_cold_shock = max(away_mean - temp, 0.0) if pd.notna(away_mean) else 0.0
            home_heat_shock = max(temp - home_mean, 0.0) if pd.notna(home_mean) else 0.0
            away_heat_shock = max(temp - away_mean, 0.0) if pd.notna(away_mean) else 0.0
        else:
            home_cold_shock = away_cold_shock = 0.0
            home_heat_shock = away_heat_shock = 0.0

        rows.append(
            {
                "game_id": row.game_id,
                "home_climate_temp": home_mean,
                "away_climate_temp": away_mean,
                "home_climate_games": home_temp_games,
                "away_climate_games": away_temp_games,
                "home_indoor_home_share": home_indoor_share,
                "away_indoor_home_share": away_indoor_share,
                "home_cold_shock": home_cold_shock,
                "away_cold_shock": away_cold_shock,
                "home_heat_shock": home_heat_shock,
                "away_heat_shock": away_heat_shock,
                "cold_shock_difference": away_cold_shock - home_cold_shock,
                "heat_shock_difference": away_heat_shock - home_heat_shock,
                "away_indoor_to_outdoor": (
                    float(away_indoor_share) if outdoor_with_temp and pd.notna(away_indoor_share) else 0.0
                ),
            }
        )

        home_indoor[row.home_team].append(1.0 if is_dome else 0.0)
        if outdoor_with_temp:
            home_temps[row.home_team].append(temp)

    return pd.DataFrame(rows)
