"""Pregame weather and acclimation features for Oak V11 totals modeling."""

from __future__ import annotations

from collections import defaultdict, deque

import numpy as np
import pandas as pd

WEATHER_COLUMNS = ["game_id", "temperature_f", "wind_mph", "precipitation_in", "is_dome"]


def normalize_weather(frame: pd.DataFrame) -> pd.DataFrame:
    missing = set(WEATHER_COLUMNS).difference(frame.columns)
    if missing:
        raise ValueError(f"weather feed missing required columns: {sorted(missing)}")
    out = frame[WEATHER_COLUMNS].copy()
    out["temperature_f"] = pd.to_numeric(out["temperature_f"], errors="coerce")
    out["wind_mph"] = pd.to_numeric(out["wind_mph"], errors="coerce").clip(lower=0.0)
    out["precipitation_in"] = pd.to_numeric(out["precipitation_in"], errors="coerce").clip(lower=0.0)
    out["is_dome"] = out["is_dome"].fillna(False).astype(bool)
    return out.drop_duplicates("game_id", keep="last").reset_index(drop=True)


def build_weather_features(weather: pd.DataFrame) -> pd.DataFrame:
    out = normalize_weather(weather)
    outdoor = ~out["is_dome"]
    wind = out["wind_mph"].fillna(0.0)
    temp = out["temperature_f"]
    precip = out["precipitation_in"].fillna(0.0)
    out["wind_over_10"] = np.where(outdoor, np.maximum(wind - 10.0, 0.0), 0.0)
    out["wind_over_15"] = np.where(outdoor, np.maximum(wind - 15.0, 0.0), 0.0)
    out["precipitation"] = np.where(outdoor, precip, 0.0)
    out["cold_below_32"] = np.where(outdoor & temp.notna(), np.maximum(32.0 - temp.fillna(32.0), 0.0), 0.0)
    out["heat_above_85"] = np.where(outdoor & temp.notna(), np.maximum(temp.fillna(85.0) - 85.0, 0.0), 0.0)
    out["outdoor"] = outdoor.astype(int)
    return out[["game_id", "temperature_f", "wind_mph", "precipitation_in", "is_dome", "outdoor", "wind_over_10", "wind_over_15", "precipitation", "cold_below_32", "heat_above_85"]]


def build_temperature_acclimation(games: pd.DataFrame, *, history_games: int = 8) -> pd.DataFrame:
    """Build leakage-safe climate mismatch from prior home climate and recent exposure."""
    required = {"game_id", "season", "week", "home_team", "away_team", "temperature_f", "is_dome"}
    missing = required.difference(games.columns)
    if missing:
        raise ValueError(f"games missing acclimation columns: {sorted(missing)}")
    if history_games < 1:
        raise ValueError("history_games must be at least 1")

    frame = games.copy()
    frame["temperature_f"] = pd.to_numeric(frame["temperature_f"], errors="coerce")
    frame["is_dome"] = frame["is_dome"].fillna(False).astype(bool)
    frame = frame.sort_values(["season", "week", "game_id"]).reset_index(drop=True)
    home_temps = defaultdict(lambda: deque(maxlen=history_games))
    home_indoor = defaultdict(lambda: deque(maxlen=history_games))
    recent_temps = defaultdict(lambda: deque(maxlen=history_games))
    recent_outdoor = defaultdict(lambda: deque(maxlen=history_games))
    rows = []

    def mean_or_nan(values):
        return float(np.mean(values)) if values else np.nan

    for row in frame.itertuples(index=False):
        temp = float(row.temperature_f) if pd.notna(row.temperature_f) else np.nan
        is_dome = bool(row.is_dome)
        outdoor_with_temp = (not is_dome) and pd.notna(temp)
        home_mean = mean_or_nan(home_temps[row.home_team])
        away_mean = mean_or_nan(home_temps[row.away_team])
        home_recent = mean_or_nan(recent_temps[row.home_team])
        away_recent = mean_or_nan(recent_temps[row.away_team])
        home_indoor_share = mean_or_nan(home_indoor[row.home_team])
        away_indoor_share = mean_or_nan(home_indoor[row.away_team])
        home_outdoor_share = mean_or_nan(recent_outdoor[row.home_team])
        away_outdoor_share = mean_or_nan(recent_outdoor[row.away_team])

        def shocks(baseline):
            if not outdoor_with_temp or pd.isna(baseline):
                return 0.0, 0.0
            return max(baseline - temp, 0.0), max(temp - baseline, 0.0)

        hc, hh = shocks(home_mean); ac, ah = shocks(away_mean)
        hrc, hrh = shocks(home_recent); arc, arh = shocks(away_recent)
        rows.append({
            "game_id": row.game_id,
            "home_climate_temp": home_mean, "away_climate_temp": away_mean,
            "home_recent_temp": home_recent, "away_recent_temp": away_recent,
            "home_indoor_home_share": home_indoor_share, "away_indoor_home_share": away_indoor_share,
            "home_recent_outdoor_share": home_outdoor_share, "away_recent_outdoor_share": away_outdoor_share,
            "home_cold_shock": hc, "away_cold_shock": ac, "home_heat_shock": hh, "away_heat_shock": ah,
            "home_recent_cold_shock": hrc, "away_recent_cold_shock": arc,
            "home_recent_heat_shock": hrh, "away_recent_heat_shock": arh,
            "cold_shock_difference": ac - hc, "heat_shock_difference": ah - hh,
            "away_indoor_to_outdoor": float(away_indoor_share) if outdoor_with_temp and pd.notna(away_indoor_share) else 0.0,
            "away_low_outdoor_exposure": (1.0 - float(away_outdoor_share)) if outdoor_with_temp and pd.notna(away_outdoor_share) else 0.0,
            "away_warm_team_freeze": float(outdoor_with_temp and temp < 35 and pd.notna(away_recent) and away_recent >= 60),
            "away_cold_team_heat": float(outdoor_with_temp and temp > 85 and pd.notna(away_recent) and away_recent <= 60),
        })
        home_indoor[row.home_team].append(1.0 if is_dome else 0.0)
        if outdoor_with_temp:
            home_temps[row.home_team].append(temp)
        for team in (row.home_team, row.away_team):
            recent_outdoor[team].append(0.0 if is_dome else 1.0)
            if outdoor_with_temp:
                recent_temps[team].append(temp)
    return pd.DataFrame(rows)
