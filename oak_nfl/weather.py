"""Pregame weather features for Oak V11 totals modeling.

The module is intentionally provider-agnostic: historical or live weather feeds
should first be normalized into one row per game, then transformed into stable,
auditable features. Indoor/closed-roof games receive no outdoor weather effect.
"""

from __future__ import annotations

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

    # Continuous thresholds let historical validation learn whether effects are
    # gradual or concentrated in genuinely adverse conditions.
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
