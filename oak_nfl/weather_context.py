"""Informational pregame weather/environment context for Oak live previews."""

from __future__ import annotations

import pandas as pd

from oak_nfl.data.weather import fetch_slate_weather


def build_game_weather_context(
    slate: pd.DataFrame,
    *,
    weather: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Return auditable weather context without changing prediction points."""
    required = {"game_id", "home_team", "away_team"}
    missing = required.difference(slate.columns)
    if missing:
        raise ValueError(f"slate missing required columns: {sorted(missing)}")

    forecast = weather if weather is not None else fetch_slate_weather(slate)
    out = slate[["game_id", "home_team", "away_team"]].copy()
    for optional in ("location", "roof", "surface", "stadium"):
        if optional in slate.columns:
            out[optional] = slate[optional]

    out = out.merge(forecast, on="game_id", how="left", validate="one_to_one")
    out["weather_available"] = out["weather_available"].fillna(False).astype(bool)
    out["weather_source"] = out["weather_source"].fillna("missing")
    out["weather_auto_points"] = 0.0
    return out
