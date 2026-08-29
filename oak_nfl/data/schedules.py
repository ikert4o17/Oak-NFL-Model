"""Current NFL schedule and market-line ingestion for production runs."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from oak_nfl.data.nflverse import download_file

SCHEDULE_URL = "https://github.com/nflverse/nflverse-data/releases/download/schedules/games.csv"


def load_schedules(cache_path: str | Path = "data/raw/schedules/games.csv", *, refresh: bool = False) -> pd.DataFrame:
    """Load nflverse's schedule file, including future games and market fields."""
    path = Path(cache_path)
    if refresh or not path.exists():
        download_file(SCHEDULE_URL, path)
    return pd.read_csv(path, low_memory=False)


def load_week(season: int, week: int, *, refresh: bool = False) -> pd.DataFrame:
    """Return one regular/postseason schedule slice for a season and week."""
    games = load_schedules(refresh=refresh)
    required = {"game_id", "season", "week", "home_team", "away_team"}
    missing = required.difference(games.columns)
    if missing:
        raise ValueError(f"schedule data missing required columns: {sorted(missing)}")
    out = games.loc[games["season"].eq(season) & games["week"].eq(week)].copy()
    return out.sort_values(["gameday", "gametime", "game_id"], na_position="last").reset_index(drop=True)
