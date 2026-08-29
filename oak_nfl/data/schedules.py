"""Current NFL schedule and market-line ingestion for production runs."""

from __future__ import annotations

from datetime import UTC, date, datetime
from pathlib import Path

import pandas as pd

from oak_nfl.data.nflverse import download_file

SCHEDULE_URL = "https://github.com/nflverse/nflverse-data/releases/download/schedules/games.csv"


def load_schedules(
    cache_path: str | Path = "data/raw/schedules/games.csv",
    *,
    refresh: bool = False,
) -> pd.DataFrame:
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
    return out.sort_values(
        ["gameday", "gametime", "game_id"], na_position="last"
    ).reset_index(drop=True)


def load_next_slate(
    *,
    as_of: date | str | None = None,
    refresh: bool = False,
) -> pd.DataFrame:
    """Return the next not-yet-completed NFL week from the current schedule."""
    games = load_schedules(refresh=refresh).copy()
    required = {"season", "week", "gameday", "home_team", "away_team", "game_id"}
    missing = required.difference(games.columns)
    if missing:
        raise ValueError(f"schedule data missing required columns: {sorted(missing)}")

    current_date = datetime.now(UTC).date()
    cutoff = pd.Timestamp(as_of or current_date)
    games["game_date"] = pd.to_datetime(games["gameday"], errors="coerce")
    future = games[games["game_date"].ge(cutoff)].copy()
    if "home_score" in future.columns and "away_score" in future.columns:
        future = future[future["home_score"].isna() | future["away_score"].isna()]
    if future.empty:
        raise ValueError("no upcoming NFL games found in schedule data")

    first = future.sort_values(["game_date", "game_id"]).iloc[0]
    return load_week(int(first["season"]), int(first["week"]), refresh=False)
