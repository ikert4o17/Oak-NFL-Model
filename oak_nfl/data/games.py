"""Game-level extraction helpers from nflverse play-by-play."""

from __future__ import annotations

import pandas as pd


_REQUIRED_GAME_COLUMNS = {
    "game_id",
    "season",
    "week",
    "home_team",
    "away_team",
    "home_score",
    "away_score",
}


def build_game_results(pbp: pd.DataFrame) -> pd.DataFrame:
    """Return one completed-game row per game from play-by-play records."""
    missing = _REQUIRED_GAME_COLUMNS.difference(pbp.columns)
    if missing:
        raise ValueError(f"play-by-play data missing game columns: {sorted(missing)}")

    games = (
        pbp[[*sorted(_REQUIRED_GAME_COLUMNS)]]
        .drop_duplicates(subset=["game_id"], keep="last")
        .copy()
    )
    games = games[games["home_score"].notna() & games["away_score"].notna()]
    games["actual_home_margin"] = games["home_score"] - games["away_score"]
    return games.sort_values(["season", "week", "game_id"]).reset_index(drop=True)
