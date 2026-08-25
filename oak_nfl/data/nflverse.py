"""Download helpers for official nflverse release datasets.

The model intentionally consumes season-partitioned parquet files directly from
nflverse-data so historical inputs can be pinned by season and cached locally.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import requests

NFLVERSE_RELEASE_BASE = "https://github.com/nflverse/nflverse-data/releases/download"


def pbp_url(season: int) -> str:
    _validate_season(season, minimum=1999)
    return f"{NFLVERSE_RELEASE_BASE}/pbp/play_by_play_{season}.parquet"


def roster_url(season: int) -> str:
    _validate_season(season, minimum=1920)
    return f"{NFLVERSE_RELEASE_BASE}/rosters/roster_{season}.parquet"


def injury_url(season: int) -> str:
    _validate_season(season, minimum=2009)
    return f"{NFLVERSE_RELEASE_BASE}/injuries/injuries_{season}.parquet"


def snap_counts_url(season: int) -> str:
    _validate_season(season, minimum=2012)
    return f"{NFLVERSE_RELEASE_BASE}/snap_counts/snap_counts_{season}.parquet"


def players_url() -> str:
    """Return the current nflverse player ID crosswalk."""
    return f"{NFLVERSE_RELEASE_BASE}/players/players.csv"


def download_file(url: str, destination: str | Path, *, timeout: int = 120) -> Path:
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    with requests.get(url, stream=True, timeout=timeout) as response:
        response.raise_for_status()
        with temporary.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)
    temporary.replace(destination)
    return destination


def _load_parquet(url: str, path: Path) -> pd.DataFrame:
    if not path.exists():
        download_file(url, path)
    return pd.read_parquet(path)


def load_pbp(season: int, cache_dir: str | Path = "data/raw/pbp") -> pd.DataFrame:
    path = Path(cache_dir) / f"play_by_play_{season}.parquet"
    return _load_parquet(pbp_url(season), path)


def load_roster(season: int, cache_dir: str | Path = "data/raw/rosters") -> pd.DataFrame:
    path = Path(cache_dir) / f"roster_{season}.parquet"
    return _load_parquet(roster_url(season), path)


def load_injuries(season: int, cache_dir: str | Path = "data/raw/injuries") -> pd.DataFrame:
    path = Path(cache_dir) / f"injuries_{season}.parquet"
    return _load_parquet(injury_url(season), path)


def load_snap_counts(season: int, cache_dir: str | Path = "data/raw/snap_counts") -> pd.DataFrame:
    path = Path(cache_dir) / f"snap_counts_{season}.parquet"
    return _load_parquet(snap_counts_url(season), path)


def load_players(cache_dir: str | Path = "data/raw/players") -> pd.DataFrame:
    """Load nflverse's GSIS/PFR player ID crosswalk."""
    path = Path(cache_dir) / "players.csv"
    if not path.exists():
        download_file(players_url(), path)
    return pd.read_csv(path, dtype=str, low_memory=False)


def _validate_season(season: int, *, minimum: int) -> None:
    if not isinstance(season, int):
        raise TypeError("season must be an integer")
    if season < minimum or season > 2100:
        raise ValueError(f"season must be between {minimum} and 2100")
