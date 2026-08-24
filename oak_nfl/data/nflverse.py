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
    """Return the nflverse parquet URL for one season of play-by-play data."""
    _validate_season(season, minimum=1999)
    return f"{NFLVERSE_RELEASE_BASE}/pbp/play_by_play_{season}.parquet"


def roster_url(season: int) -> str:
    """Return the nflverse parquet URL for one season of roster data."""
    _validate_season(season, minimum=1920)
    return f"{NFLVERSE_RELEASE_BASE}/rosters/roster_{season}.parquet"


def download_file(url: str, destination: str | Path, *, timeout: int = 120) -> Path:
    """Download a remote file atomically enough for local pipeline use."""
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


def load_pbp(season: int, cache_dir: str | Path = "data/raw/pbp") -> pd.DataFrame:
    """Load a season of nflverse play-by-play data, downloading it if necessary."""
    path = Path(cache_dir) / f"play_by_play_{season}.parquet"
    if not path.exists():
        download_file(pbp_url(season), path)
    return pd.read_parquet(path)


def load_roster(season: int, cache_dir: str | Path = "data/raw/rosters") -> pd.DataFrame:
    """Load a season of nflverse roster data, downloading it if necessary."""
    path = Path(cache_dir) / f"roster_{season}.parquet"
    if not path.exists():
        download_file(roster_url(season), path)
    return pd.read_parquet(path)


def _validate_season(season: int, *, minimum: int) -> None:
    if not isinstance(season, int):
        raise TypeError("season must be an integer")
    if season < minimum or season > 2100:
        raise ValueError(f"season must be between {minimum} and 2100")
