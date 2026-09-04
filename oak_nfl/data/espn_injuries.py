"""Adapter for ESPN's public NFL injury endpoint.

ESPN is used as a replaceable live-data provider, not as model logic. The
adapter converts the response into Oak's canonical injury schema and preserves
source/report timestamps where available. Any malformed or ambiguous status is
kept as ``unknown`` by the canonical normalizer.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pandas as pd
import requests

from oak_nfl.data.injuries import normalize_injury_feed

ESPN_INJURIES_URL = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/injuries"

# ESPN's numeric NFL team IDs are stable provider identifiers and are useful
# when the league-wide injury payload omits a team abbreviation.
ESPN_TEAM_ID_TO_ABBR = {
    "1": "ATL",
    "2": "BUF",
    "3": "CHI",
    "4": "CIN",
    "5": "CLE",
    "6": "DAL",
    "7": "DEN",
    "8": "DET",
    "9": "GB",
    "10": "TEN",
    "11": "IND",
    "12": "KC",
    "13": "LV",
    "14": "LA",
    "15": "MIA",
    "16": "MIN",
    "17": "NE",
    "18": "NO",
    "19": "NYG",
    "20": "NYJ",
    "21": "PHI",
    "22": "ARI",
    "23": "PIT",
    "24": "LAC",
    "25": "SF",
    "26": "SEA",
    "27": "TB",
    "28": "WAS",
    "29": "CAR",
    "30": "JAX",
    "33": "BAL",
    "34": "HOU",
}

TEAM_NAME_TO_ABBR = {
    "arizona cardinals": "ARI",
    "atlanta falcons": "ATL",
    "baltimore ravens": "BAL",
    "buffalo bills": "BUF",
    "carolina panthers": "CAR",
    "chicago bears": "CHI",
    "cincinnati bengals": "CIN",
    "cleveland browns": "CLE",
    "dallas cowboys": "DAL",
    "denver broncos": "DEN",
    "detroit lions": "DET",
    "green bay packers": "GB",
    "houston texans": "HOU",
    "indianapolis colts": "IND",
    "jacksonville jaguars": "JAX",
    "kansas city chiefs": "KC",
    "las vegas raiders": "LV",
    "los angeles chargers": "LAC",
    "los angeles rams": "LA",
    "miami dolphins": "MIA",
    "minnesota vikings": "MIN",
    "new england patriots": "NE",
    "new orleans saints": "NO",
    "new york giants": "NYG",
    "new york jets": "NYJ",
    "philadelphia eagles": "PHI",
    "pittsburgh steelers": "PIT",
    "san francisco 49ers": "SF",
    "seattle seahawks": "SEA",
    "tampa bay buccaneers": "TB",
    "tennessee titans": "TEN",
    "washington commanders": "WAS",
}


def _status_text(item: dict[str, Any]) -> object:
    status = item.get("status")
    if isinstance(status, dict):
        return status.get("name") or status.get("description") or status.get("type")
    return status


def _team_abbreviation(team_block: dict[str, Any]) -> object:
    """Read ESPN team identity across known league-wide injury schemas."""
    team = team_block.get("team")
    candidates: list[dict[str, Any]] = []
    if isinstance(team, dict):
        candidates.append(team)
    candidates.append(team_block)

    for candidate in candidates:
        abbreviation = candidate.get("abbreviation")
        if abbreviation:
            return abbreviation

    for candidate in candidates:
        team_id = candidate.get("id")
        if team_id is not None:
            abbreviation = ESPN_TEAM_ID_TO_ABBR.get(str(team_id))
            if abbreviation:
                return abbreviation

    for candidate in candidates:
        for key in ("displayName", "name", "shortDisplayName"):
            name = candidate.get(key)
            if isinstance(name, str):
                abbreviation = TEAM_NAME_TO_ABBR.get(name.strip().lower())
                if abbreviation:
                    return abbreviation

    return None


def parse_espn_injuries(
    payload: dict[str, Any],
    *,
    fetched_at: datetime | None = None,
    season: int | None = None,
    week: int | None = None,
) -> pd.DataFrame:
    """Convert one ESPN injury response into Oak's canonical injury frame."""
    fetched_at = fetched_at or datetime.now(UTC)
    season_obj = payload.get("season") or {}
    resolved_season = season if season is not None else season_obj.get("year")
    resolved_week = week if week is not None else payload.get("week")
    if isinstance(resolved_week, dict):
        resolved_week = resolved_week.get("number")

    rows: list[dict[str, object]] = []
    for team_block in payload.get("injuries") or []:
        team_abbr = _team_abbreviation(team_block)
        for injury in team_block.get("injuries") or []:
            athlete = injury.get("athlete") or {}
            position = athlete.get("position") or {}
            report_date = injury.get("date") or injury.get("lastUpdated") or fetched_at.isoformat()
            rows.append(
                {
                    "season": resolved_season,
                    "week": resolved_week,
                    "team": team_abbr,
                    "player_id": athlete.get("id"),
                    "player_name": athlete.get("fullName") or athlete.get("displayName"),
                    "position_group": position.get("abbreviation"),
                    "status": _status_text(injury),
                    "report_date": report_date,
                    "source": "espn",
                }
            )

    columns = [
        "season",
        "week",
        "team",
        "player_id",
        "player_name",
        "position_group",
        "status",
        "report_date",
        "source",
    ]
    raw = pd.DataFrame(rows, columns=columns)
    if raw.empty:
        return raw
    if raw["season"].isna().any() or raw["week"].isna().any():
        raise ValueError("ESPN injury payload did not provide season/week context")
    if raw["team"].isna().any():
        raise ValueError("ESPN injury payload did not provide a recognized team identity")
    return normalize_injury_feed(raw, source="espn")


def fetch_espn_injuries(
    *,
    season: int,
    week: int,
    timeout: float = 20.0,
    session: requests.Session | None = None,
) -> pd.DataFrame:
    """Fetch current ESPN NFL injuries and normalize them for Oak.

    ``season`` and ``week`` are supplied by Oak's slate rather than inferred from
    wall-clock time. This keeps weekly snapshots deterministic and auditable.
    """
    client = session or requests.Session()
    response = client.get(ESPN_INJURIES_URL, timeout=timeout)
    response.raise_for_status()
    return parse_espn_injuries(response.json(), season=season, week=week)
