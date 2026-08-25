"""Normalize injury/availability feeds into Oak's provider-agnostic schema."""

from __future__ import annotations

import pandas as pd

CANONICAL_COLUMNS = [
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

STATUS_ALIASES = {
    "out": "out",
    "doubtful": "doubtful",
    "questionable": "questionable",
    "limited": "limited",
    "limited participation": "limited",
    "full": "full",
    "full participation": "full",
    "active": "active",
}

POSITION_ALIASES = {
    "T": "OT",
    "OT": "OT",
    "LT": "OT",
    "RT": "OT",
    "G": "IOL",
    "C": "IOL",
    "OG": "IOL",
    "OL": "IOL",
    "WR": "WR",
    "TE": "TE",
    "RB": "RB",
    "FB": "RB",
    "DE": "EDGE",
    "EDGE": "EDGE",
    "OLB": "EDGE",
    "DT": "IDL",
    "NT": "IDL",
    "IDL": "IDL",
    "LB": "LB",
    "ILB": "LB",
    "CB": "CB",
    "DB": "CB",
    "S": "S",
    "FS": "S",
    "SS": "S",
    "QB": "QB",
}


def normalize_status(value: object) -> str:
    """Map provider-specific availability labels to Oak's canonical status."""
    text = str(value).strip().lower()
    return STATUS_ALIASES.get(text, "active")


def normalize_position(value: object) -> str:
    """Map detailed roster positions into Oak's modeling groups."""
    text = str(value).strip().upper()
    return POSITION_ALIASES.get(text, text or "UNK")


def normalize_injury_feed(
    frame: pd.DataFrame,
    *,
    column_map: dict[str, str] | None = None,
    source: str = "unknown",
) -> pd.DataFrame:
    """Return any provider feed in Oak's canonical weekly availability format.

    ``column_map`` maps provider column names to Oak names, e.g.
    ``{"club": "team", "designation": "status"}``.
    """
    data = frame.rename(columns=column_map or {}).copy()
    required = {"season", "week", "team", "player_name", "position_group", "status"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"injury feed missing required columns: {sorted(missing)}")

    if "player_id" not in data.columns:
        data["player_id"] = pd.NA
    if "report_date" not in data.columns:
        data["report_date"] = pd.NaT
    if "source" not in data.columns:
        data["source"] = source
    else:
        data["source"] = data["source"].fillna(source)

    data["status"] = data["status"].map(normalize_status)
    data["position_group"] = data["position_group"].map(normalize_position)
    data["team"] = data["team"].astype(str).str.upper().str.strip()
    data["season"] = pd.to_numeric(data["season"], errors="raise").astype(int)
    data["week"] = pd.to_numeric(data["week"], errors="raise").astype(int)
    data["report_date"] = pd.to_datetime(data["report_date"], errors="coerce")
    return data[CANONICAL_COLUMNS].drop_duplicates().reset_index(drop=True)


def latest_weekly_status(normalized: pd.DataFrame) -> pd.DataFrame:
    """Keep the latest known report per player/season/week without inventing health."""
    required = set(CANONICAL_COLUMNS)
    missing = required.difference(normalized.columns)
    if missing:
        raise ValueError(f"normalized injury data missing columns: {sorted(missing)}")
    order = normalized.sort_values(
        ["season", "week", "team", "player_name", "report_date"],
        na_position="first",
    )
    return order.drop_duplicates(
        ["season", "week", "team", "player_name"], keep="last"
    ).reset_index(drop=True)
