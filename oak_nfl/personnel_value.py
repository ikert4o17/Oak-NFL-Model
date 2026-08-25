"""Point-in-time non-QB player importance estimated from prior snap participation."""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from oak_nfl.data.injuries import normalize_position

OFFENSE_GROUPS = {"OT", "IOL", "WR", "TE", "RB", "QB"}
DEFENSE_GROUPS = {"EDGE", "IDL", "LB", "CB", "S"}


def _weighted_average(values: list[float], decay: float) -> float:
    clean = np.asarray([value for value in values if pd.notna(value)], dtype=float)
    if clean.size == 0:
        return 0.0
    ages = np.arange(clean.size - 1, -1, -1, dtype=float)
    return float(np.average(clean, weights=np.power(decay, ages)))


def _clean_name(value: object) -> str:
    text = str(value).lower().strip()
    text = re.sub(r"[^a-z0-9 ]+", "", text)
    text = re.sub(r"\b(jr|sr|ii|iii|iv|v)\b", "", text)
    return " ".join(text.split())


def attach_snap_player_ids(snap_counts: pd.DataFrame, players: pd.DataFrame) -> pd.DataFrame:
    """Attach GSIS IDs to PFR snap-count rows using nflverse's player crosswalk."""
    if "pfr_player_id" not in snap_counts.columns:
        raise ValueError("snap counts missing required column: pfr_player_id")
    required = {"gsis_id", "pfr_id"}
    missing = required.difference(players.columns)
    if missing:
        raise ValueError(f"players crosswalk missing required columns: {sorted(missing)}")
    crosswalk = (
        players[["gsis_id", "pfr_id"]]
        .dropna()
        .drop_duplicates("pfr_id")
        .rename(columns={"gsis_id": "player_id", "pfr_id": "pfr_player_id"})
    )
    out = snap_counts.merge(crosswalk, on="pfr_player_id", how="left", validate="many_to_one")
    return out


def build_pregame_player_values(
    snap_counts: pd.DataFrame,
    *,
    recency_decay: float = 0.90,
    minimum_prior_games: int = 1,
) -> pd.DataFrame:
    required = {
        "game_id", "season", "week", "team", "player", "position", "offense_pct", "defense_pct"
    }
    missing = required.difference(snap_counts.columns)
    if missing:
        raise ValueError(f"snap counts missing required columns: {sorted(missing)}")
    if not 0 < recency_decay <= 1:
        raise ValueError("recency_decay must be in (0, 1]")

    snaps = snap_counts.sort_values(["season", "week", "game_id", "team", "player"]).copy()
    if "player_id" not in snaps.columns:
        snaps["player_id"] = pd.NA
    snaps["position_group"] = snaps["position"].map(normalize_position)
    offense = pd.to_numeric(snaps["offense_pct"], errors="coerce").fillna(0.0)
    defense = pd.to_numeric(snaps["defense_pct"], errors="coerce").fillna(0.0)
    snaps["snap_share"] = np.where(
        snaps["position_group"].isin(OFFENSE_GROUPS),
        offense,
        np.where(snaps["position_group"].isin(DEFENSE_GROUPS), defense, np.maximum(offense, defense)),
    )
    snaps.loc[snaps["snap_share"].gt(1.0), "snap_share"] /= 100.0
    snaps["snap_share"] = snaps["snap_share"].clip(0.0, 1.0)

    histories: dict[tuple[str, str], list[float]] = {}
    rows: list[dict[str, object]] = []
    for (season, week), week_rows in snaps.groupby(["season", "week"], sort=True):
        for row in week_rows.itertuples(index=False):
            identity = str(row.player_id) if pd.notna(row.player_id) else _clean_name(row.player)
            key = (str(row.team), identity)
            history = histories.get(key, [])
            value = _weighted_average(history, recency_decay) if len(history) >= minimum_prior_games else 0.0
            rows.append({
                "game_id": row.game_id,
                "season": int(season),
                "week": int(week),
                "team": row.team,
                "player_id": row.player_id,
                "player_name": row.player,
                "position_group": row.position_group,
                "player_value": float(np.clip(value, 0.0, 1.0)),
                "prior_games": len(history),
            })
        for row in week_rows.itertuples(index=False):
            identity = str(row.player_id) if pd.notna(row.player_id) else _clean_name(row.player)
            histories.setdefault((str(row.team), identity), []).append(float(row.snap_share))
    return pd.DataFrame(rows)


def attach_player_values(availability: pd.DataFrame, player_values: pd.DataFrame) -> pd.DataFrame:
    """Join injury rows to pregame values, preferring stable GSIS IDs then names."""
    required_availability = {"season", "week", "team", "player_id", "player_name", "position_group", "status"}
    missing = required_availability.difference(availability.columns)
    if missing:
        raise ValueError(f"availability missing required columns: {sorted(missing)}")
    required_values = {"season", "week", "team", "player_id", "player_name", "player_value"}
    missing_values = required_values.difference(player_values.columns)
    if missing_values:
        raise ValueError(f"player values missing required columns: {sorted(missing_values)}")

    out = availability.copy()
    values = player_values.copy()
    out["_clean_name"] = out["player_name"].map(_clean_name)
    values["_clean_name"] = values["player_name"].map(_clean_name)

    by_id = values.dropna(subset=["player_id"])[
        ["season", "week", "team", "player_id", "player_value"]
    ].drop_duplicates(["season", "week", "team", "player_id"])
    out = out.merge(
        by_id.rename(columns={"player_value": "player_value_id"}),
        on=["season", "week", "team", "player_id"],
        how="left",
    )

    by_name = values[["season", "week", "team", "_clean_name", "player_value"]].drop_duplicates(
        ["season", "week", "team", "_clean_name"]
    )
    out = out.merge(
        by_name.rename(columns={"player_value": "player_value_name"}),
        on=["season", "week", "team", "_clean_name"],
        how="left",
    )
    out["player_value"] = out["player_value_id"].combine_first(out["player_value_name"])
    out["player_value"] = pd.to_numeric(out["player_value"], errors="coerce").fillna(0.0).clip(0.0, 1.0)
    return out.drop(columns=["_clean_name", "player_value_id", "player_value_name"])
