"""Point-in-time non-QB player importance estimated from prior snap participation."""

from __future__ import annotations

import bisect
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
    return snap_counts.merge(crosswalk, on="pfr_player_id", how="left", validate="many_to_one")


def build_pregame_player_values(
    snap_counts: pd.DataFrame,
    *,
    recency_decay: float = 0.90,
    minimum_prior_games: int = 1,
    prior_season_games: int = 8,
    prior_season_weight: float = 0.75,
) -> pd.DataFrame:
    """Build leakage-safe player values from completed snap history.

    ``player_value`` is the value before the row's game. ``postgame_player_value``
    includes that completed game's snap share and exists only so a later injury
    week can carry forward the latest known workload when the player has no snap
    row in the injury game itself.
    """
    required = {
        "game_id",
        "season",
        "week",
        "team",
        "player",
        "position",
        "offense_pct",
        "defense_pct",
    }
    missing = required.difference(snap_counts.columns)
    if missing:
        raise ValueError(f"snap counts missing required columns: {sorted(missing)}")
    if not 0 < recency_decay <= 1:
        raise ValueError("recency_decay must be in (0, 1]")
    if prior_season_games < 0:
        raise ValueError("prior_season_games must be non-negative")
    if not 0 <= prior_season_weight <= 1:
        raise ValueError("prior_season_weight must be in [0, 1]")

    snaps = snap_counts.sort_values(["season", "week", "game_id", "team", "player"]).copy()
    if "player_id" not in snaps.columns:
        snaps["player_id"] = pd.NA
    snaps["position_group"] = snaps["position"].map(normalize_position)
    offense = pd.to_numeric(snaps["offense_pct"], errors="coerce").fillna(0.0)
    defense = pd.to_numeric(snaps["defense_pct"], errors="coerce").fillna(0.0)
    snaps["snap_share"] = np.where(
        snaps["position_group"].isin(OFFENSE_GROUPS),
        offense,
        np.where(
            snaps["position_group"].isin(DEFENSE_GROUPS),
            defense,
            np.maximum(offense, defense),
        ),
    )
    snaps.loc[snaps["snap_share"].gt(1.0), "snap_share"] /= 100.0
    snaps["snap_share"] = snaps["snap_share"].clip(0.0, 1.0)

    histories: dict[str, list[tuple[int, float]]] = {}
    rows: list[dict[str, object]] = []
    for (season, week), week_rows in snaps.groupby(["season", "week"], sort=True):
        season = int(season)
        week = int(week)
        for row in week_rows.itertuples(index=False):
            identity = str(row.player_id) if pd.notna(row.player_id) else _clean_name(row.player)
            history = histories.get(identity, [])
            same_season = [value for hist_season, value in history if hist_season == season]
            if len(same_season) >= minimum_prior_games:
                value = _weighted_average(same_season, recency_decay)
                prior_games = len(same_season)
                value_source = "same_season"
            else:
                previous = [
                    hist_value
                    for hist_season, hist_value in history
                    if hist_season == season - 1
                ]
                previous = previous[-prior_season_games:] if prior_season_games else []
                if previous:
                    value = _weighted_average(previous, recency_decay) * prior_season_weight
                    prior_games = len(previous)
                    value_source = "prior_season"
                else:
                    value = 0.0
                    prior_games = 0
                    value_source = "none"

            postgame_same = same_season + [float(row.snap_share)]
            postgame_value = _weighted_average(postgame_same, recency_decay)
            rows.append(
                {
                    "game_id": row.game_id,
                    "season": season,
                    "week": week,
                    "team": row.team,
                    "player_id": row.player_id,
                    "player_name": row.player,
                    "position_group": row.position_group,
                    "player_value": float(np.clip(value, 0.0, 1.0)),
                    "postgame_player_value": float(np.clip(postgame_value, 0.0, 1.0)),
                    "prior_games": prior_games,
                    "value_source": value_source,
                }
            )
        for row in week_rows.itertuples(index=False):
            identity = str(row.player_id) if pd.notna(row.player_id) else _clean_name(row.player)
            histories.setdefault(identity, []).append((season, float(row.snap_share)))
    return pd.DataFrame(rows)


def _prior_value_lookup(player_values: pd.DataFrame) -> dict[str, list[tuple[int, float]]]:
    """Index completed-game values for leakage-safe carry-forward lookups."""
    if "postgame_player_value" not in player_values.columns:
        return {}
    lookup: dict[str, list[tuple[int, float]]] = {}
    for row in player_values.dropna(subset=["player_id"]).itertuples(index=False):
        key = str(row.player_id)
        time_key = int(row.season) * 100 + int(row.week)
        lookup.setdefault(key, []).append((time_key, float(row.postgame_player_value)))
    for history in lookup.values():
        history.sort()
    return lookup


def attach_player_values(availability: pd.DataFrame, player_values: pd.DataFrame) -> pd.DataFrame:
    """Attach pregame values, carrying latest completed usage into missed games."""
    required_availability = {
        "season",
        "week",
        "team",
        "player_name",
        "position_group",
        "status",
    }
    missing = required_availability.difference(availability.columns)
    if missing:
        raise ValueError(f"availability missing required columns: {sorted(missing)}")
    required_values = {"season", "week", "team", "player_name", "player_value"}
    missing_values = required_values.difference(player_values.columns)
    if missing_values:
        raise ValueError(f"player values missing required columns: {sorted(missing_values)}")

    out = availability.copy()
    values = player_values.copy()
    if "player_id" not in out.columns:
        out["player_id"] = pd.NA
    if "player_id" not in values.columns:
        values["player_id"] = pd.NA
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

    by_name = values[
        ["season", "week", "team", "_clean_name", "player_value"]
    ].drop_duplicates(["season", "week", "team", "_clean_name"])
    out = out.merge(
        by_name.rename(columns={"player_value": "player_value_name"}),
        on=["season", "week", "team", "_clean_name"],
        how="left",
    )
    out["player_value"] = out["player_value_id"].combine_first(out["player_value_name"])

    prior_lookup = _prior_value_lookup(values)
    missing_mask = out["player_value"].isna() & out["player_id"].notna()
    for idx in out.index[missing_mask]:
        history = prior_lookup.get(str(out.at[idx, "player_id"]), [])
        if not history:
            continue
        target = int(out.at[idx, "season"]) * 100 + int(out.at[idx, "week"])
        position = bisect.bisect_left(history, (target, -1.0)) - 1
        if position >= 0:
            out.at[idx, "player_value"] = history[position][1]

    out["player_value"] = (
        pd.to_numeric(out["player_value"], errors="coerce").fillna(0.0).clip(0.0, 1.0)
    )
    return out.drop(columns=["_clean_name", "player_value_id", "player_value_name"])
