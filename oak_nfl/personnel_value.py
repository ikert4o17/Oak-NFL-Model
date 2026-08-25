"""Point-in-time non-QB player importance estimated from prior snap participation."""

from __future__ import annotations

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


def build_pregame_player_values(
    snap_counts: pd.DataFrame,
    *,
    recency_decay: float = 0.90,
    minimum_prior_games: int = 1,
) -> pd.DataFrame:
    """Estimate player importance using only snap shares from completed prior weeks.

    The output has one row for each player appearing in a team's snap-count data
    in a given week. Values are normalized to [0, 1]. Quarterbacks are retained
    for auditing but V10's personnel adjustment layer excludes them.
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

    snaps = snap_counts.sort_values(["season", "week", "game_id", "team", "player"]).copy()
    snaps["position_group"] = snaps["position"].map(normalize_position)
    offense = pd.to_numeric(snaps["offense_pct"], errors="coerce").fillna(0.0)
    defense = pd.to_numeric(snaps["defense_pct"], errors="coerce").fillna(0.0)
    snaps["snap_share"] = np.where(
        snaps["position_group"].isin(OFFENSE_GROUPS),
        offense,
        np.where(snaps["position_group"].isin(DEFENSE_GROUPS), defense, np.maximum(offense, defense)),
    )
    # nflverse/PFR percentages are normally 0-1, but tolerate 0-100 feeds.
    snaps.loc[snaps["snap_share"].gt(1.0), "snap_share"] /= 100.0
    snaps["snap_share"] = snaps["snap_share"].clip(0.0, 1.0)

    histories: dict[tuple[str, str], list[float]] = {}
    rows: list[dict[str, object]] = []
    for (season, week), week_rows in snaps.groupby(["season", "week"], sort=True):
        for row in week_rows.itertuples(index=False):
            key = (str(row.team), str(row.player))
            history = histories.get(key, [])
            value = _weighted_average(history, recency_decay) if len(history) >= minimum_prior_games else 0.0
            rows.append(
                {
                    "game_id": row.game_id,
                    "season": int(season),
                    "week": int(week),
                    "team": row.team,
                    "player_name": row.player,
                    "position_group": row.position_group,
                    "player_value": float(np.clip(value, 0.0, 1.0)),
                    "prior_games": len(history),
                }
            )
        # Update only after every row in the week has received its pregame value.
        for row in week_rows.itertuples(index=False):
            histories.setdefault((str(row.team), str(row.player)), []).append(float(row.snap_share))

    return pd.DataFrame(rows)


def attach_player_values(
    availability: pd.DataFrame,
    player_values: pd.DataFrame,
) -> pd.DataFrame:
    """Join canonical injury rows to pregame player values by season/week/team/name."""
    required_availability = {"season", "week", "team", "player_name", "position_group", "status"}
    missing = required_availability.difference(availability.columns)
    if missing:
        raise ValueError(f"availability missing required columns: {sorted(missing)}")
    required_values = {"season", "week", "team", "player_name", "player_value"}
    missing_values = required_values.difference(player_values.columns)
    if missing_values:
        raise ValueError(f"player values missing required columns: {sorted(missing_values)}")

    values = player_values[["season", "week", "team", "player_name", "player_value"]]
    out = availability.merge(
        values,
        on=["season", "week", "team", "player_name"],
        how="left",
        validate="many_to_one",
    )
    out["player_value"] = out["player_value"].fillna(0.0).clip(0.0, 1.0)
    return out
