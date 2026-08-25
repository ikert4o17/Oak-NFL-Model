"""Position-weighted non-quarterback personnel availability adjustments.

V10 intentionally keeps quarterback changes separate in ``qb_adjustment``.
This module provides a conservative, auditable framework for translating
non-QB availability into team point adjustments once weekly player statuses
and player values are available.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# Initial priors are deliberately conservative. Historical validation should
# tune/replace them before V10 is promoted to production.
POSITION_POINT_CAPS = {
    "OT": 0.75,
    "IOL": 0.50,
    "WR": 0.75,
    "TE": 0.40,
    "RB": 0.30,
    "EDGE": 0.75,
    "IDL": 0.45,
    "LB": 0.35,
    "CB": 0.70,
    "S": 0.40,
}

STATUS_AVAILABILITY = {
    "out": 0.0,
    "doubtful": 0.15,
    "questionable": 0.65,
    "limited": 0.85,
    "active": 1.0,
    "full": 1.0,
}


@dataclass(frozen=True)
class PersonnelAdjustment:
    team: str
    points: float
    missing_value: float
    players_affected: int


def player_absence_points(
    position: str,
    player_value: float,
    status: str,
    *,
    max_player_points: float | None = None,
) -> float:
    """Estimate negative team points from one non-QB player's availability.

    ``player_value`` is normalized from 0 to 1, where 1 represents an elite /
    highly important player at that position. Status controls the expected
    fraction of that value unavailable for the game.
    """
    pos = str(position).upper()
    if pos == "QB":
        raise ValueError("QB changes must use oak_nfl.qb_adjustment")
    cap = POSITION_POINT_CAPS.get(pos, 0.25) if max_player_points is None else float(max_player_points)
    value = float(np.clip(player_value, 0.0, 1.0))
    availability = STATUS_AVAILABILITY.get(str(status).lower(), 1.0)
    return -cap * value * (1.0 - availability)


def team_personnel_adjustment(
    availability: pd.DataFrame,
    *,
    team_col: str = "team",
    position_col: str = "position_group",
    value_col: str = "player_value",
    status_col: str = "status",
    team_cap: float = 3.0,
) -> pd.DataFrame:
    """Aggregate non-QB player statuses into bounded team point adjustments."""
    required = {team_col, position_col, value_col, status_col}
    missing = required.difference(availability.columns)
    if missing:
        raise ValueError(f"personnel availability missing columns: {sorted(missing)}")

    frame = availability.copy()
    frame = frame[frame[position_col].astype(str).str.upper().ne("QB")].copy()
    frame["absence_points"] = [
        player_absence_points(pos, value, status)
        for pos, value, status in zip(frame[position_col], frame[value_col], frame[status_col])
    ]
    frame["missing_value"] = -frame["absence_points"]
    grouped = (
        frame.groupby(team_col, as_index=False)
        .agg(
            raw_personnel_points=("absence_points", "sum"),
            missing_value=("missing_value", "sum"),
            players_affected=("absence_points", lambda s: int((s < 0).sum())),
        )
    )
    grouped["personnel_points"] = grouped["raw_personnel_points"].clip(-team_cap, team_cap)
    return grouped[[team_col, "personnel_points", "missing_value", "players_affected"]]


def apply_personnel_adjustments(
    games: pd.DataFrame,
    team_adjustments: pd.DataFrame,
    *,
    team_cap: float = 3.0,
) -> pd.DataFrame:
    """Apply home/away non-QB personnel values to predicted home margin."""
    required_games = {"home_team", "away_team", "predicted_home_margin"}
    missing = required_games.difference(games.columns)
    if missing:
        raise ValueError(f"games missing required columns: {sorted(missing)}")
    required_adj = {"team", "personnel_points"}
    missing_adj = required_adj.difference(team_adjustments.columns)
    if missing_adj:
        raise ValueError(f"team adjustments missing columns: {sorted(missing_adj)}")

    home = team_adjustments[["team", "personnel_points"]].rename(
        columns={"team": "home_team", "personnel_points": "home_personnel_points"}
    )
    away = team_adjustments[["team", "personnel_points"]].rename(
        columns={"team": "away_team", "personnel_points": "away_personnel_points"}
    )
    out = games.merge(home, on="home_team", how="left").merge(away, on="away_team", how="left")
    out[["home_personnel_points", "away_personnel_points"]] = out[
        ["home_personnel_points", "away_personnel_points"]
    ].fillna(0.0).clip(-team_cap, team_cap)
    out["personnel_net_points"] = out["home_personnel_points"] - out["away_personnel_points"]
    out["predicted_home_margin_pre_personnel"] = out["predicted_home_margin"]
    out["predicted_home_margin"] = out["predicted_home_margin"] + out["personnel_net_points"]
    return out
