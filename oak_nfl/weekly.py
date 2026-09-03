"""One-call weekly Oak production engine for spreads and totals."""

from __future__ import annotations

import numpy as np
import pandas as pd

from oak_nfl.features import build_team_game_features
from oak_nfl.personnel import apply_personnel_adjustments
from oak_nfl.production import build_weekly_game_card
from oak_nfl.qb_adjustment import add_qb_change_adjustments
from oak_nfl.ratings.v5 import build_v5_game_predictions, build_v5_pregame_ratings
from oak_nfl.totals_v12 import predict_v12_totals


def _spread_core(pbp: pd.DataFrame, slate: pd.DataFrame) -> pd.DataFrame:
    team_games = build_team_game_features(pbp)
    metric_cols = [c for c in team_games.columns if c not in {"game_id", "season", "week", "posteam", "defteam"}]
    dummy = []
    for row in slate.itertuples(index=False):
        for offense, defense in ((row.home_team, row.away_team), (row.away_team, row.home_team)):
            item = {
                "game_id": row.game_id,
                "season": row.season,
                "week": row.week,
                "posteam": offense,
                "defteam": defense,
            }
            item.update({c: np.nan for c in metric_cols})
            dummy.append(item)
    ratings = build_v5_pregame_ratings(pd.concat([team_games, pd.DataFrame(dummy)], ignore_index=True))
    games = slate[["game_id", "season", "week", "home_team", "away_team"]].copy()
    return build_v5_game_predictions(games, ratings)


def run_weekly_predictions(
    pbp: pd.DataFrame,
    slate: pd.DataFrame,
    *,
    qb_inputs: pd.DataFrame | None = None,
    personnel_adjustments: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Generate Oak's unified spread + total card for an upcoming slate.

    ``slate`` should come from nflverse schedules and include game_id, season,
    week, home_team, away_team, total_line, and (when available) spread_line.
    Optional quarterback and non-QB personnel inputs apply the already-defined
    bounded point adjustments after the promoted V5 spread core.
    """
    required = {"game_id", "season", "week", "home_team", "away_team", "total_line"}
    missing = required.difference(slate.columns)
    if missing:
        raise ValueError(f"slate missing required columns: {sorted(missing)}")

    spread = _spread_core(pbp, slate)
    if qb_inputs is not None:
        spread = spread.merge(qb_inputs, on="game_id", how="left")
        spread = add_qb_change_adjustments(spread)
    if personnel_adjustments is not None:
        spread = apply_personnel_adjustments(spread, personnel_adjustments)

    totals = predict_v12_totals(pbp, slate)
    out = slate.copy()
    spread_context = [
        c
        for c in spread.columns
        if c != "game_id"
        and (
            c == "predicted_home_margin"
            or c.startswith("home_qb_")
            or c.startswith("away_qb_")
            or c.startswith("home_expected_qb_")
            or c.startswith("away_expected_qb_")
            or c.startswith("home_baseline_qb_")
            or c.startswith("away_baseline_qb_")
            or c.startswith("home_depth_chart_")
            or c.startswith("away_depth_chart_")
        )
    ]
    out = out.merge(
        spread[["game_id", *spread_context]], on="game_id", how="left", validate="one_to_one"
    )
    out = out.merge(
        totals[["game_id", "predicted_total"]], on="game_id", how="left", validate="one_to_one"
    )
    return build_weekly_game_card(out)
