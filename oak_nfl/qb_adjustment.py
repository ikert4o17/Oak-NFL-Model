"""Translate quarterback changes into bounded point-spread adjustments."""

from __future__ import annotations

import numpy as np
import pandas as pd

QB_EPA_TO_POINTS = 19.6244
QB_CHANGE_DAMPING = 0.50
QB_CHANGE_MAX_POINTS = 2.0


def qb_change_points(
    expected_qb_epa: float,
    baseline_qb_epa: float,
    *,
    epa_to_points: float = QB_EPA_TO_POINTS,
    damping: float = QB_CHANGE_DAMPING,
    max_adjustment: float = QB_CHANGE_MAX_POINTS,
) -> float:
    """Return the validated point adjustment for expected QB versus baseline QB.

    Positive values improve the team's expected margin; negative values reduce it.
    V9 historical validation selected 50% damping with a +/-2 point cap on
    2023-2025 quarterback-change games.
    """
    if pd.isna(expected_qb_epa) or pd.isna(baseline_qb_epa):
        return 0.0
    raw = (float(expected_qb_epa) - float(baseline_qb_epa)) * epa_to_points * damping
    return float(np.clip(raw, -max_adjustment, max_adjustment))


def add_qb_change_adjustments(
    games: pd.DataFrame,
    *,
    home_expected_col: str = "home_expected_qb_epa",
    home_baseline_col: str = "home_baseline_qb_epa",
    away_expected_col: str = "away_expected_qb_epa",
    away_baseline_col: str = "away_baseline_qb_epa",
    damping: float = QB_CHANGE_DAMPING,
    max_adjustment: float = QB_CHANGE_MAX_POINTS,
) -> pd.DataFrame:
    """Apply home/away QB changes to an existing predicted home margin."""
    required = {
        "predicted_home_margin",
        home_expected_col,
        home_baseline_col,
        away_expected_col,
        away_baseline_col,
    }
    missing = required.difference(games.columns)
    if missing:
        raise ValueError(f"QB adjustment missing required columns: {sorted(missing)}")
    out = games.copy()
    out["home_qb_change_points"] = [
        qb_change_points(e, b, damping=damping, max_adjustment=max_adjustment)
        for e, b in zip(out[home_expected_col], out[home_baseline_col])
    ]
    out["away_qb_change_points"] = [
        qb_change_points(e, b, damping=damping, max_adjustment=max_adjustment)
        for e, b in zip(out[away_expected_col], out[away_baseline_col])
    ]
    out["qb_change_net_points"] = out["home_qb_change_points"] - out["away_qb_change_points"]
    out["predicted_home_margin_pre_qb_change"] = out["predicted_home_margin"]
    out["predicted_home_margin"] = out["predicted_home_margin"] + out["qb_change_net_points"]
    return out
