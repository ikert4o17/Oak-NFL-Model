"""Production-facing weekly output for Oak spread and total forecasts.

This module intentionally does not retrain or retune models. It combines the
already-produced spread and total projections with market lines into one stable,
auditable weekly game card.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

SPREAD_MODEL = "Oak spread: promoted V5 core + validated adjustments"
TOTAL_MODEL = "Oak totals: locked V12"

_REQUIRED = {
    "game_id",
    "home_team",
    "away_team",
    "predicted_home_margin",
    "predicted_total",
}


def _missing_label(value: float, positive: str, negative: str, push: str) -> str:
    if pd.isna(value):
        return "NO LINE"
    if np.isclose(value, 0.0):
        return push
    return positive if value > 0 else negative


def build_weekly_game_card(predictions: pd.DataFrame) -> pd.DataFrame:
    """Combine Oak spread and total projections into one weekly game card.

    Market convention: ``spread_line`` is the home-team spread, so a home
    favorite is negative (for example -3.5). The market-implied home margin is
    therefore ``-spread_line``.

    Market lines are optional so preseason/offline model runs can still emit a
    complete projection card. When a line is absent, the corresponding edge and
    side are left unscored rather than guessed.
    """
    missing = _REQUIRED.difference(predictions.columns)
    if missing:
        raise ValueError(f"weekly predictions missing required columns: {sorted(missing)}")

    out = predictions.copy()
    if "spread_line" not in out:
        out["spread_line"] = np.nan
    if "total_line" not in out:
        out["total_line"] = np.nan

    out["market_home_margin"] = -pd.to_numeric(out["spread_line"], errors="coerce")
    out["spread_edge"] = out["predicted_home_margin"] - out["market_home_margin"]
    out["total_edge"] = out["predicted_total"] - pd.to_numeric(out["total_line"], errors="coerce")

    out["spread_side"] = [
        _missing_label(edge, home, away, "PUSH")
        for edge, home, away in zip(out["spread_edge"], out["home_team"], out["away_team"])
    ]
    out["total_side"] = [
        _missing_label(edge, "OVER", "UNDER", "PUSH") for edge in out["total_edge"]
    ]

    out["projected_home_score"] = (
        out["predicted_total"] + out["predicted_home_margin"]
    ) / 2.0
    out["projected_away_score"] = (
        out["predicted_total"] - out["predicted_home_margin"]
    ) / 2.0

    out["spread_model"] = SPREAD_MODEL
    out["total_model"] = TOTAL_MODEL

    preferred = [
        "game_id",
        "season",
        "week",
        "away_team",
        "home_team",
        "predicted_home_margin",
        "spread_line",
        "spread_edge",
        "spread_side",
        "predicted_total",
        "total_line",
        "total_edge",
        "total_side",
        "projected_away_score",
        "projected_home_score",
        "spread_model",
        "total_model",
    ]
    columns = [c for c in preferred if c in out.columns]
    extras = [c for c in out.columns if c not in columns]
    return out[columns + extras]
