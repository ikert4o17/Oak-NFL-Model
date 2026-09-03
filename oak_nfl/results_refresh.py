"""Helpers for refreshing Oak's public results ledger."""

from pathlib import Path

import pandas as pd

PREDICTION_GLOB = "oak_*_week_*.csv"


def load_frozen_predictions(directory: Path) -> pd.DataFrame:
    """Load archived weekly prediction cards and keep the first game snapshot."""
    files = sorted(directory.glob(PREDICTION_GLOB))
    if not files:
        return pd.DataFrame()
    frames = [pd.read_csv(path) for path in files]
    out = pd.concat(frames, ignore_index=True)
    return out.drop_duplicates("game_id", keep="first")


def finals_from_schedules(schedules: pd.DataFrame) -> pd.DataFrame:
    """Extract completed scores and normalized nflverse closing lines."""
    required = {"game_id", "home_score", "away_score"}
    missing = required.difference(schedules.columns)
    if missing:
        raise ValueError(f"schedule data missing required result columns: {sorted(missing)}")

    cols = ["game_id", "home_score", "away_score"]
    for col in ("spread_line", "total_line"):
        if col in schedules.columns:
            cols.append(col)
    finals = schedules[cols].copy()
    finals = finals.loc[finals["home_score"].notna() & finals["away_score"].notna()]
    finals = finals.rename(
        columns={
            "spread_line": "closing_spread_line",
            "total_line": "closing_total_line",
        }
    )
    return finals.drop_duplicates("game_id", keep="last")
