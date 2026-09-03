"""Conservative normalization for live NFL injury-report context.

This module is intentionally informational first. It does not invent player
values or turn ambiguous injury-report language into model points. Numerical
personnel adjustments remain a separate, explicitly supplied input.
"""

from __future__ import annotations

import pandas as pd

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

STATUS_CONFIDENCE = {
    "out": "high",
    "doubtful": "medium",
    "questionable": "low",
    "limited": "low",
    "full": "high",
    "active": "high",
    "unknown": "unknown",
}


def normalize_status(value: object) -> str:
    """Map common official-report labels to Oak's small status vocabulary."""
    if pd.isna(value):
        return "unknown"
    text = str(value).strip().lower()
    return STATUS_ALIASES.get(text, "unknown")


def normalize_injury_report(report: pd.DataFrame) -> pd.DataFrame:
    """Normalize a provider injury report without assigning model points.

    Required fields are deliberately small so a provider adapter can rename
    source-specific columns before calling this function. Unknown or ambiguous
    statuses remain ``unknown`` and therefore imply no automatic adjustment.
    """
    required = {"team", "player_name", "position", "status"}
    missing = required.difference(report.columns)
    if missing:
        raise ValueError(f"injury report missing required columns: {sorted(missing)}")

    out = report.copy()
    out["team"] = out["team"].astype(str).str.strip().str.upper()
    out["player_name"] = out["player_name"].astype(str).str.strip()
    out["position"] = out["position"].astype(str).str.strip().str.upper()
    out["injury_status"] = out["status"].map(normalize_status)
    out["injury_confidence"] = out["injury_status"].map(STATUS_CONFIDENCE)
    out["automatic_adjustment_allowed"] = False
    return out


def team_injury_context(report: pd.DataFrame) -> pd.DataFrame:
    """Build auditable team-level live context from a normalized report."""
    frame = normalize_injury_report(report)
    frame["is_out"] = frame["injury_status"].eq("out")
    frame["is_doubtful"] = frame["injury_status"].eq("doubtful")
    frame["is_questionable"] = frame["injury_status"].eq("questionable")
    frame["is_unknown"] = frame["injury_status"].eq("unknown")

    grouped = (
        frame.groupby("team", as_index=False)
        .agg(
            injury_players_reported=("player_name", "size"),
            injury_out_count=("is_out", "sum"),
            injury_doubtful_count=("is_doubtful", "sum"),
            injury_questionable_count=("is_questionable", "sum"),
            injury_unknown_count=("is_unknown", "sum"),
        )
    )
    for col in (
        "injury_players_reported",
        "injury_out_count",
        "injury_doubtful_count",
        "injury_questionable_count",
        "injury_unknown_count",
    ):
        grouped[col] = grouped[col].astype(int)
    grouped["injury_auto_points"] = 0.0
    return grouped
