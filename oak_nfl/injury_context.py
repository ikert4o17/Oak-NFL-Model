"""Informational live NFL injury context for Oak weekly previews.

This layer intentionally does not convert raw injury-report labels into model
points. Provider adapters normalize data first; this module summarizes what is
known for display/audit purposes. Numerical personnel adjustments stay in the
separate validated personnel path.
"""

from __future__ import annotations

import pandas as pd

from oak_nfl.data.injuries import normalize_position, normalize_status

STATUS_CONFIDENCE = {
    "out": "high",
    "doubtful": "medium",
    "questionable": "low",
    "limited": "low",
    "full": "high",
    "active": "high",
    "unknown": "unknown",
}


def normalize_injury_report(report: pd.DataFrame) -> pd.DataFrame:
    """Normalize report columns needed by the live informational layer.

    The canonical provider schema uses ``position_group``. ``position`` is also
    accepted for small/manual inputs and converted into the canonical grouping.
    Unknown or ambiguous statuses remain explicit and never authorize an
    automatic point adjustment.
    """
    data = report.copy()
    if "position_group" not in data.columns and "position" in data.columns:
        data["position_group"] = data["position"]

    required = {"team", "player_name", "position_group", "status"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"injury report missing required columns: {sorted(missing)}")

    data["team"] = data["team"].astype(str).str.strip().str.upper()
    data["player_name"] = data["player_name"].astype(str).str.strip()
    data["position_group"] = data["position_group"].map(normalize_position)
    data["injury_status"] = data["status"].map(normalize_status)
    data["injury_confidence"] = data["injury_status"].map(STATUS_CONFIDENCE)
    data["automatic_adjustment_allowed"] = False
    if "source" not in data.columns:
        data["source"] = "unknown"
    if "report_date" not in data.columns:
        data["report_date"] = pd.NaT
    data["report_date"] = pd.to_datetime(data["report_date"], errors="coerce", utc=True)
    return data


def team_injury_context(report: pd.DataFrame) -> pd.DataFrame:
    """Build auditable team-level live context without changing predictions."""
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
            injury_report_date=("report_date", "max"),
            injury_source=("source", lambda s: ",".join(sorted(set(s.dropna().astype(str))))),
        )
    )
    count_cols = (
        "injury_players_reported",
        "injury_out_count",
        "injury_doubtful_count",
        "injury_questionable_count",
        "injury_unknown_count",
    )
    grouped[list(count_cols)] = grouped[list(count_cols)].astype(int)
    grouped["injury_auto_points"] = 0.0
    return grouped


def build_game_injury_context(report: pd.DataFrame, slate: pd.DataFrame) -> pd.DataFrame:
    """Attach home/away injury summaries to each slate game for live display."""
    required = {"game_id", "home_team", "away_team"}
    missing = required.difference(slate.columns)
    if missing:
        raise ValueError(f"slate missing required columns: {sorted(missing)}")

    team_context = team_injury_context(report)
    context_cols = [c for c in team_context.columns if c != "team"]

    home = team_context.rename(
        columns={"team": "home_team", **{c: f"home_{c}" for c in context_cols}}
    )
    away = team_context.rename(
        columns={"team": "away_team", **{c: f"away_{c}" for c in context_cols}}
    )
    out = slate[["game_id", "home_team", "away_team"]].copy()
    out = out.merge(home, on="home_team", how="left").merge(away, on="away_team", how="left")

    count_suffixes = (
        "injury_players_reported",
        "injury_out_count",
        "injury_doubtful_count",
        "injury_questionable_count",
        "injury_unknown_count",
    )
    for side in ("home", "away"):
        for suffix in count_suffixes:
            col = f"{side}_{suffix}"
            out[col] = out[col].fillna(0).astype(int)
        out[f"{side}_injury_auto_points"] = out[f"{side}_injury_auto_points"].fillna(0.0)
        out[f"{side}_injury_source"] = out[f"{side}_injury_source"].fillna("missing")
    return out
