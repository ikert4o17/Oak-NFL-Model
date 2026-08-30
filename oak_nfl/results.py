"""Grade frozen Oak predictions against final scores and closing market lines."""
from __future__ import annotations

import numpy as np
import pandas as pd


def _wl(value: float) -> str:
    if pd.isna(value):
        return "PENDING"
    if np.isclose(value, 0.0):
        return "P"
    return "W" if value > 0 else "L"


def grade_predictions(predictions: pd.DataFrame, finals: pd.DataFrame) -> pd.DataFrame:
    """Grade Oak's frozen predictions against final scores and closing lines.

    ``finals`` must provide game_id, home_score, away_score and may provide
    closing_spread_line / closing_total_line. Closing spread follows Oak's
    home-team convention: home favorites are negative.
    """
    required = {"game_id", "home_team", "away_team", "predicted_home_margin", "predicted_total"}
    missing = required.difference(predictions.columns)
    if missing:
        raise ValueError(f"predictions missing required columns: {sorted(missing)}")
    final_required = {"game_id", "home_score", "away_score"}
    missing = final_required.difference(finals.columns)
    if missing:
        raise ValueError(f"finals missing required columns: {sorted(missing)}")

    cols = ["game_id", "home_score", "away_score"]
    for col in ("closing_spread_line", "closing_total_line"):
        if col in finals:
            cols.append(col)
    out = predictions.merge(finals[cols], on="game_id", how="left")
    if "closing_spread_line" not in out:
        out["closing_spread_line"] = np.nan
    if "closing_total_line" not in out:
        out["closing_total_line"] = np.nan

    out["final_home_margin"] = pd.to_numeric(out.home_score, errors="coerce") - pd.to_numeric(out.away_score, errors="coerce")
    out["final_total"] = pd.to_numeric(out.home_score, errors="coerce") + pd.to_numeric(out.away_score, errors="coerce")

    model_side = np.sign(pd.to_numeric(out.predicted_home_margin, errors="coerce"))
    out["su_result"] = [_wl(side * margin) if side else "P" for side, margin in zip(model_side, out.final_home_margin)]

    market_margin = -pd.to_numeric(out.closing_spread_line, errors="coerce")
    spread_side = np.sign(pd.to_numeric(out.predicted_home_margin, errors="coerce") - market_margin)
    cover_margin = out.final_home_margin - market_margin
    out["ats_result"] = [_wl(side * margin) if side else "P" for side, margin in zip(spread_side, cover_margin)]

    total_side = np.sign(pd.to_numeric(out.predicted_total, errors="coerce") - pd.to_numeric(out.closing_total_line, errors="coerce"))
    total_margin = out.final_total - pd.to_numeric(out.closing_total_line, errors="coerce")
    out["ou_result"] = [_wl(side * margin) if side else "P" for side, margin in zip(total_side, total_margin)]
    return out


def summarize_results(graded: pd.DataFrame) -> pd.DataFrame:
    """Return one row per season/week with SU, ATS and O/U W-L-P records."""
    rows = []
    for (season, week), group in graded.groupby(["season", "week"], dropna=False):
        row = {"season": int(season), "week": int(week), "games": len(group)}
        for key, col in (("su", "su_result"), ("ats", "ats_result"), ("ou", "ou_result")):
            counts = group[col].value_counts()
            row[f"{key}_w"] = int(counts.get("W", 0)); row[f"{key}_l"] = int(counts.get("L", 0)); row[f"{key}_p"] = int(counts.get("P", 0))
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["season", "week"]).reset_index(drop=True)
