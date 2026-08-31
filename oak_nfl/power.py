"""Neutral-field Oak power ratings derived directly from the promoted V5 spread core."""

from __future__ import annotations

import pandas as pd

from oak_nfl.ratings.v5 import V5_EPA_COEF, V5_EXPLOSIVE_COEF, V5_SUCCESS_COEF


def build_power_ratings(snapshot: pd.DataFrame) -> pd.DataFrame:
    """Convert one V5 pregame team snapshot into points above/below NFL average.

    V5 predicts a matchup as its intercept plus the difference between each
    team's EPA, success-rate, and explosive-rate offense-minus-defense strength.
    The same linear contribution is therefore Oak's neutral-field team rating.
    Centering the ratings at zero changes no matchup differences.
    """
    required = {
        "team",
        "pregame_off_epa_per_play",
        "pregame_def_epa_per_play_allowed",
        "pregame_off_success_rate",
        "pregame_def_success_rate_allowed",
        "pregame_off_explosive_rate",
        "pregame_def_explosive_rate_allowed",
    }
    missing = required.difference(snapshot.columns)
    if missing:
        raise ValueError(f"power-rating snapshot missing required columns: {sorted(missing)}")

    out = snapshot.copy().drop_duplicates("team")
    out["epa_points"] = V5_EPA_COEF * (
        out["pregame_off_epa_per_play"] - out["pregame_def_epa_per_play_allowed"]
    )
    out["success_points"] = V5_SUCCESS_COEF * (
        out["pregame_off_success_rate"] - out["pregame_def_success_rate_allowed"]
    )
    out["explosive_points"] = V5_EXPLOSIVE_COEF * (
        out["pregame_off_explosive_rate"] - out["pregame_def_explosive_rate_allowed"]
    )
    out["raw_rating"] = out[["epa_points", "success_points", "explosive_points"]].sum(axis=1)
    out["rating"] = out["raw_rating"] - out["raw_rating"].mean()
    out = out.sort_values(["rating", "team"], ascending=[False, True]).reset_index(drop=True)
    out["rank"] = out.index + 1
    return out[["rank", "team", "rating", "epa_points", "success_points", "explosive_points"]]
