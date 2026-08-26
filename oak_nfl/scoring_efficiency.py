"""Leakage-safe scoring conversion efficiency features for Oak V12."""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd


def build_scoring_efficiency(games: pd.DataFrame, *, windows=(4, 8)) -> pd.DataFrame:
    """Build pregame points-per-100-yards offense and defense features.

    Defensive DPA/100 = 100 * opponent offensive points / yards allowed.
    Offensive PPA/100 = 100 * offensive points / offensive yards.
    Current-game results are appended only after its pregame features are emitted.
    """
    required = {
        "game_id",
        "season",
        "week",
        "home_team",
        "away_team",
        "home_score",
        "away_score",
        "home_yards",
        "away_yards",
    }
    missing = required.difference(games.columns)
    if missing:
        raise ValueError(
            f"games missing scoring efficiency columns: {sorted(missing)}"
        )

    frame = games.sort_values(["season", "week", "game_id"]).copy()
    hist = defaultdict(list)
    rows = []
    for row in frame.itertuples(index=False):
        out = {"game_id": row.game_id}
        for side, team in (("home", row.home_team), ("away", row.away_team)):
            history = hist[team]
            for window in windows:
                sample = history[-window:]
                off_pts = sum(x[0] for x in sample)
                off_yds = sum(x[1] for x in sample)
                def_pts = sum(x[2] for x in sample)
                def_yds = sum(x[3] for x in sample)
                out[f"{side}_ppa100_{window}"] = (
                    100.0 * off_pts / off_yds if off_yds > 0 else np.nan
                )
                out[f"{side}_dpa100_{window}"] = (
                    100.0 * def_pts / def_yds if def_yds > 0 else np.nan
                )

            sample = history
            off_pts = sum(x[0] for x in sample)
            off_yds = sum(x[1] for x in sample)
            def_pts = sum(x[2] for x in sample)
            def_yds = sum(x[3] for x in sample)
            out[f"{side}_ppa100_season"] = (
                100.0 * off_pts / off_yds if off_yds > 0 else np.nan
            )
            out[f"{side}_dpa100_season"] = (
                100.0 * def_pts / def_yds if def_yds > 0 else np.nan
            )

        rows.append(out)
        complete = (
            pd.notna(row.home_score)
            and pd.notna(row.away_score)
            and pd.notna(row.home_yards)
            and pd.notna(row.away_yards)
        )
        if complete:
            hist[row.home_team].append(
                (
                    float(row.home_score),
                    float(row.home_yards),
                    float(row.away_score),
                    float(row.away_yards),
                )
            )
            hist[row.away_team].append(
                (
                    float(row.away_score),
                    float(row.away_yards),
                    float(row.home_score),
                    float(row.home_yards),
                )
            )

    return pd.DataFrame(rows)
