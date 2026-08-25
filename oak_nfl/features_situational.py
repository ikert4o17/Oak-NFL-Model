"""Situational efficiency features derived from nflverse play-by-play."""

from __future__ import annotations

import numpy as np
import pandas as pd

from oak_nfl.features import clean_scrimmage_plays


def build_situational_team_game_features(pbp: pd.DataFrame) -> pd.DataFrame:
    """Aggregate early-down, third-down, red-zone, and sack indicators per team-game."""
    plays = clean_scrimmage_plays(pbp).copy()
    required = {"down", "yardline_100", "sack", "success", "epa", "pass"}
    missing = required.difference(plays.columns)
    if missing:
        raise ValueError(f"situational features missing required columns: {sorted(missing)}")

    plays["early_down"] = plays["down"].isin([1, 2])
    plays["third_down"] = plays["down"].eq(3)
    plays["red_zone"] = plays["yardline_100"].le(20)
    plays["dropback"] = plays["pass"].eq(1)
    plays["sack_flag"] = plays["sack"].fillna(0).eq(1).astype(float)

    keys = ["game_id", "season", "week", "posteam", "defteam"]
    base = plays[keys].drop_duplicates()

    def aggregate(mask: pd.Series, prefix: str) -> pd.DataFrame:
        subset = plays.loc[mask]
        return (
            subset.groupby(keys, dropna=False)
            .agg(**{
                f"{prefix}_epa_per_play": ("epa", "mean"),
                f"{prefix}_success_rate": ("success", "mean"),
            })
            .reset_index()
        )

    out = base
    for mask, prefix in [
        (plays["early_down"], "early_down"),
        (plays["third_down"], "third_down"),
        (plays["red_zone"], "red_zone"),
    ]:
        out = out.merge(aggregate(mask, prefix), on=keys, how="left")

    sacks = (
        plays.loc[plays["dropback"]]
        .groupby(keys, dropna=False)
        .agg(dropbacks=("epa", "size"), sacks=("sack_flag", "sum"))
        .reset_index()
    )
    sacks["sack_rate"] = sacks["sacks"] / sacks["dropbacks"].replace(0, np.nan)
    out = out.merge(sacks[keys + ["sack_rate"]], on=keys, how="left")
    return out.sort_values(["season", "week", "game_id", "posteam"]).reset_index(drop=True)
