"""Point-in-time feature construction from nflverse play-by-play data."""

from __future__ import annotations

import numpy as np
import pandas as pd

_REQUIRED_COLUMNS = {
    "game_id",
    "season",
    "week",
    "posteam",
    "defteam",
    "play_type",
    "epa",
    "success",
    "yards_gained",
    "pass",
    "rush",
    "qb_kneel",
    "qb_spike",
}


def clean_scrimmage_plays(pbp: pd.DataFrame) -> pd.DataFrame:
    """Return offensive scrimmage plays suitable for efficiency modeling.

    The function removes kneels/spikes, requires an offensive and defensive team,
    and keeps only pass/rush plays with non-null EPA.
    """
    missing = _REQUIRED_COLUMNS.difference(pbp.columns)
    if missing:
        raise ValueError(f"play-by-play data missing required columns: {sorted(missing)}")

    plays = pbp.copy()
    plays = plays[plays["posteam"].notna() & plays["defteam"].notna()]
    plays = plays[(plays["pass"].eq(1)) | (plays["rush"].eq(1))]
    plays = plays[~plays["qb_kneel"].eq(1) & ~plays["qb_spike"].eq(1)]
    plays = plays[plays["epa"].notna()]
    return plays


def build_team_game_features(pbp: pd.DataFrame) -> pd.DataFrame:
    """Aggregate cleaned play-by-play into one offensive row per team-game."""
    plays = clean_scrimmage_plays(pbp)

    plays = plays.assign(
        pass_play=plays["pass"].eq(1).astype(int),
        rush_play=plays["rush"].eq(1).astype(int),
        explosive=plays["yards_gained"].ge(20).astype(int),
    )

    grouped = plays.groupby(
        ["game_id", "season", "week", "posteam", "defteam"], dropna=False, sort=True
    )

    features = grouped.agg(
        plays=("epa", "size"),
        epa_per_play=("epa", "mean"),
        success_rate=("success", "mean"),
        explosive_rate=("explosive", "mean"),
        pass_plays=("pass_play", "sum"),
        rush_plays=("rush_play", "sum"),
    ).reset_index()

    pass_only = (
        plays.loc[plays["pass_play"].eq(1)]
        .groupby(["game_id", "posteam"], sort=True)
        .agg(pass_epa_per_play=("epa", "mean"), pass_success_rate=("success", "mean"))
        .reset_index()
    )
    rush_only = (
        plays.loc[plays["rush_play"].eq(1)]
        .groupby(["game_id", "posteam"], sort=True)
        .agg(rush_epa_per_play=("epa", "mean"), rush_success_rate=("success", "mean"))
        .reset_index()
    )

    features = features.merge(pass_only, on=["game_id", "posteam"], how="left")
    features = features.merge(rush_only, on=["game_id", "posteam"], how="left")

    numeric = features.select_dtypes(include=[np.number]).columns
    features[numeric] = features[numeric].replace([np.inf, -np.inf], np.nan)
    return features.sort_values(["season", "week", "game_id", "posteam"]).reset_index(drop=True)


def build_team_weekly_ratings(team_games: pd.DataFrame) -> pd.DataFrame:
    """Create pregame team ratings using only games completed before each week.

    Each row represents a team's information set entering a week. Ratings are
    simple expanding means and therefore leak no current- or future-game data.
    """
    required = {
        "game_id",
        "season",
        "week",
        "posteam",
        "defteam",
        "epa_per_play",
        "success_rate",
        "explosive_rate",
        "pass_epa_per_play",
        "rush_epa_per_play",
    }
    missing = required.difference(team_games.columns)
    if missing:
        raise ValueError(f"team-game features missing required columns: {sorted(missing)}")

    offense = team_games.rename(columns={"posteam": "team"}).copy()
    defense = team_games.rename(columns={"defteam": "team"}).copy()

    offense_metrics = [
        "epa_per_play",
        "success_rate",
        "explosive_rate",
        "pass_epa_per_play",
        "rush_epa_per_play",
    ]

    defense = defense[["season", "week", "game_id", "team", *offense_metrics]].rename(
        columns={metric: f"def_{metric}_allowed" for metric in offense_metrics}
    )
    offense = offense[["season", "week", "game_id", "team", *offense_metrics]].rename(
        columns={metric: f"off_{metric}" for metric in offense_metrics}
    )

    combined = pd.merge(
        offense,
        defense,
        on=["season", "week", "game_id", "team"],
        how="outer",
        validate="one_to_one",
    ).sort_values(["team", "season", "week", "game_id"])

    metric_columns = [column for column in combined.columns if column.startswith(("off_", "def_"))]
    for column in metric_columns:
        combined[f"pregame_{column}"] = combined.groupby(["team", "season"])[column].transform(
            lambda values: values.expanding().mean().shift(1)
        )

    keep = ["season", "week", "game_id", "team", *[f"pregame_{c}" for c in metric_columns]]
    return combined[keep].sort_values(["season", "week", "game_id", "team"]).reset_index(drop=True)
