"""Quarterback identity and point-in-time value features for Oak V8."""

from __future__ import annotations

import numpy as np
import pandas as pd


def identify_game_qbs(pbp: pd.DataFrame) -> pd.DataFrame:
    """Identify each team's primary QB in a game by pass-attempt/dropback volume.

    Identity is taken from the game itself, but no current-game performance enters
    the quarterback rating. In live production this field will be replaced by the
    expected starter from the weekly depth-chart/injury pipeline.
    """
    required = {"game_id", "season", "week", "posteam", "passer_player_id", "passer_player_name", "pass"}
    missing = required.difference(pbp.columns)
    if missing:
        raise ValueError(f"QB identification missing required columns: {sorted(missing)}")
    passes = pbp.loc[pbp["pass"].eq(1) & pbp["passer_player_id"].notna()].copy()
    counts = (
        passes.groupby(
            ["game_id", "season", "week", "posteam", "passer_player_id", "passer_player_name"],
            dropna=False,
        )
        .size()
        .rename("dropbacks")
        .reset_index()
    )
    counts = counts.sort_values(
        ["game_id", "posteam", "dropbacks", "passer_player_id"],
        ascending=[True, True, False, True],
    )
    return counts.drop_duplicates(["game_id", "posteam"]).reset_index(drop=True)


def build_qb_game_efficiency(pbp: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-game QB passing efficiency for historical rating updates."""
    required = {
        "game_id", "season", "week", "posteam", "passer_player_id", "passer_player_name",
        "pass", "epa", "cpoe", "sack",
    }
    missing = required.difference(pbp.columns)
    if missing:
        raise ValueError(f"QB efficiency missing required columns: {sorted(missing)}")
    plays = pbp.loc[pbp["pass"].eq(1) & pbp["passer_player_id"].notna() & pbp["epa"].notna()].copy()
    plays["sack_flag"] = plays["sack"].fillna(0).eq(1).astype(float)
    return (
        plays.groupby(
            ["game_id", "season", "week", "posteam", "passer_player_id", "passer_player_name"],
            dropna=False,
        )
        .agg(
            qb_dropbacks=("epa", "size"),
            qb_epa_per_dropback=("epa", "mean"),
            qb_cpoe=("cpoe", "mean"),
            qb_sack_rate=("sack_flag", "mean"),
        )
        .reset_index()
    )


def build_pregame_qb_ratings(
    pbp: pd.DataFrame,
    *,
    prior_dropbacks: float = 150.0,
    recency_decay: float = 0.92,
) -> pd.DataFrame:
    """Create leakage-safe QB ratings for each actual game starter.

    Ratings use only that quarterback's completed prior games. Small samples are
    regressed toward season-level league averages using ``prior_dropbacks``.
    """
    starters = identify_game_qbs(pbp)
    qb_games = build_qb_game_efficiency(pbp).sort_values(["season", "week", "game_id"])
    league = qb_games.groupby("season").agg(
        league_epa=("qb_epa_per_dropback", "mean"),
        league_cpoe=("qb_cpoe", "mean"),
        league_sack=("qb_sack_rate", "mean"),
    )
    history: dict[str, list[tuple[int, float, float, float, float]]] = {}
    rows: list[dict[str, float | int | str]] = []

    for starter in starters.sort_values(["season", "week", "game_id"]).itertuples(index=False):
        season = int(starter.season)
        qb_id = starter.passer_player_id
        previous_league = league.loc[season - 1] if season - 1 in league.index else league.loc[season]
        games = history.get(qb_id, [])
        if games:
            ages = np.arange(len(games) - 1, -1, -1, dtype=float)
            weights = np.power(recency_decay, ages) * np.array([g[0] for g in games], dtype=float)
            total_hist = float(weights.sum())
            epa = float(np.average([g[1] for g in games], weights=weights))
            cpoe = float(np.average([g[2] for g in games], weights=weights))
            sack = float(np.average([g[3] for g in games], weights=weights))
        else:
            total_hist = 0.0
            epa = float(previous_league.league_epa)
            cpoe = float(previous_league.league_cpoe)
            sack = float(previous_league.league_sack)
        denom = prior_dropbacks + total_hist
        prior_epa = float(previous_league.league_epa)
        prior_cpoe = float(previous_league.league_cpoe)
        prior_sack = float(previous_league.league_sack)
        rows.append({
            "game_id": starter.game_id,
            "season": season,
            "week": int(starter.week),
            "team": starter.posteam,
            "qb_id": qb_id,
            "qb_name": starter.passer_player_name,
            "pregame_qb_epa": (prior_dropbacks * prior_epa + total_hist * epa) / denom,
            "pregame_qb_cpoe": (prior_dropbacks * prior_cpoe + total_hist * cpoe) / denom,
            "pregame_qb_sack_rate": (prior_dropbacks * prior_sack + total_hist * sack) / denom,
            "prior_qb_dropbacks": total_hist,
        })
        current = qb_games.loc[
            qb_games["game_id"].eq(starter.game_id)
            & qb_games["posteam"].eq(starter.posteam)
            & qb_games["passer_player_id"].eq(qb_id)
        ]
        if not current.empty:
            r = current.iloc[0]
            history.setdefault(qb_id, []).append((
                int(r.qb_dropbacks), float(r.qb_epa_per_dropback),
                float(r.qb_cpoe) if pd.notna(r.qb_cpoe) else prior_cpoe,
                float(r.qb_sack_rate), float(season),
            ))
    return pd.DataFrame(rows)
