"""Oak V3 ratings: V2 priors and recency plus leakage-safe opponent adjustment."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _weighted_mean(values: list[float], decay: float) -> float:
    if not values:
        return np.nan
    clean = np.asarray(values, dtype=float)
    ages = np.arange(clean.size - 1, -1, -1, dtype=float)
    weights = np.power(decay, ages)
    return float(np.average(clean, weights=weights))


def _blend(prior: float, history: list[float], prior_games: float, decay: float) -> float:
    n = len(history)
    if n == 0:
        return float(prior)
    current = _weighted_mean(history, decay)
    return float((prior_games * prior + n * current) / (prior_games + n))


def build_v3_pregame_ratings(
    team_games: pd.DataFrame,
    *,
    prior_games: float = 4.0,
    prior_regression: float = 0.50,
    recency_decay: float = 0.85,
    opponent_weight: float = 0.50,
) -> pd.DataFrame:
    """Build leakage-safe pregame ratings with opponent-adjusted observations.

    V3 keeps V2's prior-season priors and recency weighting. After each completed
    week, that week's EPA observations are adjusted by the opponent ratings that
    existed *before* those games. Strong defenses therefore raise the value of an
    offensive performance, while strong offenses lower EPA allowed for defenses.
    No current-week result is used to rate that same week's games.
    """
    required = {"game_id", "season", "week", "posteam", "defteam", "epa_per_play"}
    missing = required.difference(team_games.columns)
    if missing:
        raise ValueError(f"team-game features missing required columns: {sorted(missing)}")
    if prior_games < 0:
        raise ValueError("prior_games must be non-negative")
    if not 0 <= prior_regression <= 1:
        raise ValueError("prior_regression must be between 0 and 1")
    if not 0 < recency_decay <= 1:
        raise ValueError("recency_decay must be in (0, 1]")
    if opponent_weight < 0:
        raise ValueError("opponent_weight must be non-negative")

    games = team_games.sort_values(["season", "week", "game_id"]).copy()
    seasons = sorted(games["season"].dropna().astype(int).unique())
    rows: list[dict[str, float | int | str]] = []

    offense_year = games.groupby(["season", "posteam"])["epa_per_play"].mean()
    defense_year = games.groupby(["season", "defteam"])["epa_per_play"].mean()
    league_year = games.groupby("season")["epa_per_play"].mean()

    for season in seasons:
        season_games = games.loc[games["season"].eq(season)]
        previous = season - 1
        league_prior = float(league_year.get(previous, 0.0))
        teams = sorted(set(season_games["posteam"].dropna()) | set(season_games["defteam"].dropna()))

        priors: dict[str, tuple[float, float]] = {}
        for team in teams:
            raw_off = float(offense_year.get((previous, team), league_prior))
            raw_def = float(defense_year.get((previous, team), league_prior))
            prior_off = league_prior + prior_regression * (raw_off - league_prior)
            prior_def = league_prior + prior_regression * (raw_def - league_prior)
            priors[team] = (prior_off, prior_def)

        off_history: dict[str, list[float]] = {team: [] for team in teams}
        def_history: dict[str, list[float]] = {team: [] for team in teams}

        for week in sorted(season_games["week"].dropna().unique()):
            week_games = season_games.loc[season_games["week"].eq(week)]
            snapshot: dict[str, tuple[float, float]] = {}
            for team in teams:
                prior_off, prior_def = priors[team]
                snapshot[team] = (
                    _blend(prior_off, off_history[team], prior_games, recency_decay),
                    _blend(prior_def, def_history[team], prior_games, recency_decay),
                )

            league_off = float(np.mean([value[0] for value in snapshot.values()]))
            league_def = float(np.mean([value[1] for value in snapshot.values()]))

            for game in week_games.itertuples(index=False):
                for team in (game.posteam, game.defteam):
                    if team not in snapshot:
                        continue
                    blended_off, blended_def = snapshot[team]
                    rows.append(
                        {
                            "season": season,
                            "week": int(week),
                            "game_id": game.game_id,
                            "team": team,
                            "pregame_off_epa_per_play": blended_off,
                            "pregame_def_epa_per_play_allowed": blended_def,
                            "games_played": len(off_history[team]),
                        }
                    )

            # Only completed prior-week information enters the next week's history.
            for game in week_games.itertuples(index=False):
                if game.posteam not in snapshot or game.defteam not in snapshot:
                    continue
                opp_off, opp_def = snapshot[game.defteam]
                observed = float(game.epa_per_play)
                adjusted_off = observed - opponent_weight * (opp_def - league_def)
                adjusted_def = observed - opponent_weight * (snapshot[game.posteam][0] - league_off)
                off_history[game.posteam].append(adjusted_off)
                def_history[game.defteam].append(adjusted_def)

    return pd.DataFrame(rows).drop_duplicates(["season", "week", "game_id", "team"])
