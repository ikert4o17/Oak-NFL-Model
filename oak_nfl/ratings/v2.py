"""Oak V2 ratings: prior-season priors blended with recency-weighted current form."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _weighted_mean(values: pd.Series, decay: float) -> float:
    clean = values.dropna().to_numpy(dtype=float)
    if clean.size == 0:
        return np.nan
    ages = np.arange(clean.size - 1, -1, -1, dtype=float)
    weights = np.power(decay, ages)
    return float(np.average(clean, weights=weights))


def build_v2_pregame_ratings(
    team_games: pd.DataFrame,
    *,
    prior_games: float = 4.0,
    prior_regression: float = 0.50,
    recency_decay: float = 0.85,
) -> pd.DataFrame:
    """Build leakage-safe pregame ratings with preseason priors and recency.

    A team's preseason prior is its previous-season EPA/play offense and defense,
    regressed toward league average. During the current season, completed games
    receive exponentially larger weights as they become more recent. ``prior_games``
    controls how much evidence the preseason prior contributes at the start.
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

    games = team_games.sort_values(["season", "week", "game_id"]).copy()
    seasons = sorted(games["season"].dropna().astype(int).unique())
    rows: list[dict[str, float | int | str]] = []

    # Previous-season team means become next season's priors. Defensive EPA is
    # stored as EPA/play allowed, so lower is better until baseline.py flips sign.
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
            for game in week_games.itertuples(index=False):
                for team in (game.posteam, game.defteam):
                    if team not in priors:
                        continue
                    prior_off, prior_def = priors[team]
                    current_off = _weighted_mean(pd.Series(off_history[team]), recency_decay)
                    current_def = _weighted_mean(pd.Series(def_history[team]), recency_decay)
                    n = len(off_history[team])
                    current_weight = float(n)
                    total_weight = prior_games + current_weight
                    if total_weight == 0:
                        blended_off, blended_def = 0.0, 0.0
                    elif n == 0:
                        blended_off, blended_def = prior_off, prior_def
                    else:
                        blended_off = (prior_games * prior_off + current_weight * current_off) / total_weight
                        blended_def = (prior_games * prior_def + current_weight * current_def) / total_weight
                    rows.append(
                        {
                            "season": season,
                            "week": int(week),
                            "game_id": game.game_id,
                            "team": team,
                            "pregame_off_epa_per_play": blended_off,
                            "pregame_def_epa_per_play_allowed": blended_def,
                            "games_played": n,
                        }
                    )

            # Update histories only after every game in the week has been rated.
            for game in week_games.itertuples(index=False):
                off_history.setdefault(game.posteam, []).append(float(game.epa_per_play))
                def_history.setdefault(game.defteam, []).append(float(game.epa_per_play))

    return pd.DataFrame(rows).drop_duplicates(["season", "week", "game_id", "team"])
