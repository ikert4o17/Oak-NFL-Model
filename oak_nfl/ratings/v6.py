"""Oak V6 challenger: test passing/rushing EPA beyond promoted V5."""

from __future__ import annotations

import numpy as np
import pandas as pd

_METRICS = (
    "epa_per_play",
    "pass_epa_per_play",
    "rush_epa_per_play",
    "success_rate",
    "explosive_rate",
)


def _weighted_mean(values: list[float], decay: float) -> float:
    clean = np.asarray([value for value in values if pd.notna(value)], dtype=float)
    if clean.size == 0:
        return np.nan
    ages = np.arange(clean.size - 1, -1, -1, dtype=float)
    return float(np.average(clean, weights=np.power(decay, ages)))


def _blend(prior: float, history: list[float], prior_games: float, decay: float) -> float:
    current = _weighted_mean(history, decay)
    n = sum(pd.notna(value) for value in history)
    if n == 0 or pd.isna(current):
        return float(prior)
    return float((prior_games * prior + n * current) / (prior_games + n))


def build_v6_pregame_ratings(
    team_games: pd.DataFrame,
    *,
    prior_games: float = 4.0,
    prior_regression: float = 0.50,
    recency_decay: float = 0.85,
) -> pd.DataFrame:
    """Build leakage-safe total/pass/rush offense and defense ratings plus V5 context."""
    required = {"game_id", "season", "week", "posteam", "defteam", *_METRICS}
    missing = required.difference(team_games.columns)
    if missing:
        raise ValueError(f"team-game features missing required columns: {sorted(missing)}")

    games = team_games.sort_values(["season", "week", "game_id"]).copy()
    seasons = sorted(games["season"].dropna().astype(int).unique())
    rows: list[dict[str, float | int | str]] = []
    yearly_off = {m: games.groupby(["season", "posteam"])[m].mean() for m in _METRICS}
    yearly_def = {m: games.groupby(["season", "defteam"])[m].mean() for m in _METRICS}
    yearly_league = {m: games.groupby("season")[m].mean() for m in _METRICS}

    for season in seasons:
        season_games = games.loc[games["season"].eq(season)]
        teams = sorted(set(season_games["posteam"].dropna()) | set(season_games["defteam"].dropna()))
        previous = season - 1
        priors: dict[str, dict[str, tuple[float, float]]] = {team: {} for team in teams}
        for team in teams:
            for metric in _METRICS:
                league = float(yearly_league[metric].get(previous, games[metric].mean()))
                raw_off = float(yearly_off[metric].get((previous, team), league))
                raw_def = float(yearly_def[metric].get((previous, team), league))
                priors[team][metric] = (
                    league + prior_regression * (raw_off - league),
                    league + prior_regression * (raw_def - league),
                )

        off_history = {team: {m: [] for m in _METRICS} for team in teams}
        def_history = {team: {m: [] for m in _METRICS} for team in teams}
        for week in sorted(season_games["week"].dropna().unique()):
            week_games = season_games.loc[season_games["week"].eq(week)]
            snapshot: dict[str, dict[str, tuple[float, float]]] = {team: {} for team in teams}
            for team in teams:
                for metric in _METRICS:
                    prior_off, prior_def = priors[team][metric]
                    snapshot[team][metric] = (
                        _blend(prior_off, off_history[team][metric], prior_games, recency_decay),
                        _blend(prior_def, def_history[team][metric], prior_games, recency_decay),
                    )
            for game in week_games.itertuples(index=False):
                for team in (game.posteam, game.defteam):
                    row: dict[str, float | int | str] = {
                        "season": season,
                        "week": int(week),
                        "game_id": game.game_id,
                        "team": team,
                    }
                    for metric in _METRICS:
                        off_value, def_value = snapshot[team][metric]
                        row[f"pregame_off_{metric}"] = off_value
                        row[f"pregame_def_{metric}_allowed"] = def_value
                    rows.append(row)
            # Freeze each week's snapshot before incorporating any games from that week.
            for game in week_games.itertuples(index=False):
                for metric in _METRICS:
                    value = float(getattr(game, metric))
                    off_history[game.posteam][metric].append(value)
                    def_history[game.defteam][metric].append(value)

    return pd.DataFrame(rows).drop_duplicates(["season", "week", "game_id", "team"])
