"""Survivor-pool recommendations derived downstream from Oak outputs."""
from __future__ import annotations

from dataclasses import dataclass
from math import exp

import pandas as pd


@dataclass(frozen=True)
class Candidate:
    team: str
    opponent: str
    location: str
    margin: float
    win_confidence: float
    future_value: float
    path_a_score: float
    path_b_score: float


def margin_to_confidence(margin: float) -> float:
    """Monotonic survival-confidence transform from Oak projected margin.

    This is a ranking aid, not a calibrated betting probability.
    """
    return 1.0 / (1.0 + exp(-float(margin) / 6.5))


def _team_margin(row: pd.Series, team: str) -> float:
    home_margin = float(row["predicted_home_margin"])
    return home_margin if team == row["home_team"] else -home_margin


def current_candidates(card: pd.DataFrame, used: set[str]) -> list[dict]:
    rows: list[dict] = []
    for _, game in card.iterrows():
        for team, opp, loc in (
            (str(game.home_team), str(game.away_team), "vs"),
            (str(game.away_team), str(game.home_team), "@"),
        ):
            if team in used:
                continue
            margin = _team_margin(game, team)
            if margin <= 0:
                continue
            rows.append({"team": team, "opponent": opp, "location": loc, "margin": margin})
    return rows


def future_value_by_team(
    schedule: pd.DataFrame,
    ratings: dict[str, float],
    *,
    current_week: int,
    horizon: int = 6,
    home_field_proxy: float = 1.5,
) -> dict[str, float]:
    """Estimate each team's best remaining survivor spot from Oak power ratings."""
    future = schedule.loc[
        schedule["week"].gt(current_week) & schedule["week"].le(current_week + horizon)
    ].copy()
    values = {team: 0.0 for team in ratings}
    for row in future.itertuples(index=False):
        home, away = str(row.home_team), str(row.away_team)
        if home not in ratings or away not in ratings:
            continue
        neutral = str(getattr(row, "location", "Home")).lower() == "neutral"
        hfa = 0.0 if neutral else home_field_proxy
        home_margin = ratings[home] - ratings[away] + hfa
        away_margin = -home_margin
        values[home] = max(values.get(home, 0.0), margin_to_confidence(home_margin))
        values[away] = max(values.get(away, 0.0), margin_to_confidence(away_margin))
    return values


def rank_candidates(card: pd.DataFrame, future_values: dict[str, float], used: set[str]) -> list[Candidate]:
    ranked: list[Candidate] = []
    for row in current_candidates(card, used):
        conf = margin_to_confidence(row["margin"])
        future = float(future_values.get(row["team"], 0.0))
        # Path A protects teams that have premium future spots. Path B is almost
        # entirely current-week survival with only a light future tiebreaker.
        path_a = conf - 0.12 * future
        path_b = conf - 0.02 * future
        ranked.append(Candidate(
            team=row["team"], opponent=row["opponent"], location=row["location"],
            margin=float(row["margin"]), win_confidence=conf, future_value=future,
            path_a_score=path_a, path_b_score=path_b,
        ))
    return ranked


def pick_two(candidates: list[Candidate], path: str) -> list[Candidate]:
    key = (lambda c: c.path_a_score) if path == "a" else (lambda c: c.path_b_score)
    return sorted(candidates, key=lambda c: (key(c), c.win_confidence), reverse=True)[:2]
