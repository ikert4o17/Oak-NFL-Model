"""Build auditable current-week QB inputs for Oak's validated QB adjustment."""

from __future__ import annotations

import numpy as np
import pandas as pd

from oak_nfl.data.depth_charts import expected_starting_qbs
from oak_nfl.qb import build_qb_game_efficiency, identify_game_qbs


def _league_prior(qb_games: pd.DataFrame) -> float:
    if qb_games.empty:
        return 0.0
    latest_season = int(pd.to_numeric(qb_games["season"], errors="coerce").max())
    prior = qb_games.loc[qb_games["season"].eq(latest_season), "qb_epa_per_dropback"].mean()
    return float(prior) if pd.notna(prior) else 0.0


def current_qb_epa_ratings(
    pbp: pd.DataFrame,
    *,
    prior_dropbacks: float = 150.0,
    recency_decay: float = 0.92,
) -> pd.DataFrame:
    """Estimate each QB's current EPA/dropback using completed games only.

    This mirrors the already-researched QB rating philosophy: player-game history
    is recency weighted and regressed toward the latest completed season's league
    level. It is context generation, not a new fitted model.
    """
    games = build_qb_game_efficiency(pbp).sort_values(["season", "week", "game_id"])
    if games.empty:
        return pd.DataFrame(columns=["qb_id", "qb_name", "current_qb_epa", "prior_qb_dropbacks"])

    prior = _league_prior(games)
    rows: list[dict[str, object]] = []
    for qb_id, group in games.groupby("passer_player_id", dropna=False):
        group = group.reset_index(drop=True)
        ages = np.arange(len(group) - 1, -1, -1, dtype=float)
        dropbacks = pd.to_numeric(group["qb_dropbacks"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        weights = np.power(recency_decay, ages) * dropbacks
        total = float(weights.sum())
        values = pd.to_numeric(group["qb_epa_per_dropback"], errors="coerce").fillna(prior).to_numpy(dtype=float)
        weighted = float(np.average(values, weights=weights)) if total > 0 else prior
        rating = (prior_dropbacks * prior + total * weighted) / (prior_dropbacks + total)
        rows.append(
            {
                "qb_id": qb_id,
                "qb_name": group.iloc[-1]["passer_player_name"],
                "current_qb_epa": float(rating),
                "prior_qb_dropbacks": total,
            }
        )
    return pd.DataFrame(rows)


def latest_team_qbs(pbp: pd.DataFrame) -> pd.DataFrame:
    """Return each team's primary QB from its latest completed game."""
    starters = identify_game_qbs(pbp)
    if starters.empty:
        return pd.DataFrame(columns=["team", "baseline_qb_id", "baseline_qb_name"])
    starters = starters.sort_values(["season", "week", "game_id"])
    latest = starters.drop_duplicates("posteam", keep="last")
    return latest.rename(
        columns={
            "posteam": "team",
            "passer_player_id": "baseline_qb_id",
            "passer_player_name": "baseline_qb_name",
        }
    )[["team", "baseline_qb_id", "baseline_qb_name"]].reset_index(drop=True)


def build_live_qb_inputs(
    pbp: pd.DataFrame,
    slate: pd.DataFrame,
    depth_charts: pd.DataFrame,
) -> pd.DataFrame:
    """Build game-level expected-vs-baseline QB EPA inputs.

    Expected QBs come from the newest available depth chart. Baseline QBs are the
    team's primary passer in its latest completed game, which is the safest proxy
    for the QB already embedded in V5 team performance. Missing context is left as
    NaN so Oak's validated adjustment safely resolves to zero rather than guessing.
    """
    expected = expected_starting_qbs(depth_charts)
    baseline = latest_team_qbs(pbp)
    ratings = current_qb_epa_ratings(pbp)

    expected = expected.merge(
        ratings[["qb_id", "current_qb_epa"]],
        left_on="expected_qb_id",
        right_on="qb_id",
        how="left",
    ).drop(columns=["qb_id"], errors="ignore")
    expected = expected.rename(columns={"current_qb_epa": "expected_qb_epa"})

    baseline = baseline.merge(
        ratings[["qb_id", "current_qb_epa"]],
        left_on="baseline_qb_id",
        right_on="qb_id",
        how="left",
    ).drop(columns=["qb_id"], errors="ignore")
    baseline = baseline.rename(columns={"current_qb_epa": "baseline_qb_epa"})

    team_context = baseline.merge(expected, on="team", how="outer")
    team_context["qb_context_source"] = "nflverse depth charts + completed nflverse PBP"
    team_context["qb_context_confidence"] = np.where(
        team_context["expected_qb_id"].notna() & team_context["baseline_qb_id"].notna(),
        "depth-chart",
        "missing",
    )

    by_team = team_context.set_index("team", drop=False) if not team_context.empty else team_context
    rows: list[dict[str, object]] = []
    for game in slate.itertuples(index=False):
        item: dict[str, object] = {"game_id": game.game_id}
        for side in ("home", "away"):
            team = getattr(game, f"{side}_team")
            context = by_team.loc[team] if not team_context.empty and team in by_team.index else None
            if isinstance(context, pd.DataFrame):
                context = context.iloc[0]
            item[f"{side}_expected_qb_epa"] = context["expected_qb_epa"] if context is not None else np.nan
            item[f"{side}_baseline_qb_epa"] = context["baseline_qb_epa"] if context is not None else np.nan
            item[f"{side}_expected_qb_name"] = context["expected_qb_name"] if context is not None else pd.NA
            item[f"{side}_baseline_qb_name"] = context["baseline_qb_name"] if context is not None else pd.NA
            item[f"{side}_depth_chart_date"] = context["depth_chart_date"] if context is not None else pd.NaT
            item[f"{side}_qb_context_confidence"] = (
                context["qb_context_confidence"] if context is not None else "missing"
            )
        rows.append(item)
    return pd.DataFrame(rows)
