"""Build auditable current-week QB inputs for Oak's validated QB adjustment."""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from oak_nfl.data.depth_charts import expected_starting_qbs, normalize_depth_charts
from oak_nfl.data.injuries import latest_weekly_status
from oak_nfl.qb import build_qb_game_efficiency


def _league_prior(qb_games: pd.DataFrame) -> float:
    if qb_games.empty:
        return 0.0
    latest_season = int(pd.to_numeric(qb_games["season"], errors="coerce").max())
    prior = qb_games.loc[qb_games["season"].eq(latest_season), "qb_epa_per_dropback"].mean()
    return float(prior) if pd.notna(prior) else 0.0


def _player_name_key(value: object) -> str:
    """Normalize player names for conservative cross-provider matching."""
    if pd.isna(value):
        return ""
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


def expected_starting_qbs_with_availability(
    depth_charts: pd.DataFrame,
    injury_report: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Select each team's expected QB after confirmed-out availability filtering."""
    if injury_report is None or injury_report.empty:
        out = expected_starting_qbs(depth_charts).copy()
        if not out.empty:
            out["expected_qb_source"] = "nflverse-depth-chart"
        return out

    depth = normalize_depth_charts(depth_charts)
    columns = [
        "team",
        "expected_qb_name",
        "expected_qb_id",
        "depth_chart_date",
        "depth_rank",
        "expected_qb_source",
    ]
    if depth.empty:
        return pd.DataFrame(columns=columns)

    position = depth["position"].fillna("")
    qbs = depth.loc[position.eq("QB") | position.str.contains("QUARTERBACK", regex=False)].copy()
    if qbs.empty:
        return pd.DataFrame(columns=columns)

    if qbs["snapshot_date"].notna().any():
        newest = qbs.groupby("team")["snapshot_date"].transform("max")
        qbs = qbs.loc[qbs["snapshot_date"].eq(newest)].copy()

    qbs["depth_rank"] = qbs["depth_rank"].fillna(9999)
    qbs = qbs.sort_values(["team", "depth_rank", "player_name"])
    original_qb1 = qbs.drop_duplicates("team", keep="first").set_index("team")["player_name"]

    statuses = latest_weekly_status(injury_report)
    unavailable = statuses.loc[
        statuses["position_group"].eq("QB") & statuses["status"].eq("out"),
        ["team", "player_name"],
    ].copy()
    unavailable["name_key"] = unavailable["player_name"].map(_player_name_key)
    unavailable_keys = set(zip(unavailable["team"], unavailable["name_key"]))

    qbs["name_key"] = qbs["player_name"].map(_player_name_key)
    qbs = qbs.loc[
        ~qbs.apply(lambda row: (row["team"], row["name_key"]) in unavailable_keys, axis=1)
    ].copy()
    qbs = qbs.drop_duplicates("team", keep="first")

    out = qbs.rename(
        columns={
            "player_name": "expected_qb_name",
            "player_id": "expected_qb_id",
            "snapshot_date": "depth_chart_date",
        }
    )[["team", "expected_qb_name", "expected_qb_id", "depth_chart_date", "depth_rank"]].reset_index(drop=True)
    out["expected_qb_source"] = out.apply(
        lambda row: (
            "nflverse-depth-chart+espn-out-filter"
            if row["team"] in original_qb1.index
            and _player_name_key(row["expected_qb_name"]) != _player_name_key(original_qb1.loc[row["team"]])
            else "nflverse-depth-chart"
        ),
        axis=1,
    )
    return out


def current_qb_epa_ratings(
    pbp: pd.DataFrame,
    *,
    prior_dropbacks: float = 150.0,
    recency_decay: float = 0.92,
) -> pd.DataFrame:
    """Estimate each QB's current EPA/dropback using completed games only."""
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


def embedded_team_qb_values(
    pbp: pd.DataFrame,
    *,
    prior_dropbacks: float = 150.0,
    recency_decay: float = 0.92,
) -> pd.DataFrame:
    """Estimate the QB value embedded in each team's recent performance.

    The live adjustment should compare the upcoming starter to the quarterback
    contribution already present in the team rating, not merely to whichever QB
    handled the most dropbacks in the latest game. We therefore aggregate every
    completed QB game for the team, weight each contribution by dropbacks and game
    recency, and regress the result toward the same league prior used by the
    individual-QB ratings.

    This naturally handles split games, mid-game injuries, multi-QB seasons and
    offseason roster turnover. A backup finishing one game cannot instantly become
    the entire baseline, while departed quarterbacks remain represented only to the
    extent their historical play is still part of the team's recent performance.
    """
    games = build_qb_game_efficiency(pbp).sort_values(["posteam", "season", "week", "game_id"])
    if games.empty:
        return pd.DataFrame(
            columns=["team", "baseline_qb_epa", "baseline_qb_name", "baseline_qb_id", "baseline_qb_source"]
        )

    prior = _league_prior(games)
    game_order = (
        games[["posteam", "season", "week", "game_id"]]
        .drop_duplicates()
        .sort_values(["posteam", "season", "week", "game_id"])
        .copy()
    )
    game_order["team_game_index"] = game_order.groupby("posteam").cumcount()
    game_order["team_game_max"] = game_order.groupby("posteam")["team_game_index"].transform("max")
    game_order["game_age"] = game_order["team_game_max"] - game_order["team_game_index"]

    weighted = games.merge(
        game_order[["posteam", "season", "week", "game_id", "game_age"]],
        on=["posteam", "season", "week", "game_id"],
        how="left",
        validate="many_to_one",
    )
    weighted["qb_dropbacks"] = pd.to_numeric(weighted["qb_dropbacks"], errors="coerce").fillna(0.0)
    weighted["qb_epa_per_dropback"] = pd.to_numeric(
        weighted["qb_epa_per_dropback"], errors="coerce"
    ).fillna(prior)
    weighted["embedded_weight"] = (
        np.power(recency_decay, pd.to_numeric(weighted["game_age"], errors="coerce").fillna(0.0))
        * weighted["qb_dropbacks"]
    )
    weighted["weighted_epa"] = weighted["embedded_weight"] * weighted["qb_epa_per_dropback"]

    rows: list[dict[str, object]] = []
    for team, group in weighted.groupby("posteam"):
        total = float(group["embedded_weight"].sum())
        observed = float(group["weighted_epa"].sum() / total) if total > 0 else prior
        rating = (prior_dropbacks * prior + total * observed) / (prior_dropbacks + total)
        rows.append(
            {
                "team": team,
                "baseline_qb_epa": float(rating),
                "baseline_qb_name": "Embedded team QB",
                "baseline_qb_id": pd.NA,
                "baseline_qb_source": "recency-dropback-weighted-team-qb",
            }
        )
    return pd.DataFrame(rows)


def build_live_qb_inputs(
    pbp: pd.DataFrame,
    slate: pd.DataFrame,
    depth_charts: pd.DataFrame,
    injury_report: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build game-level expected-vs-embedded QB EPA inputs."""
    expected = expected_starting_qbs_with_availability(depth_charts, injury_report)
    baseline = embedded_team_qb_values(pbp)
    ratings = current_qb_epa_ratings(pbp)

    expected = expected.merge(
        ratings[["qb_id", "current_qb_epa"]],
        left_on="expected_qb_id",
        right_on="qb_id",
        how="left",
    ).drop(columns=["qb_id"], errors="ignore")
    expected = expected.rename(columns={"current_qb_epa": "expected_qb_epa"})

    team_context = baseline.merge(expected, on="team", how="outer")
    has_expected = team_context["expected_qb_id"].notna() & team_context["expected_qb_epa"].notna()
    has_baseline = team_context["baseline_qb_epa"].notna()
    team_context["qb_context_confidence"] = np.where(
        has_expected & has_baseline,
        np.where(
            team_context.get("expected_qb_source", pd.Series(index=team_context.index, dtype="object"))
            .fillna("")
            .eq("nflverse-depth-chart+espn-out-filter"),
            "availability-filtered+embedded-baseline",
            "depth-chart+embedded-baseline",
        ),
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
            item[f"{side}_qb_context_source"] = context.get("expected_qb_source", pd.NA) if context is not None else pd.NA
            item[f"{side}_baseline_qb_source"] = context.get("baseline_qb_source", pd.NA) if context is not None else pd.NA
            item[f"{side}_qb_context_confidence"] = context["qb_context_confidence"] if context is not None else "missing"
        rows.append(item)
    return pd.DataFrame(rows)
