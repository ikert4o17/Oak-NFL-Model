"""Benchmark non-QB injury availability against frozen Oak V5.

Historical nflverse injury reports are reliable through 2024, so V10 trains
on 2014-2022 and grades on 2023-2024. Player importance is estimated only from
prior snap share; current-week snaps never enter the injury value.
"""

from __future__ import annotations

import pandas as pd

from oak_nfl.backtest import evaluate_margin_predictions
from oak_nfl.data.games import build_game_results
from oak_nfl.data.injuries import latest_weekly_status, normalize_injury_feed
from oak_nfl.data.nflverse import load_injuries, load_pbp, load_players, load_snap_counts
from oak_nfl.features import build_team_game_features
from oak_nfl.personnel import POSITION_POINT_CAPS, player_absence_points
from oak_nfl.personnel_value import (
    attach_player_values,
    attach_snap_player_ids,
    build_pregame_player_values,
)
from oak_nfl.ratings.v5 import build_v5_game_predictions, build_v5_pregame_ratings

GROUPS = {
    "ol": {"OT", "IOL"},
    "receivers": {"WR", "TE"},
    "rb": {"RB"},
    "pass_rush": {"EDGE", "IDL"},
    "lb": {"LB"},
    "secondary": {"CB", "S"},
    "all_non_qb": set(POSITION_POINT_CAPS),
}


def _canonical_injuries(raw: pd.DataFrame) -> pd.DataFrame:
    data = raw.loc[
        raw["report_status"].notna()
        & raw["report_status"].astype(str).str.strip().ne("")
    ].copy()
    normalized = normalize_injury_feed(
        data,
        column_map={
            "gsis_id": "player_id",
            "full_name": "player_name",
            "position": "position_group",
            "report_status": "status",
            "date_modified": "report_date",
        },
        source="nflverse_injuries",
    )
    return latest_weekly_status(normalized)


def _weekly_team_adjustments(
    availability: pd.DataFrame,
    *,
    positions: set[str],
    scale: float,
    team_cap: float,
) -> pd.DataFrame:
    frame = availability[availability["position_group"].isin(positions)].copy()
    frame = frame[frame["position_group"].ne("QB")]
    frame["absence_points"] = [
        scale * player_absence_points(pos, value, status)
        for pos, value, status in zip(
            frame.position_group,
            frame.player_value,
            frame.status,
        )
    ]
    grouped = (
        frame.groupby(["season", "week", "team"], as_index=False)["absence_points"]
        .sum()
        .rename(columns={"absence_points": "personnel_points"})
    )
    grouped["personnel_points"] = grouped["personnel_points"].clip(-team_cap, team_cap)
    return grouped


def _apply_weekly_adjustments(
    predictions: pd.DataFrame,
    adjustments: pd.DataFrame,
) -> pd.DataFrame:
    home = adjustments.rename(
        columns={"team": "home_team", "personnel_points": "home_personnel_points"}
    )
    away = adjustments.rename(
        columns={"team": "away_team", "personnel_points": "away_personnel_points"}
    )
    out = predictions.merge(
        home,
        on=["season", "week", "home_team"],
        how="left",
    ).merge(
        away,
        on=["season", "week", "away_team"],
        how="left",
    )
    out[["home_personnel_points", "away_personnel_points"]] = out[
        ["home_personnel_points", "away_personnel_points"]
    ].fillna(0.0)
    out["predicted_home_margin"] = (
        out["predicted_home_margin"]
        + out["home_personnel_points"]
        - out["away_personnel_points"]
    )
    return out


def _coverage_diagnostics(
    availability: pd.DataFrame,
    values: pd.DataFrame,
) -> None:
    """Report whether missing coverage is concentrated among low-usage players."""
    matched = availability["player_value"].gt(0)
    print("=== OAK V10 DATA COVERAGE ===")
    print(f"injury rows: {len(availability)}")
    print(f"injury rows with GSIS id: {availability['player_id'].notna().mean():.4f}")
    print(f"rows with prior snap value: {matched.mean():.4f}")

    latest = values[values["player_value"].gt(0)].sort_values(["season", "week"]).copy()
    known_ids = set(latest["player_id"].dropna().astype(str))
    availability = availability.copy()
    availability["ever_known_snap_player"] = availability["player_id"].astype(str).isin(
        known_ids
    )
    known_share = availability["ever_known_snap_player"].mean()
    print(f"injury rows belonging to a known snap player: {known_share:.4f}")
    known = availability[availability["ever_known_snap_player"]]
    if len(known):
        print(
            "coverage among known snap players: "
            f"{known['player_value'].gt(0).mean():.4f}"
        )

    print("\ncoverage by position group")
    by_pos = availability.groupby("position_group").agg(
        injury_rows=("player_name", "size"),
        matched=("player_value", lambda s: int(s.gt(0).sum())),
        mean_value=("player_value", "mean"),
    )
    by_pos["coverage"] = by_pos["matched"] / by_pos["injury_rows"]
    print(by_pos.sort_values("injury_rows", ascending=False).to_string())

    print("\ncoverage by season-week bucket")
    buckets = availability.copy()
    buckets["week_bucket"] = pd.cut(
        buckets["week"],
        bins=[0, 3, 8, 13, 30],
        labels=["W1-3", "W4-8", "W9-13", "W14+"],
    )
    by_week = buckets.groupby("week_bucket", observed=True).agg(
        injury_rows=("player_name", "size"),
        matched=("player_value", lambda s: int(s.gt(0).sum())),
    )
    by_week["coverage"] = by_week["matched"] / by_week["injury_rows"]
    print(by_week.to_string())


def run() -> None:
    pbp = pd.concat(
        [load_pbp(year) for year in range(2014, 2025)],
        ignore_index=True,
    )
    snaps = pd.concat(
        [load_snap_counts(year) for year in range(2014, 2025)],
        ignore_index=True,
    )
    injuries = pd.concat(
        [load_injuries(year) for year in range(2014, 2025)],
        ignore_index=True,
    )
    players = load_players()
    snaps = attach_snap_player_ids(snaps, players)
    games = build_game_results(pbp)
    v5 = build_v5_game_predictions(
        games,
        build_v5_pregame_ratings(build_team_game_features(pbp)),
    )
    values = build_pregame_player_values(snaps)
    availability = attach_player_values(_canonical_injuries(injuries), values)
    _coverage_diagnostics(availability, values)

    holdout = v5[v5["season"].between(2023, 2024)].dropna(
        subset=["predicted_home_margin", "actual_home_margin"]
    )
    print("\n=== OAK V10 HOLDOUT 2023-2024 ===")
    print("V5 CONTROL", evaluate_margin_predictions(holdout))

    rows = []
    for group, positions in GROUPS.items():
        for scale in [0.25, 0.50, 0.75, 1.00]:
            for cap in [1.0, 2.0, 3.0]:
                adjustments = _weekly_team_adjustments(
                    availability,
                    positions=positions,
                    scale=scale,
                    team_cap=cap,
                )
                adjusted = _apply_weekly_adjustments(holdout, adjustments)
                metrics = evaluate_margin_predictions(adjusted)
                rows.append({"group": group, "scale": scale, "cap": cap, **metrics})
    results = pd.DataFrame(rows).sort_values(["mae", "rmse"])
    print(results.head(20).to_string(index=False))


if __name__ == "__main__":
    run()
