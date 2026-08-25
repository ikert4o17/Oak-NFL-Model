"""Benchmark explicit quarterback-change adjustments on historical starter switches."""

from __future__ import annotations

import pandas as pd

from oak_nfl.backtest import evaluate_margin_predictions
from oak_nfl.data.games import build_game_results
from oak_nfl.data.nflverse import load_pbp
from oak_nfl.features import build_team_game_features
from oak_nfl.qb import build_pregame_qb_ratings
from oak_nfl.qb_adjustment import qb_change_points
from oak_nfl.ratings.v5 import build_v5_game_predictions, build_v5_pregame_ratings


def run() -> None:
    pbp = pd.concat([load_pbp(y) for y in range(2014, 2026)], ignore_index=True)
    games = build_game_results(pbp)
    v5 = build_v5_game_predictions(games, build_v5_pregame_ratings(build_team_game_features(pbp)))
    qb = build_pregame_qb_ratings(pbp).sort_values(["team", "season", "week", "game_id"])
    qb["previous_qb_id"] = qb.groupby(["team", "season"])["qb_id"].shift(1)
    qb["previous_qb_epa"] = qb.groupby(["team", "season"])["pregame_qb_epa"].shift(1)
    qb["qb_changed"] = qb["previous_qb_id"].notna() & qb["qb_id"].ne(qb["previous_qb_id"])

    cols = ["game_id", "team", "qb_id", "previous_qb_id", "pregame_qb_epa", "previous_qb_epa", "qb_changed"]
    home = qb[cols].rename(columns={"team": "home_team", **{c: f"home_{c}" for c in cols if c not in {"game_id", "team"}}})
    away = qb[cols].rename(columns={"team": "away_team", **{c: f"away_{c}" for c in cols if c not in {"game_id", "team"}}})
    frame = v5.merge(home, on=["game_id", "home_team"], how="left").merge(away, on=["game_id", "away_team"], how="left")
    switches = frame[(frame["season"].between(2023, 2025)) & (frame["home_qb_changed"].fillna(False) | frame["away_qb_changed"].fillna(False))].copy()
    print(f"QB-change holdout games: {len(switches)}")
    print("V5 CONTROL", evaluate_margin_predictions(switches))

    for damping in [0.25, 0.50, 0.75, 1.00]:
        for cap in [2.0, 3.0, 4.0, 6.0]:
            adjusted = switches.copy()
            adjusted["home_change"] = [
                damping * qb_change_points(e, b, max_adjustment=cap / damping)
                if pd.notna(ch) and ch else 0.0
                for e, b, ch in zip(adjusted["home_pregame_qb_epa"], adjusted["home_previous_qb_epa"], adjusted["home_qb_changed"])
            ]
            adjusted["away_change"] = [
                damping * qb_change_points(e, b, max_adjustment=cap / damping)
                if pd.notna(ch) and ch else 0.0
                for e, b, ch in zip(adjusted["away_pregame_qb_epa"], adjusted["away_previous_qb_epa"], adjusted["away_qb_changed"])
            ]
            adjusted["predicted_home_margin"] = adjusted["predicted_home_margin"] + adjusted["home_change"] - adjusted["away_change"]
            metrics = evaluate_margin_predictions(adjusted)
            print(f"damping={damping:.2f} cap={cap:.1f}", metrics)


if __name__ == "__main__":
    run()
