"""Compare Oak V1 and V2 across the exact same historical game sample."""

from __future__ import annotations

import pandas as pd

from oak_nfl.backtest import evaluate_margin_predictions
from oak_nfl.data.games import build_game_results
from oak_nfl.data.nflverse import load_pbp
from oak_nfl.features import build_team_game_features, build_team_weekly_ratings
from oak_nfl.ratings.baseline import build_game_predictions, efficiency_rating_frame
from oak_nfl.ratings.v2 import build_v2_pregame_ratings


def run() -> None:
    # Load one extra season so 2015 has a genuine 2014 preseason prior.
    frames = [load_pbp(season) for season in range(2014, 2026)]
    pbp = pd.concat(frames, ignore_index=True)
    team_games = build_team_game_features(pbp)
    games = build_game_results(pbp)

    # Reconstruct V1 from the exact same source frame, then freeze its eligible
    # 2015-2025 game IDs. V2 is scored on this identical sample for fair comparison.
    v1_pregame = build_team_weekly_ratings(team_games)
    v1_ratings = efficiency_rating_frame(v1_pregame)
    v1_predictions = build_game_predictions(games, v1_ratings)
    v1_predictions = v1_predictions[v1_predictions["season"].between(2015, 2025)]
    v1_metrics = evaluate_margin_predictions(v1_predictions)
    eligible_game_ids = set(v1_predictions.dropna(subset=["predicted_home_margin"])["game_id"])

    parameter_grid = [
        (2.0, 0.35, 0.75),
        (2.0, 0.50, 0.85),
        (4.0, 0.35, 0.85),
        (4.0, 0.50, 0.85),
        (6.0, 0.50, 0.90),
        (8.0, 0.50, 0.90),
    ]
    rows = []
    for prior_games, regression, decay in parameter_grid:
        pregame = build_v2_pregame_ratings(
            team_games,
            prior_games=prior_games,
            prior_regression=regression,
            recency_decay=decay,
        )
        ratings = efficiency_rating_frame(pregame)
        predictions = build_game_predictions(games, ratings)
        predictions = predictions[
            predictions["season"].between(2015, 2025)
            & predictions["game_id"].isin(eligible_game_ids)
        ]
        metrics = evaluate_margin_predictions(predictions)
        rows.append(
            {
                "prior_games": prior_games,
                "prior_regression": regression,
                "recency_decay": decay,
                **metrics,
            }
        )

    results = pd.DataFrame(rows).sort_values("mae")
    print("=== OAK V2 APPLES-TO-APPLES PARAMETER COMPARISON ===")
    print(results.to_string(index=False))
    print("\n=== RECONSTRUCTED V1 CONTROL ===")
    for key, value in v1_metrics.items():
        print(f"{key}: {value:.4f}")


if __name__ == "__main__":
    run()
