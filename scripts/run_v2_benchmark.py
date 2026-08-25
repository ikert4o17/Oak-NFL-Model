"""Compare Oak V1 and V2 across historical seasons."""

from __future__ import annotations

import pandas as pd

from oak_nfl.backtest import evaluate_margin_predictions
from oak_nfl.data.nflverse import load_pbp
from oak_nfl.features import build_team_game_features
from oak_nfl.data.games import build_game_results
from oak_nfl.ratings.baseline import build_game_predictions, efficiency_rating_frame
from oak_nfl.ratings.v2 import build_v2_pregame_ratings


def run() -> None:
    # Load one extra season so 2015 has a genuine 2014 preseason prior.
    frames = [load_pbp(season) for season in range(2014, 2026)]
    pbp = pd.concat(frames, ignore_index=True)
    team_games = build_team_game_features(pbp)
    games = build_game_results(pbp)

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
        predictions = predictions[predictions["season"].between(2015, 2025)]
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
    print("=== OAK V2 PARAMETER COMPARISON ===")
    print(results.to_string(index=False))
    print("\nV1 CONTROL: MAE=10.5300 RMSE=13.4800 WINNER_ACCURACY=0.6206")


if __name__ == "__main__":
    run()
