"""Compare Oak V3 opponent adjustment against the promoted V2 control."""

from __future__ import annotations

import pandas as pd

from oak_nfl.backtest import evaluate_margin_predictions
from oak_nfl.data.games import build_game_results
from oak_nfl.data.nflverse import load_pbp
from oak_nfl.features import build_team_game_features
from oak_nfl.ratings.baseline import build_game_predictions, efficiency_rating_frame
from oak_nfl.ratings.v2 import build_v2_pregame_ratings
from oak_nfl.ratings.v3 import build_v3_pregame_ratings


def _eligible_game_ids(v2_predictions: pd.DataFrame) -> set[str]:
    eligible = v2_predictions[v2_predictions["season"].between(2015, 2025)]
    return set(eligible["game_id"].astype(str))


def run() -> None:
    frames = [load_pbp(season) for season in range(2014, 2026)]
    pbp = pd.concat(frames, ignore_index=True)
    team_games = build_team_game_features(pbp)
    games = build_game_results(pbp)

    v2_pregame = build_v2_pregame_ratings(
        team_games,
        prior_games=4.0,
        prior_regression=0.50,
        recency_decay=0.85,
    )
    v2_ratings = efficiency_rating_frame(v2_pregame)
    v2_predictions = build_game_predictions(games, v2_ratings)
    eligible_ids = _eligible_game_ids(v2_predictions)
    v2_predictions = v2_predictions[v2_predictions["game_id"].astype(str).isin(eligible_ids)]
    v2_metrics = evaluate_margin_predictions(v2_predictions)

    rows = []
    for opponent_weight in (0.0, 0.25, 0.50, 0.75, 1.00, 1.25):
        pregame = build_v3_pregame_ratings(
            team_games,
            prior_games=4.0,
            prior_regression=0.50,
            recency_decay=0.85,
            opponent_weight=opponent_weight,
        )
        ratings = efficiency_rating_frame(pregame)
        predictions = build_game_predictions(games, ratings)
        predictions = predictions[predictions["game_id"].astype(str).isin(eligible_ids)]
        metrics = evaluate_margin_predictions(predictions)
        rows.append({"opponent_weight": opponent_weight, **metrics})

    results = pd.DataFrame(rows).sort_values("mae")
    print("=== OAK V3 OPPONENT-ADJUSTMENT COMPARISON ===")
    print(results.to_string(index=False))
    print("\n=== PROMOTED V2 CONTROL ===")
    for key, value in v2_metrics.items():
        print(f"{key}: {value:.4f}")


if __name__ == "__main__":
    run()
