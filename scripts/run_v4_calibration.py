"""Calibrate Oak's EPA-to-points scale and home-field value on past seasons."""

from __future__ import annotations

import pandas as pd

from oak_nfl.backtest import evaluate_margin_predictions
from oak_nfl.data.games import build_game_results
from oak_nfl.data.nflverse import load_pbp
from oak_nfl.features import build_team_game_features
from oak_nfl.ratings.baseline import build_game_predictions, efficiency_rating_frame
from oak_nfl.ratings.v2 import build_v2_pregame_ratings


def run() -> None:
    frames = [load_pbp(season) for season in range(2014, 2026)]
    pbp = pd.concat(frames, ignore_index=True)
    team_games = build_team_game_features(pbp)
    games = build_game_results(pbp)

    pregame = build_v2_pregame_ratings(
        team_games,
        prior_games=4.0,
        prior_regression=0.50,
        recency_decay=0.85,
    )
    ratings = efficiency_rating_frame(pregame)

    rows = []
    for epa_to_points in (14.0, 16.0, 18.0, 20.0, 22.0, 24.0, 26.0):
        for home_field_points in (0.5, 1.0, 1.5, 2.0, 2.5, 3.0):
            predictions = build_game_predictions(
                games,
                ratings,
                home_field_points=home_field_points,
                epa_to_points=epa_to_points,
            )
            train = predictions[predictions["season"].between(2015, 2022)]
            metrics = evaluate_margin_predictions(train)
            rows.append(
                {
                    "epa_to_points": epa_to_points,
                    "home_field_points": home_field_points,
                    **metrics,
                }
            )

    results = pd.DataFrame(rows).sort_values(["mae", "rmse"]).reset_index(drop=True)
    best = results.iloc[0]
    best_epa = float(best["epa_to_points"])
    best_hfa = float(best["home_field_points"])

    control = build_game_predictions(games, ratings, home_field_points=1.5, epa_to_points=20.0)
    challenger = build_game_predictions(
        games,
        ratings,
        home_field_points=best_hfa,
        epa_to_points=best_epa,
    )

    control_holdout = control[control["season"].between(2023, 2025)]
    challenger_holdout = challenger[challenger["season"].between(2023, 2025)]
    control_metrics = evaluate_margin_predictions(control_holdout)
    challenger_metrics = evaluate_margin_predictions(challenger_holdout)

    print("=== OAK V4 TRAINING CALIBRATION: 2015-2022 ===")
    print(results.head(12).to_string(index=False))
    print(f"\nSELECTED: EPA_TO_POINTS={best_epa:.2f} HOME_FIELD_POINTS={best_hfa:.2f}")

    print("\n=== HOLDOUT: 2023-2025 ===")
    print("V2 CONTROL")
    for key, value in control_metrics.items():
        print(f"{key}: {value:.4f}")
    print("V4 CALIBRATED")
    for key, value in challenger_metrics.items():
        print(f"{key}: {value:.4f}")


if __name__ == "__main__":
    run()
