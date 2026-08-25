"""Run Oak Baseline V1 across historical NFL seasons and print a compact scorecard."""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd

from oak_nfl.data.nflverse import load_pbp
from oak_nfl.pipeline import run_baseline_backtest


def run_seasons(seasons: Iterable[int]) -> pd.DataFrame:
    rows: list[dict[str, float | int]] = []
    all_predictions: list[pd.DataFrame] = []

    for season in seasons:
        print(f"Loading {season} play-by-play...")
        pbp = load_pbp(season)
        predictions, metrics = run_baseline_backtest(pbp)
        predictions = predictions.copy()
        predictions["season"] = season
        all_predictions.append(predictions)
        rows.append({"season": season, **metrics})
        print(f"{season}: {metrics}")

    scorecard = pd.DataFrame(rows)
    combined = pd.concat(all_predictions, ignore_index=True)
    _, combined_metrics = run_combined_metrics(combined)

    print("\n=== OAK BASELINE V1 SCORECARD ===")
    print(scorecard.to_string(index=False))
    print("\n=== COMBINED ===")
    for key, value in combined_metrics.items():
        print(f"{key}: {value:.4f}")

    return scorecard


def run_combined_metrics(predictions: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    from oak_nfl.backtest import evaluate_margin_predictions

    metrics = evaluate_margin_predictions(predictions)
    return predictions, metrics


if __name__ == "__main__":
    run_seasons(range(2015, 2026))
