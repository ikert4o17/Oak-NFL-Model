"""Evaluation helpers for historical Oak NFL predictions."""

from __future__ import annotations

import numpy as np
import pandas as pd


def evaluate_margin_predictions(predictions: pd.DataFrame) -> dict[str, float]:
    """Return core baseline metrics for completed games.

    Required columns are ``predicted_home_margin`` and ``actual_home_margin``.
    Winner accuracy excludes ties and games whose predicted margin is exactly zero.
    """
    required = {"predicted_home_margin", "actual_home_margin"}
    missing = required.difference(predictions.columns)
    if missing:
        raise ValueError(f"predictions missing required columns: {sorted(missing)}")

    completed = predictions.dropna(subset=["predicted_home_margin", "actual_home_margin"]).copy()
    if completed.empty:
        raise ValueError("no completed predictions available for evaluation")

    error = completed["predicted_home_margin"] - completed["actual_home_margin"]
    mae = float(error.abs().mean())
    rmse = float(np.sqrt((error**2).mean()))

    decisive = completed[
        completed["actual_home_margin"].ne(0) & completed["predicted_home_margin"].ne(0)
    ]
    if decisive.empty:
        winner_accuracy = float("nan")
    else:
        winner_accuracy = float(
            (
                np.sign(decisive["predicted_home_margin"])
                == np.sign(decisive["actual_home_margin"])
            ).mean()
        )

    return {
        "games": float(len(completed)),
        "mae": mae,
        "rmse": rmse,
        "winner_accuracy": winner_accuracy,
    }
