import pandas as pd
import pytest

from oak_nfl.backtest import evaluate_margin_predictions


def test_evaluate_margin_predictions_returns_core_metrics() -> None:
    predictions = pd.DataFrame(
        {
            "predicted_home_margin": [3.0, -4.0, 7.0],
            "actual_home_margin": [1.0, -10.0, -2.0],
        }
    )
    metrics = evaluate_margin_predictions(predictions)
    assert metrics["games"] == 3.0
    assert metrics["mae"] == pytest.approx((2 + 6 + 9) / 3)
    assert metrics["winner_accuracy"] == pytest.approx(2 / 3)


def test_evaluate_margin_predictions_rejects_empty_completed_sample() -> None:
    predictions = pd.DataFrame(
        {"predicted_home_margin": [None], "actual_home_margin": [None]}
    )
    with pytest.raises(ValueError):
        evaluate_margin_predictions(predictions)
