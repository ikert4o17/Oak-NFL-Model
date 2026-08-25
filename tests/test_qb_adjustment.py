import pandas as pd

from oak_nfl.qb_adjustment import add_qb_change_adjustments, qb_change_points


def test_qb_change_defaults_use_half_strength_and_two_point_cap():
    assert round(qb_change_points(0.10, 0.00), 4) == 0.9812
    assert qb_change_points(1.00, 0.00) == 2.0
    assert qb_change_points(-1.00, 0.00) == -2.0


def test_qb_change_adjustments_apply_home_minus_away():
    games = pd.DataFrame(
        {
            "predicted_home_margin": [1.0],
            "home_expected_qb_epa": [0.20],
            "home_baseline_qb_epa": [0.10],
            "away_expected_qb_epa": [0.00],
            "away_baseline_qb_epa": [0.10],
        }
    )
    adjusted = add_qb_change_adjustments(games)
    assert adjusted.loc[0, "home_qb_change_points"] > 0
    assert adjusted.loc[0, "away_qb_change_points"] < 0
    assert adjusted.loc[0, "predicted_home_margin"] > 1.0
