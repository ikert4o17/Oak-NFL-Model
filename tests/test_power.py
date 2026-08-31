import pandas as pd
import pytest

from oak_nfl.power import build_power_ratings
from oak_nfl.ratings.v5 import V5_EPA_COEF, V5_EXPLOSIVE_COEF, V5_SUCCESS_COEF


def test_power_rating_difference_matches_v5_neutral_strength_gap():
    snapshot = pd.DataFrame([
        {"team":"AAA","pregame_off_epa_per_play":0.12,"pregame_def_epa_per_play_allowed":-0.03,"pregame_off_success_rate":0.47,"pregame_def_success_rate_allowed":0.41,"pregame_off_explosive_rate":0.14,"pregame_def_explosive_rate_allowed":0.10},
        {"team":"BBB","pregame_off_epa_per_play":0.02,"pregame_def_epa_per_play_allowed":0.06,"pregame_off_success_rate":0.42,"pregame_def_success_rate_allowed":0.45,"pregame_off_explosive_rate":0.09,"pregame_def_explosive_rate_allowed":0.12},
    ])
    ratings = build_power_ratings(snapshot).set_index("team")
    expected = (
        V5_EPA_COEF * ((0.12 - -0.03) - (0.02 - 0.06))
        + V5_SUCCESS_COEF * ((0.47 - 0.41) - (0.42 - 0.45))
        + V5_EXPLOSIVE_COEF * ((0.14 - 0.10) - (0.09 - 0.12))
    )
    assert ratings.loc["AAA", "rating"] - ratings.loc["BBB", "rating"] == pytest.approx(expected)
    assert ratings["rating"].mean() == pytest.approx(0.0)
    assert ratings.loc["AAA", "rank"] == 1
