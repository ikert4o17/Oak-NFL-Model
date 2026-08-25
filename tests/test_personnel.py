import pandas as pd
import pytest

from oak_nfl.personnel import (
    apply_personnel_adjustments,
    player_absence_points,
    team_personnel_adjustment,
)


def test_elite_out_tackle_costs_more_than_out_running_back():
    assert abs(player_absence_points("OT", 1.0, "out")) > abs(
        player_absence_points("RB", 1.0, "out")
    )


def test_qb_is_rejected_from_non_qb_layer():
    with pytest.raises(ValueError):
        player_absence_points("QB", 1.0, "out")


def test_team_adjustment_aggregates_and_caps():
    availability = pd.DataFrame(
        {
            "team": ["KC", "KC", "KC", "BUF"],
            "position_group": ["OT", "WR", "EDGE", "CB"],
            "player_value": [1.0, 1.0, 1.0, 0.5],
            "status": ["out", "out", "out", "questionable"],
        }
    )
    result = team_personnel_adjustment(availability, team_cap=1.5)
    kc = result[result["team"].eq("KC")].iloc[0]
    assert kc.personnel_points == -1.5
    assert kc.players_affected == 3


def test_apply_personnel_adjustments_moves_home_margin_by_net_value():
    games = pd.DataFrame(
        {"home_team": ["KC"], "away_team": ["BUF"], "predicted_home_margin": [2.0]}
    )
    adjustments = pd.DataFrame(
        {"team": ["KC", "BUF"], "personnel_points": [-1.0, -0.25]}
    )
    result = apply_personnel_adjustments(games, adjustments)
    assert result.loc[0, "personnel_net_points"] == -0.75
    assert result.loc[0, "predicted_home_margin"] == 1.25
