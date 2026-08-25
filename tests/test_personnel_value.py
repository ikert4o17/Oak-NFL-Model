import pandas as pd

from oak_nfl.personnel_value import attach_player_values, build_pregame_player_values


def test_player_value_uses_only_prior_weeks():
    snaps = pd.DataFrame(
        {
            "game_id": ["g1", "g2"],
            "season": [2024, 2024],
            "week": [1, 2],
            "team": ["KC", "KC"],
            "player": ["Tackle One", "Tackle One"],
            "position": ["LT", "LT"],
            "offense_pct": [0.80, 0.20],
            "defense_pct": [0.0, 0.0],
        }
    )
    values = build_pregame_player_values(snaps)
    week1 = values[values["week"].eq(1)].iloc[0]
    week2 = values[values["week"].eq(2)].iloc[0]
    assert week1.player_value == 0.0
    assert week2.player_value == 0.80


def test_current_week_snap_share_cannot_change_its_own_value():
    base = pd.DataFrame(
        {
            "game_id": ["g1", "g2"],
            "season": [2024, 2024],
            "week": [1, 2],
            "team": ["KC", "KC"],
            "player": ["Receiver", "Receiver"],
            "position": ["WR", "WR"],
            "offense_pct": [0.70, 0.10],
            "defense_pct": [0.0, 0.0],
        }
    )
    changed = base.copy()
    changed.loc[changed["week"].eq(2), "offense_pct"] = 1.0
    original_value = build_pregame_player_values(base).query("week == 2").iloc[0].player_value
    changed_value = build_pregame_player_values(changed).query("week == 2").iloc[0].player_value
    assert original_value == changed_value == 0.70


def test_attach_values_defaults_unknown_player_to_zero():
    availability = pd.DataFrame(
        {
            "season": [2024],
            "week": [3],
            "team": ["KC"],
            "player_name": ["Unknown Player"],
            "position_group": ["WR"],
            "status": ["out"],
        }
    )
    values = pd.DataFrame(
        columns=["season", "week", "team", "player_name", "player_value"]
    )
    attached = attach_player_values(availability, values)
    assert attached.loc[0, "player_value"] == 0.0
