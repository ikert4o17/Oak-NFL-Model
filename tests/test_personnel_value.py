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


def test_prior_season_value_carries_into_week_one_with_discount():
    snaps = pd.DataFrame(
        {
            "game_id": ["old", "new"],
            "season": [2023, 2024],
            "week": [18, 1],
            "team": ["KC", "KC"],
            "player": ["Veteran Tackle", "Veteran Tackle"],
            "player_id": ["00-vet", "00-vet"],
            "position": ["LT", "LT"],
            "offense_pct": [0.80, 0.50],
            "defense_pct": [0.0, 0.0],
        }
    )
    values = build_pregame_player_values(snaps, prior_season_weight=0.75)
    week1 = values.query("season == 2024 and week == 1").iloc[0]
    assert abs(week1.player_value - 0.60) < 1e-9
    assert week1.value_source == "prior_season"


def test_player_history_survives_team_change():
    snaps = pd.DataFrame(
        {
            "game_id": ["old", "new"],
            "season": [2023, 2024],
            "week": [18, 1],
            "team": ["KC", "DEN"],
            "player": ["Veteran Edge", "Veteran Edge"],
            "player_id": ["00-edge", "00-edge"],
            "position": ["DE", "DE"],
            "offense_pct": [0.0, 0.0],
            "defense_pct": [0.90, 0.50],
        }
    )
    values = build_pregame_player_values(snaps, prior_season_weight=0.75)
    den = values.query("season == 2024 and team == 'DEN'").iloc[0]
    assert abs(den.player_value - 0.675) < 1e-9


def test_out_player_carries_latest_completed_snap_value_forward():
    snaps = pd.DataFrame(
        {
            "game_id": ["g1", "g2"],
            "season": [2024, 2024],
            "week": [1, 2],
            "team": ["KC", "KC"],
            "player": ["Edge One", "Edge One"],
            "player_id": ["00-edge", "00-edge"],
            "position": ["DE", "DE"],
            "offense_pct": [0.0, 0.0],
            "defense_pct": [0.80, 0.60],
        }
    )
    values = build_pregame_player_values(snaps)
    availability = pd.DataFrame(
        {
            "season": [2024],
            "week": [3],
            "team": ["KC"],
            "player_id": ["00-edge"],
            "player_name": ["Edge One"],
            "position_group": ["EDGE"],
            "status": ["out"],
        }
    )
    attached = attach_player_values(availability, values)
    assert attached.loc[0, "player_value"] > 0.0
    assert abs(attached.loc[0, "player_value"] - values.iloc[-1].postgame_player_value) < 1e-9


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
