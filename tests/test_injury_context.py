import pandas as pd
import pytest

from oak_nfl.injury_context import (
    build_game_injury_context,
    normalize_injury_report,
    normalize_status,
    team_injury_context,
)


def test_status_normalization_is_conservative():
    assert normalize_status("Out") == "out"
    assert normalize_status("Questionable") == "questionable"
    assert normalize_status("game-time decision") == "unknown"
    assert normalize_status(None) == "unknown"


def test_normalized_report_never_auto_adjusts_points():
    report = pd.DataFrame(
        [
            {"team": "buf", "player_name": "Player A", "position": "WR", "status": "Out"},
            {"team": "buf", "player_name": "Player B", "position": "CB", "status": "Questionable"},
        ]
    )
    out = normalize_injury_report(report)
    assert out["team"].tolist() == ["BUF", "BUF"]
    assert out["injury_status"].tolist() == ["out", "questionable"]
    assert not out["automatic_adjustment_allowed"].any()


def test_team_context_is_informational_and_zero_point():
    report = pd.DataFrame(
        [
            {"team": "KC", "player_name": "A", "position": "OT", "status": "Out"},
            {"team": "KC", "player_name": "B", "position": "WR", "status": "Doubtful"},
            {"team": "KC", "player_name": "C", "position": "S", "status": "Questionable"},
            {"team": "KC", "player_name": "D", "position": "TE", "status": "TBD"},
        ]
    )
    row = team_injury_context(report).iloc[0]
    assert row["injury_players_reported"] == 4
    assert row["injury_out_count"] == 1
    assert row["injury_doubtful_count"] == 1
    assert row["injury_questionable_count"] == 1
    assert row["injury_unknown_count"] == 1
    assert row["injury_auto_points"] == 0.0


def test_game_context_adds_both_teams_without_model_points():
    report = pd.DataFrame(
        [
            {"team": "KC", "player_name": "A", "position": "OT", "status": "Out"},
            {"team": "BUF", "player_name": "B", "position": "WR", "status": "Questionable"},
        ]
    )
    slate = pd.DataFrame(
        [{"game_id": "g1", "home_team": "KC", "away_team": "BUF"}]
    )
    row = build_game_injury_context(report, slate).iloc[0]
    assert row["home_injury_out_count"] == 1
    assert row["away_injury_questionable_count"] == 1
    assert row["home_injury_auto_points"] == 0.0
    assert row["away_injury_auto_points"] == 0.0


def test_missing_required_columns_raise():
    with pytest.raises(ValueError, match="injury report missing required columns"):
        normalize_injury_report(pd.DataFrame({"team": ["DAL"]}))
