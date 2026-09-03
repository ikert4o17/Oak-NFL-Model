import pandas as pd
import pytest

from oak_nfl.data.injuries import latest_weekly_status, normalize_injury_feed, normalize_status


def test_normalize_injury_feed_maps_provider_columns_and_positions():
    raw = pd.DataFrame(
        {
            "yr": [2024],
            "wk": [7],
            "club": ["kc"],
            "name": ["Example Player"],
            "pos": ["LT"],
            "designation": ["Limited Participation"],
            "date": ["2024-10-18"],
        }
    )
    normalized = normalize_injury_feed(
        raw,
        column_map={
            "yr": "season",
            "wk": "week",
            "club": "team",
            "name": "player_name",
            "pos": "position_group",
            "designation": "status",
            "date": "report_date",
        },
        source="test",
    )
    row = normalized.iloc[0]
    assert row.team == "KC"
    assert row.position_group == "OT"
    assert row.status == "limited"
    assert row.source == "test"


def test_unknown_status_never_becomes_active():
    assert normalize_status("game-time decision") == "unknown"
    assert normalize_status(None) == "unknown"


def test_latest_weekly_status_keeps_latest_report():
    frame = pd.DataFrame(
        {
            "season": [2024, 2024],
            "week": [7, 7],
            "team": ["KC", "KC"],
            "player_id": ["p1", "p1"],
            "player_name": ["Player", "Player"],
            "position_group": ["WR", "WR"],
            "status": ["questionable", "out"],
            "report_date": pd.to_datetime(["2024-10-17", "2024-10-18"]),
            "source": ["test", "test"],
        }
    )
    latest = latest_weekly_status(frame)
    assert len(latest) == 1
    assert latest.loc[0, "status"] == "out"


def test_missing_required_columns_fail_loudly():
    with pytest.raises(ValueError):
        normalize_injury_feed(pd.DataFrame({"season": [2024]}))
