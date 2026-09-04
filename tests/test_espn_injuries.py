from datetime import UTC, datetime

import pytest

from oak_nfl.data.espn_injuries import parse_espn_injuries


def test_parse_espn_injuries_normalizes_provider_payload():
    payload = {
        "season": {"year": 2026},
        "injuries": [
            {
                "team": {"abbreviation": "DAL"},
                "injuries": [
                    {
                        "athlete": {
                            "id": "123",
                            "fullName": "Player One",
                            "position": {"abbreviation": "LT"},
                        },
                        "status": "Out",
                        "date": "2026-09-09T18:00:00Z",
                    },
                    {
                        "athlete": {
                            "id": "456",
                            "displayName": "Player Two",
                            "position": {"abbreviation": "WR"},
                        },
                        "status": {"name": "Questionable"},
                    },
                ],
            }
        ],
    }
    out = parse_espn_injuries(
        payload,
        season=2026,
        week=1,
        fetched_at=datetime(2026, 9, 9, 20, 0, tzinfo=UTC),
    )

    assert out["team"].tolist() == ["DAL", "DAL"]
    assert out["position_group"].tolist() == ["OT", "WR"]
    assert out["status"].tolist() == ["out", "questionable"]
    assert out["source"].tolist() == ["espn", "espn"]
    assert out["season"].tolist() == [2026, 2026]
    assert out["week"].tolist() == [1, 1]


def test_parse_espn_injuries_accepts_flat_team_block_schema():
    payload = {
        "injuries": [
            {
                "abbreviation": "ARI",
                "displayName": "Arizona Cardinals",
                "injuries": [
                    {
                        "athlete": {
                            "id": "10",
                            "fullName": "Player Flat",
                            "position": {"abbreviation": "RB"},
                        },
                        "status": "Questionable",
                    }
                ],
            }
        ]
    }
    out = parse_espn_injuries(payload, season=2026, week=1)
    assert out["team"].tolist() == ["ARI"]


def test_parse_espn_injuries_accepts_nested_team_id_without_abbreviation():
    payload = {
        "injuries": [
            {
                "team": {"id": "6", "displayName": "Dallas Cowboys"},
                "injuries": [
                    {
                        "athlete": {
                            "id": "12",
                            "fullName": "Player Id",
                            "position": {"abbreviation": "G"},
                        },
                        "status": "Out",
                    }
                ],
            }
        ]
    }
    out = parse_espn_injuries(payload, season=2026, week=1)
    assert out["team"].tolist() == ["DAL"]


def test_parse_espn_injuries_accepts_team_name_without_id_or_abbreviation():
    payload = {
        "injuries": [
            {
                "displayName": "Baltimore Ravens",
                "injuries": [
                    {
                        "athlete": {
                            "id": "13",
                            "fullName": "Player Name",
                            "position": {"abbreviation": "LB"},
                        },
                        "status": "Questionable",
                    }
                ],
            }
        ]
    }
    out = parse_espn_injuries(payload, season=2026, week=1)
    assert out["team"].tolist() == ["BAL"]


def test_parse_espn_injuries_rejects_unmapped_team_blocks():
    payload = {
        "injuries": [
            {
                "displayName": "Unknown Club",
                "injuries": [
                    {
                        "athlete": {
                            "id": "11",
                            "fullName": "Player Unknown",
                            "position": {"abbreviation": "WR"},
                        },
                        "status": "Out",
                    }
                ],
            }
        ]
    }
    with pytest.raises(ValueError, match="recognized team identity"):
        parse_espn_injuries(payload, season=2026, week=1)


def test_parse_espn_injuries_keeps_unknown_status_unknown():
    payload = {
        "injuries": [
            {
                "team": {"abbreviation": "BUF"},
                "injuries": [
                    {
                        "athlete": {
                            "id": "9",
                            "fullName": "Player Three",
                            "position": {"abbreviation": "CB"},
                        },
                        "status": "Game-time decision",
                    }
                ],
            }
        ]
    }
    out = parse_espn_injuries(payload, season=2026, week=1)
    assert out.iloc[0]["status"] == "unknown"


def test_empty_espn_injury_response_is_safe():
    out = parse_espn_injuries({"injuries": []}, season=2026, week=1)
    assert out.empty
