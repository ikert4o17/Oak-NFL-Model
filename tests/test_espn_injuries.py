from datetime import datetime, timezone

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
        fetched_at=datetime(2026, 9, 9, 20, 0, tzinfo=timezone.utc),
    )

    assert out["team"].tolist() == ["DAL", "DAL"]
    assert out["position_group"].tolist() == ["OT", "WR"]
    assert out["status"].tolist() == ["out", "questionable"]
    assert out["source"].tolist() == ["espn", "espn"]
    assert out["season"].tolist() == [2026, 2026]
    assert out["week"].tolist() == [1, 1]


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
