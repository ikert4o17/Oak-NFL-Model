import pandas as pd

from oak_nfl.results import grade_predictions, summarize_results


def test_grade_predictions_uses_frozen_lines():
    pred = pd.DataFrame(
        [
            {
                "game_id": "g1",
                "season": 2026,
                "week": 1,
                "home_team": "SEA",
                "away_team": "NE",
                "predicted_home_margin": 2.364,
                "predicted_total": 45.91,
                "spread_line": -3.5,
                "total_line": 44.5,
            }
        ]
    )
    finals = pd.DataFrame(
        [
            {
                "game_id": "g1",
                "home_score": 13,
                "away_score": 10,
                "closing_spread_line": -3.0,
                "closing_total_line": 44.5,
            }
        ]
    )
    out = grade_predictions(pred, finals).iloc[0]
    assert out.su_result == "W"
    assert out.ats_result == "W"
    assert out.ou_result == "L"
    assert out.closing_spread_line == -3.5
    assert out.closing_total_line == 44.5
    assert out.spread_line == -3.5


def test_grade_predictions_falls_back_to_schedule_lines():
    pred = pd.DataFrame(
        [
            {
                "game_id": "g1",
                "season": 2026,
                "week": 1,
                "home_team": "A",
                "away_team": "B",
                "predicted_home_margin": 4.0,
                "predicted_total": 48.0,
            }
        ]
    )
    finals = pd.DataFrame(
        [
            {
                "game_id": "g1",
                "home_score": 24,
                "away_score": 21,
                "closing_spread_line": -3.5,
                "closing_total_line": 46.5,
            }
        ]
    )
    out = grade_predictions(pred, finals).iloc[0]
    assert out.closing_spread_line == -3.5
    assert out.closing_total_line == 46.5


def test_grade_predictions_replaces_schedule_score_columns():
    pred = pd.DataFrame(
        [
            {
                "game_id": "g1",
                "season": 2026,
                "week": 1,
                "home_team": "A",
                "away_team": "B",
                "predicted_home_margin": 4.0,
                "predicted_total": 48.0,
                "home_score": None,
                "away_score": None,
            }
        ]
    )
    finals = pd.DataFrame(
        [
            {
                "game_id": "g1",
                "home_score": 27,
                "away_score": 20,
                "closing_spread_line": -3.5,
                "closing_total_line": 46.5,
            }
        ]
    )

    out = grade_predictions(pred, finals).iloc[0]

    assert out.home_score == 27
    assert out.away_score == 20
    assert out.final_home_margin == 7


def test_weekly_summary_tracks_pushes():
    graded = pd.DataFrame(
        [
            {
                "season": 2026,
                "week": 1,
                "su_result": "W",
                "ats_result": "P",
                "ou_result": "L",
            },
            {
                "season": 2026,
                "week": 1,
                "su_result": "L",
                "ats_result": "W",
                "ou_result": "W",
            },
        ]
    )
    row = summarize_results(graded).iloc[0]
    assert (row.su_w, row.su_l, row.su_p) == (1, 1, 0)
    assert (row.ats_w, row.ats_l, row.ats_p) == (1, 0, 1)
    assert (row.ou_w, row.ou_l, row.ou_p) == (1, 1, 0)
