import pandas as pd

import oak_nfl.weekly as weekly


def test_run_weekly_predictions_combines_spread_and_total_models(monkeypatch) -> None:
    slate = pd.DataFrame(
        [
            {
                "game_id": "g1",
                "season": 2026,
                "week": 1,
                "home_team": "KC",
                "away_team": "BUF",
                "spread_line": -2.5,
                "total_line": 48.0,
                "gameday": "2026-09-10",
            }
        ]
    )

    monkeypatch.setattr(
        weekly,
        "_spread_core",
        lambda pbp, games: pd.DataFrame(
            [{"game_id": "g1", "predicted_home_margin": 4.0}]
        ),
    )
    monkeypatch.setattr(
        weekly,
        "predict_v12_totals",
        lambda pbp, games: pd.DataFrame(
            [{"game_id": "g1", "predicted_total": 51.5}]
        ),
    )

    out = weekly.run_weekly_predictions(pd.DataFrame(), slate).iloc[0]
    assert out["spread_edge"] == 1.5
    assert out["spread_side"] == "KC"
    assert out["total_edge"] == 3.5
    assert out["total_side"] == "OVER"
