import pandas as pd

from oak_nfl import weekly


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


def test_weekly_card_handles_jax_week_one_market_line_correctly(monkeypatch) -> None:
    slate = pd.DataFrame(
        [
            {
                "game_id": "2026_01_CLE_JAX",
                "season": 2026,
                "week": 1,
                "home_team": "JAX",
                "away_team": "CLE",
                "spread_line": -7.5,
                "total_line": 40.5,
                "gameday": "2026-09-13",
            }
        ]
    )

    monkeypatch.setattr(
        weekly,
        "_spread_core",
        lambda pbp, games: pd.DataFrame(
            [{"game_id": "2026_01_CLE_JAX", "predicted_home_margin": 6.5}]
        ),
    )
    monkeypatch.setattr(
        weekly,
        "predict_v12_totals",
        lambda pbp, games: pd.DataFrame(
            [{"game_id": "2026_01_CLE_JAX", "predicted_total": 42.2}]
        ),
    )

    out = weekly.run_weekly_predictions(pd.DataFrame(), slate).iloc[0]
    assert out["spread_edge"] == -1.0
    assert out["spread_side"] == "CLE"


def test_neutral_site_removes_v5_home_intercept(monkeypatch) -> None:
    slate = pd.DataFrame(
        [
            {
                "game_id": "2026_01_SF_LA",
                "season": 2026,
                "week": 1,
                "home_team": "LA",
                "away_team": "SF",
                "spread_line": -3.5,
                "total_line": 48.5,
                "gameday": "2026-09-10",
                "location": "Neutral",
            }
        ]
    )

    monkeypatch.setattr(
        weekly,
        "_spread_core",
        lambda pbp, games: pd.DataFrame(
            [{"game_id": "2026_01_SF_LA", "predicted_home_margin": 6.281628156341251}]
        ),
    )
    monkeypatch.setattr(
        weekly,
        "predict_v12_totals",
        lambda pbp, games: pd.DataFrame(
            [{"game_id": "2026_01_SF_LA", "predicted_total": 48.775633461297176}]
        ),
    )

    out = weekly.run_weekly_predictions(pd.DataFrame(), slate).iloc[0]
    expected_margin = 6.281628156341251 - weekly.V5_INTERCEPT
    assert abs(out["predicted_home_margin"] - expected_margin) < 1e-12
    assert abs(out["spread_edge"] - (expected_margin - 3.5)) < 1e-12
