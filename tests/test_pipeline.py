import pandas as pd

from oak_nfl.pipeline import run_baseline_backtest


def test_run_baseline_backtest_returns_predictions_and_metrics() -> None:
    rows = []
    for game_id, week, home, away, home_score, away_score, home_epa, away_epa in [
        ("g1", 1, "A", "B", 24, 17, 0.2, -0.1),
        ("g2", 2, "A", "B", 20, 14, 0.1, -0.2),
    ]:
        rows.extend(
            [
                {
                    "game_id": game_id,
                    "season": 2025,
                    "week": week,
                    "home_team": home,
                    "away_team": away,
                    "home_score": home_score,
                    "away_score": away_score,
                    "posteam": home,
                    "defteam": away,
                    "play_type": "pass",
                    "epa": home_epa,
                    "success": float(home_epa > 0),
                    "yards_gained": 10,
                    "pass": 1,
                    "rush": 0,
                    "qb_kneel": 0,
                    "qb_spike": 0,
                },
                {
                    "game_id": game_id,
                    "season": 2025,
                    "week": week,
                    "home_team": home,
                    "away_team": away,
                    "home_score": home_score,
                    "away_score": away_score,
                    "posteam": away,
                    "defteam": home,
                    "play_type": "pass",
                    "epa": away_epa,
                    "success": float(away_epa > 0),
                    "yards_gained": 8,
                    "pass": 1,
                    "rush": 0,
                    "qb_kneel": 0,
                    "qb_spike": 0,
                },
            ]
        )

    predictions, metrics = run_baseline_backtest(pd.DataFrame(rows))
    scored = predictions.dropna(subset=["predicted_home_margin"])
    assert len(scored) == 1
    assert metrics["games"] == 1.0
