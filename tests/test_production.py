import numpy as np
import pandas as pd

from oak_nfl.production import build_weekly_game_card


def test_build_weekly_game_card_scores_spread_and_total_edges() -> None:
    frame = pd.DataFrame(
        [
            {
                "game_id": "g1",
                "season": 2026,
                "week": 1,
                "home_team": "KC",
                "away_team": "BUF",
                "predicted_home_margin": 4.0,
                "spread_line": -2.5,
                "predicted_total": 51.5,
                "total_line": 48.0,
            },
            {
                "game_id": "g2",
                "season": 2026,
                "week": 1,
                "home_team": "DAL",
                "away_team": "PHI",
                "predicted_home_margin": -1.0,
                "spread_line": 1.5,
                "predicted_total": 43.0,
                "total_line": 46.5,
            },
        ]
    )

    out = build_weekly_game_card(frame).set_index("game_id")

    assert out.loc["g1", "spread_edge"] == 1.5
    assert out.loc["g1", "spread_side"] == "KC"
    assert out.loc["g1", "total_edge"] == 3.5
    assert out.loc["g1", "total_side"] == "OVER"
    assert out.loc["g1", "projected_home_score"] == 27.75
    assert out.loc["g1", "projected_away_score"] == 23.75

    assert out.loc["g2", "spread_edge"] == 0.5
    assert out.loc["g2", "spread_side"] == "DAL"
    assert out.loc["g2", "total_edge"] == -3.5
    assert out.loc["g2", "total_side"] == "UNDER"


def test_build_weekly_game_card_preserves_projection_without_market_lines() -> None:
    frame = pd.DataFrame(
        [
            {
                "game_id": "g1",
                "home_team": "A",
                "away_team": "B",
                "predicted_home_margin": 2.0,
                "predicted_total": 44.0,
            }
        ]
    )

    out = build_weekly_game_card(frame).iloc[0]
    assert np.isnan(out["spread_edge"])
    assert np.isnan(out["total_edge"])
    assert out["spread_side"] == "NO LINE"
    assert out["total_side"] == "NO LINE"
