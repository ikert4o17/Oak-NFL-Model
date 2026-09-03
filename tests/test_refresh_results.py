import pandas as pd

from scripts.refresh_results import _finals_from_schedules, _load_frozen_predictions


def test_finals_from_schedules_uses_normalized_closing_lines():
    schedules = pd.DataFrame(
        [
            {
                "game_id": "g1",
                "home_score": 24,
                "away_score": 21,
                "spread_line": -3.5,
                "total_line": 46.5,
            },
            {
                "game_id": "g2",
                "home_score": None,
                "away_score": None,
                "spread_line": 2.5,
                "total_line": 43.0,
            },
        ]
    )

    finals = _finals_from_schedules(schedules)

    assert finals.game_id.tolist() == ["g1"]
    row = finals.iloc[0]
    assert row.closing_spread_line == -3.5
    assert row.closing_total_line == 46.5


def test_load_frozen_predictions_keeps_first_snapshot(tmp_path):
    first = pd.DataFrame([{"game_id": "g1", "season": 2026, "week": 1, "predicted_home_margin": 3.0}])
    duplicate = pd.DataFrame([{"game_id": "g1", "season": 2026, "week": 1, "predicted_home_margin": 9.0}])
    other = pd.DataFrame([{"game_id": "g2", "season": 2026, "week": 2, "predicted_home_margin": -1.0}])
    first.to_csv(tmp_path / "oak_2026_week_1.csv", index=False)
    duplicate.to_csv(tmp_path / "oak_2026_week_1_copy.csv", index=False)
    other.to_csv(tmp_path / "oak_2026_week_2.csv", index=False)

    loaded = _load_frozen_predictions(tmp_path)

    assert loaded.game_id.tolist() == ["g1", "g2"]
    assert loaded.loc[loaded.game_id.eq("g1"), "predicted_home_margin"].iloc[0] == 3.0
