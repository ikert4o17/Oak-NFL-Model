import numpy as np
import pandas as pd

from oak_nfl import live_qb
from oak_nfl.data.depth_charts import expected_starting_qbs, normalize_depth_charts
from oak_nfl.qb_adjustment import qb_change_points


def test_current_depth_chart_schema_selects_latest_qb1():
    raw = pd.DataFrame(
        {
            "dt": ["2026-09-01", "2026-09-01", "2026-09-03", "2026-09-03"],
            "team": ["SEA", "SEA", "SEA", "SEA"],
            "player_name": ["Old Starter", "Old Backup", "New Starter", "New Backup"],
            "gsis_id": ["old1", "old2", "new1", "new2"],
            "pos_grp": ["QB", "QB", "QB", "QB"],
            "pos_abb": ["QB", "QB", "QB", "QB"],
            "pos_rank": [1, 2, 1, 2],
        }
    )
    qbs = expected_starting_qbs(raw)
    assert len(qbs) == 1
    assert qbs.loc[0, "expected_qb_name"] == "New Starter"
    assert qbs.loc[0, "expected_qb_id"] == "new1"
    assert qbs.loc[0, "depth_rank"] == 1


def test_legacy_depth_chart_schema_still_normalizes():
    raw = pd.DataFrame(
        {
            "club_code": ["NE"],
            "full_name": ["Example QB"],
            "gsis_id": ["qb1"],
            "position": ["QB"],
            "depth_team": [1],
        }
    )
    out = normalize_depth_charts(raw)
    assert out.loc[0, "team"] == "NE"
    assert out.loc[0, "player_name"] == "Example QB"
    assert out.loc[0, "depth_rank"] == 1


def test_live_qb_inputs_compare_expected_to_latest_team_starter(monkeypatch):
    monkeypatch.setattr(
        live_qb,
        "latest_team_qbs",
        lambda pbp: pd.DataFrame(
            {
                "team": ["SEA", "NE"],
                "baseline_qb_id": ["sea_old", "ne_qb"],
                "baseline_qb_name": ["Seattle Old", "New England QB"],
            }
        ),
    )
    monkeypatch.setattr(
        live_qb,
        "current_qb_epa_ratings",
        lambda pbp: pd.DataFrame(
            {
                "qb_id": ["sea_old", "sea_new", "ne_qb"],
                "qb_name": ["Seattle Old", "Seattle New", "New England QB"],
                "current_qb_epa": [0.15, -0.05, 0.10],
                "prior_qb_dropbacks": [500.0, 150.0, 400.0],
            }
        ),
    )
    depth = pd.DataFrame(
        {
            "dt": ["2026-09-03", "2026-09-03"],
            "team": ["SEA", "NE"],
            "player_name": ["Seattle New", "New England QB"],
            "gsis_id": ["sea_new", "ne_qb"],
            "pos_abb": ["QB", "QB"],
            "pos_rank": [1, 1],
        }
    )
    slate = pd.DataFrame(
        {"game_id": ["g1"], "home_team": ["SEA"], "away_team": ["NE"]}
    )
    out = live_qb.build_live_qb_inputs(pd.DataFrame(), slate, depth)
    assert out.loc[0, "home_expected_qb_name"] == "Seattle New"
    assert out.loc[0, "home_baseline_qb_name"] == "Seattle Old"
    assert out.loc[0, "home_expected_qb_epa"] == -0.05
    assert out.loc[0, "home_baseline_qb_epa"] == 0.15
    assert qb_change_points(
        out.loc[0, "home_expected_qb_epa"], out.loc[0, "home_baseline_qb_epa"]
    ) < 0
    assert out.loc[0, "away_expected_qb_epa"] == out.loc[0, "away_baseline_qb_epa"]


def test_missing_depth_chart_context_fails_safe_to_zero(monkeypatch):
    monkeypatch.setattr(
        live_qb,
        "latest_team_qbs",
        lambda pbp: pd.DataFrame(
            {"team": ["SEA"], "baseline_qb_id": ["sea_qb"], "baseline_qb_name": ["SEA QB"]}
        ),
    )
    monkeypatch.setattr(
        live_qb,
        "current_qb_epa_ratings",
        lambda pbp: pd.DataFrame(
            {
                "qb_id": ["sea_qb"],
                "qb_name": ["SEA QB"],
                "current_qb_epa": [0.1],
                "prior_qb_dropbacks": [400.0],
            }
        ),
    )
    slate = pd.DataFrame(
        {"game_id": ["g1"], "home_team": ["SEA"], "away_team": ["NE"]}
    )
    out = live_qb.build_live_qb_inputs(pd.DataFrame(), slate, pd.DataFrame())
    assert np.isnan(out.loc[0, "home_expected_qb_epa"])
    assert qb_change_points(
        out.loc[0, "home_expected_qb_epa"], out.loc[0, "home_baseline_qb_epa"]
    ) == 0.0
    assert out.loc[0, "home_qb_context_confidence"] == "missing"
