import numpy as np
import pandas as pd

from oak_nfl import live_qb
from oak_nfl.data.depth_charts import expected_starting_qbs, normalize_depth_charts
from oak_nfl.qb_adjustment import qb_change_points


def test_current_depth_chart_schema_selects_latest_qb1():
    raw = pd.DataFrame({"dt": ["2026-09-01", "2026-09-01", "2026-09-03", "2026-09-03"], "team": ["SEA"] * 4, "player_name": ["Old Starter", "Old Backup", "New Starter", "New Backup"], "gsis_id": ["old1", "old2", "new1", "new2"], "pos_grp": ["QB"] * 4, "pos_abb": ["QB"] * 4, "pos_rank": [1, 2, 1, 2]})
    qbs = expected_starting_qbs(raw)
    assert len(qbs) == 1
    assert qbs.loc[0, "expected_qb_name"] == "New Starter"
    assert qbs.loc[0, "expected_qb_id"] == "new1"
    assert qbs.loc[0, "depth_rank"] == 1


def test_legacy_depth_chart_schema_still_normalizes():
    raw = pd.DataFrame({"club_code": ["NE"], "full_name": ["Example QB"], "gsis_id": ["qb1"], "position": ["QB"], "depth_team": [1]})
    out = normalize_depth_charts(raw)
    assert out.loc[0, "team"] == "NE"
    assert out.loc[0, "player_name"] == "Example QB"
    assert out.loc[0, "depth_rank"] == 1


def test_confirmed_qb_outs_promote_highest_remaining_depth_chart_qb():
    depth = pd.DataFrame({"dt": ["2026-09-12 11:36:06+00:00"] * 3, "team": ["ATL"] * 3, "player_name": ["Tua Tagovailoa", "Michael Penix Jr.", "Cooper Rush"], "gsis_id": ["tua", "penix", "rush"], "pos_abb": ["QB"] * 3, "pos_rank": [1, 2, 3]})
    injuries = pd.DataFrame({"season": [2026] * 3, "week": [1] * 3, "team": ["ATL"] * 3, "player_id": [pd.NA] * 3, "player_name": ["Tua Tagovailoa", "Michael Penix Jr.", "Cooper Rush"], "position_group": ["QB"] * 3, "status": ["out", "out", "questionable"], "report_date": pd.to_datetime(["2026-09-11T17:36:00Z", "2026-09-11T17:55:00Z", "2026-09-12T20:08:00Z"]), "source": ["espn"] * 3})
    qbs = live_qb.expected_starting_qbs_with_availability(depth, injuries)
    assert len(qbs) == 1
    assert qbs.loc[0, "expected_qb_name"] == "Cooper Rush"
    assert qbs.loc[0, "expected_qb_id"] == "rush"
    assert qbs.loc[0, "depth_rank"] == 3
    assert qbs.loc[0, "expected_qb_source"] == "nflverse-depth-chart+espn-out-filter"


def test_offseason_departure_invalidates_latest_team_qb_baseline(monkeypatch):
    monkeypatch.setattr(live_qb, "latest_team_qbs", lambda pbp: pd.DataFrame({"team": ["ATL", "LV"], "baseline_qb_id": ["cousins", "oconnell"], "baseline_qb_name": ["Kirk Cousins", "Aidan O'Connell"]}))
    depth = pd.DataFrame({"dt": ["2026-09-12"] * 4, "team": ["ATL", "ATL", "LV", "LV"], "player_name": ["Tua Tagovailoa", "Cooper Rush", "Kirk Cousins", "Fernando Mendoza"], "gsis_id": ["tua", "rush", "cousins", "mendoza"], "pos_abb": ["QB"] * 4, "pos_rank": [1, 2, 1, 2]})
    out = live_qb.roster_validated_team_qbs(pd.DataFrame(), depth)
    atl = out.loc[out["team"].eq("ATL")].iloc[0]
    lv = out.loc[out["team"].eq("LV")].iloc[0]
    assert pd.isna(atl["baseline_qb_id"])
    assert pd.isna(atl["baseline_qb_name"])
    assert pd.isna(lv["baseline_qb_id"])


def test_current_roster_qb_remains_valid_baseline(monkeypatch):
    monkeypatch.setattr(live_qb, "latest_team_qbs", lambda pbp: pd.DataFrame({"team": ["SEA"], "baseline_qb_id": ["darnold"], "baseline_qb_name": ["Sam Darnold"]}))
    depth = pd.DataFrame({"dt": ["2026-09-12"], "team": ["SEA"], "player_name": ["Sam Darnold"], "gsis_id": ["darnold"], "pos_abb": ["QB"], "pos_rank": [1]})
    out = live_qb.roster_validated_team_qbs(pd.DataFrame(), depth)
    assert out.loc[0, "baseline_qb_id"] == "darnold"
    assert out.loc[0, "baseline_qb_name"] == "Sam Darnold"


def test_live_qb_inputs_compare_expected_to_latest_team_starter(monkeypatch):
    monkeypatch.setattr(live_qb, "roster_validated_team_qbs", lambda pbp, depth: pd.DataFrame({"team": ["SEA", "NE"], "baseline_qb_id": ["sea_old", "ne_qb"], "baseline_qb_name": ["Seattle Old", "New England QB"]}))
    monkeypatch.setattr(live_qb, "current_qb_epa_ratings", lambda pbp: pd.DataFrame({"qb_id": ["sea_old", "sea_new", "ne_qb"], "qb_name": ["Seattle Old", "Seattle New", "New England QB"], "current_qb_epa": [0.15, -0.05, 0.10], "prior_qb_dropbacks": [500.0, 150.0, 400.0]}))
    depth = pd.DataFrame({"dt": ["2026-09-03", "2026-09-03"], "team": ["SEA", "NE"], "player_name": ["Seattle New", "New England QB"], "gsis_id": ["sea_new", "ne_qb"], "pos_abb": ["QB", "QB"], "pos_rank": [1, 1]})
    slate = pd.DataFrame({"game_id": ["g1"], "home_team": ["SEA"], "away_team": ["NE"]})
    out = live_qb.build_live_qb_inputs(pd.DataFrame(), slate, depth)
    assert out.loc[0, "home_expected_qb_name"] == "Seattle New"
    assert out.loc[0, "home_baseline_qb_name"] == "Seattle Old"
    assert out.loc[0, "home_expected_qb_epa"] == -0.05
    assert out.loc[0, "home_baseline_qb_epa"] == 0.15
    assert qb_change_points(out.loc[0, "home_expected_qb_epa"], out.loc[0, "home_baseline_qb_epa"]) < 0
    assert out.loc[0, "away_expected_qb_epa"] == out.loc[0, "away_baseline_qb_epa"]


def test_missing_depth_chart_context_fails_safe_to_zero(monkeypatch):
    monkeypatch.setattr(live_qb, "roster_validated_team_qbs", lambda pbp, depth: pd.DataFrame({"team": ["SEA"], "baseline_qb_id": ["sea_qb"], "baseline_qb_name": ["SEA QB"]}))
    monkeypatch.setattr(live_qb, "current_qb_epa_ratings", lambda pbp: pd.DataFrame({"qb_id": ["sea_qb"], "qb_name": ["SEA QB"], "current_qb_epa": [0.1], "prior_qb_dropbacks": [400.0]}))
    slate = pd.DataFrame({"game_id": ["g1"], "home_team": ["SEA"], "away_team": ["NE"]})
    out = live_qb.build_live_qb_inputs(pd.DataFrame(), slate, pd.DataFrame())
    assert np.isnan(out.loc[0, "home_expected_qb_epa"])
    assert qb_change_points(out.loc[0, "home_expected_qb_epa"], out.loc[0, "home_baseline_qb_epa"]) == 0.0
    assert out.loc[0, "home_qb_context_confidence"] == "missing"
