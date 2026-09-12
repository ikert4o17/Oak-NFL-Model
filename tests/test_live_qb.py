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


def test_embedded_baseline_does_not_flip_to_backup_after_one_split_game(monkeypatch):
    games = pd.DataFrame(
        {
            "game_id": ["g1", "g2", "g2"],
            "season": [2026, 2026, 2026],
            "week": [1, 2, 2],
            "posteam": ["SEA", "SEA", "SEA"],
            "passer_player_id": ["darnold", "darnold", "lock"],
            "passer_player_name": ["Sam Darnold", "Sam Darnold", "Drew Lock"],
            "qb_dropbacks": [30, 2, 28],
            "qb_epa_per_dropback": [0.20, 0.20, -0.10],
            "qb_cpoe": [0.0, 0.0, 0.0],
            "qb_sack_rate": [0.0, 0.0, 0.0],
        }
    )
    monkeypatch.setattr(live_qb, "build_qb_game_efficiency", lambda pbp: games)

    out = live_qb.embedded_team_qb_values(pd.DataFrame(), prior_dropbacks=0.0, recency_decay=0.92)
    sea = out.loc[out["team"].eq("SEA")].iloc[0]

    # Lock handled almost all of the latest game, but the embedded team baseline
    # still retains the prior Darnold-heavy game instead of becoming Lock outright.
    assert sea["baseline_qb_name"] == "Embedded team QB"
    assert sea["baseline_qb_epa"] > 0.0
    assert sea["baseline_qb_source"] == "recency-dropback-weighted-team-qb"


def test_offseason_departure_remains_only_as_part_of_team_embedded_value(monkeypatch):
    games = pd.DataFrame(
        {
            "game_id": ["g1", "g2"],
            "season": [2025, 2025],
            "week": [17, 18],
            "posteam": ["ATL", "ATL"],
            "passer_player_id": ["cousins", "cousins"],
            "passer_player_name": ["Kirk Cousins", "Kirk Cousins"],
            "qb_dropbacks": [35, 35],
            "qb_epa_per_dropback": [0.08, 0.02],
            "qb_cpoe": [0.0, 0.0],
            "qb_sack_rate": [0.0, 0.0],
        }
    )
    monkeypatch.setattr(live_qb, "build_qb_game_efficiency", lambda pbp: games)

    out = live_qb.embedded_team_qb_values(pd.DataFrame(), prior_dropbacks=0.0)
    atl = out.loc[out["team"].eq("ATL")].iloc[0]

    assert atl["baseline_qb_name"] == "Embedded team QB"
    assert pd.isna(atl["baseline_qb_id"])
    assert atl["baseline_qb_epa"] > 0.0


def test_live_qb_inputs_compare_expected_to_embedded_team_value(monkeypatch):
    monkeypatch.setattr(
        live_qb,
        "embedded_team_qb_values",
        lambda pbp: pd.DataFrame(
            {
                "team": ["SEA", "NE"],
                "baseline_qb_epa": [0.05, 0.10],
                "baseline_qb_name": ["Embedded team QB", "Embedded team QB"],
                "baseline_qb_id": [pd.NA, pd.NA],
                "baseline_qb_source": ["recency-dropback-weighted-team-qb"] * 2,
            }
        ),
    )
    monkeypatch.setattr(
        live_qb,
        "current_qb_epa_ratings",
        lambda pbp: pd.DataFrame(
            {
                "qb_id": ["lock", "ne_qb"],
                "qb_name": ["Drew Lock", "New England QB"],
                "current_qb_epa": [-0.05, 0.10],
                "prior_qb_dropbacks": [150.0, 400.0],
            }
        ),
    )
    depth = pd.DataFrame(
        {
            "dt": ["2026-09-03", "2026-09-03"],
            "team": ["SEA", "NE"],
            "player_name": ["Drew Lock", "New England QB"],
            "gsis_id": ["lock", "ne_qb"],
            "pos_abb": ["QB", "QB"],
            "pos_rank": [1, 1],
        }
    )
    slate = pd.DataFrame({"game_id": ["g1"], "home_team": ["SEA"], "away_team": ["NE"]})
    out = live_qb.build_live_qb_inputs(pd.DataFrame(), slate, depth)

    assert out.loc[0, "home_expected_qb_name"] == "Drew Lock"
    assert out.loc[0, "home_baseline_qb_name"] == "Embedded team QB"
    assert out.loc[0, "home_expected_qb_epa"] == -0.05
    assert out.loc[0, "home_baseline_qb_epa"] == 0.05
    assert qb_change_points(out.loc[0, "home_expected_qb_epa"], out.loc[0, "home_baseline_qb_epa"]) < 0
    assert out.loc[0, "home_baseline_qb_source"] == "recency-dropback-weighted-team-qb"
    assert out.loc[0, "away_expected_qb_epa"] == out.loc[0, "away_baseline_qb_epa"]


def test_missing_depth_chart_context_fails_safe_to_zero(monkeypatch):
    monkeypatch.setattr(
        live_qb,
        "embedded_team_qb_values",
        lambda pbp: pd.DataFrame(
            {
                "team": ["SEA"],
                "baseline_qb_epa": [0.1],
                "baseline_qb_name": ["Embedded team QB"],
                "baseline_qb_id": [pd.NA],
                "baseline_qb_source": ["recency-dropback-weighted-team-qb"],
            }
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
    slate = pd.DataFrame({"game_id": ["g1"], "home_team": ["SEA"], "away_team": ["NE"]})
    out = live_qb.build_live_qb_inputs(pd.DataFrame(), slate, pd.DataFrame())
    assert np.isnan(out.loc[0, "home_expected_qb_epa"])
    assert qb_change_points(out.loc[0, "home_expected_qb_epa"], out.loc[0, "home_baseline_qb_epa"]) == 0.0
    assert out.loc[0, "home_qb_context_confidence"] == "missing"
