import pandas as pd

from oak_nfl.survivor import future_value_by_team, margin_to_confidence, pick_two, rank_candidates


def test_confidence_increases_with_margin():
    assert margin_to_confidence(10) > margin_to_confidence(3) > margin_to_confidence(0)


def test_path_a_can_preserve_premium_future_team():
    card = pd.DataFrame([
        {"home_team": "AAA", "away_team": "BBB", "predicted_home_margin": 8.0},
        {"home_team": "CCC", "away_team": "DDD", "predicted_home_margin": 7.2},
    ])
    future = {"AAA": 0.95, "CCC": 0.52}
    ranked = rank_candidates(card, future, set())
    path_a = pick_two(ranked, "a")
    path_b = pick_two(ranked, "b")
    assert path_a[0].team == "CCC"
    assert path_b[0].team == "AAA"


def test_used_team_is_excluded():
    card = pd.DataFrame([{"home_team": "AAA", "away_team": "BBB", "predicted_home_margin": 10.0}])
    assert rank_candidates(card, {"AAA": 0.5}, {"AAA"}) == []


def test_future_value_uses_power_and_home_field():
    schedule = pd.DataFrame([
        {"week": 2, "home_team": "AAA", "away_team": "BBB", "location": "Home"},
        {"week": 3, "home_team": "BBB", "away_team": "AAA", "location": "Home"},
    ])
    values = future_value_by_team(schedule, {"AAA": 5.0, "BBB": -5.0}, current_week=1)
    assert values["AAA"] > values["BBB"]
