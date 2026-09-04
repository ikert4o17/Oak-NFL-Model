import json
from pathlib import Path

import pandas as pd


def test_week1_neutral_site_frozen_card_matches_audit_record() -> None:
    card = pd.read_csv("data/predictions/oak_2026_week_1.csv")
    row = card.loc[card["game_id"].eq("2026_01_SF_LA")].iloc[0]
    audit = json.loads(Path("data/context/corrections/oak_2026_week_1_neutral_site.json").read_text())

    assert row["location_x"] == "Neutral"
    assert row["stadium_x"] == "Melbourne Cricket Ground"
    assert row["predicted_home_margin"] == audit["after"]["predicted_home_margin"]
    assert row["spread_edge"] == audit["after"]["spread_edge"]
    assert row["spread_side"] == audit["after"]["spread_side"]
    assert row["projected_away_score"] == audit["after"]["projected_away_score"]
    assert row["projected_home_score"] == audit["after"]["projected_home_score"]
    assert row["predicted_total"] == audit["unchanged"]["predicted_total"]

    correction = audit["before"]["predicted_home_margin"] - audit["after"]["predicted_home_margin"]
    assert correction == audit["correction"]["removed_v5_home_intercept"]
