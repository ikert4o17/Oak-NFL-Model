"""Benchmark quarterback value as an incremental feature beyond frozen Oak V5."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from oak_nfl.backtest import evaluate_margin_predictions
from oak_nfl.data.games import build_game_results
from oak_nfl.data.nflverse import load_pbp
from oak_nfl.features import build_team_game_features
from oak_nfl.qb import build_pregame_qb_ratings
from oak_nfl.ratings.v5 import build_v5_game_predictions, build_v5_pregame_ratings


def _attach_qb(games: pd.DataFrame, qb: pd.DataFrame) -> pd.DataFrame:
    cols = ["pregame_qb_epa", "pregame_qb_cpoe", "pregame_qb_sack_rate", "prior_qb_dropbacks"]
    home = qb[["game_id", "team", "qb_id", "qb_name", *cols]].rename(
        columns={"team": "home_team", "qb_id": "home_qb_id", "qb_name": "home_qb_name", **{c: f"home_{c}" for c in cols}}
    )
    away = qb[["game_id", "team", "qb_id", "qb_name", *cols]].rename(
        columns={"team": "away_team", "qb_id": "away_qb_id", "qb_name": "away_qb_name", **{c: f"away_{c}" for c in cols}}
    )
    frame = games.merge(home, on=["game_id", "home_team"], how="left", validate="one_to_one")
    frame = frame.merge(away, on=["game_id", "away_team"], how="left", validate="one_to_one")
    for metric in ["pregame_qb_epa", "pregame_qb_cpoe", "pregame_qb_sack_rate"]:
        frame[f"{metric}_gap"] = frame[f"home_{metric}"] - frame[f"away_{metric}"]
    return frame


def run() -> None:
    pbp = pd.concat([load_pbp(y) for y in range(2014, 2026)], ignore_index=True)
    team_games = build_team_game_features(pbp)
    games = build_game_results(pbp)
    v5 = build_v5_game_predictions(games, build_v5_pregame_ratings(team_games))
    qb = build_pregame_qb_ratings(pbp)
    frame = _attach_qb(games, qb).merge(
        v5[["game_id", "predicted_home_margin"]], on="game_id", how="left"
    )

    feature_sets = {
        "qb_epa": ["predicted_home_margin", "pregame_qb_epa_gap"],
        "qb_epa_cpoe": ["predicted_home_margin", "pregame_qb_epa_gap", "pregame_qb_cpoe_gap"],
        "qb_full": [
            "predicted_home_margin", "pregame_qb_epa_gap", "pregame_qb_cpoe_gap",
            "pregame_qb_sack_rate_gap",
        ],
    }
    print("=== OAK V8 QB VALUE TOURNAMENT ===")
    holdout_ids = set(games.loc[games["season"].between(2023, 2025), "game_id"])
    control = v5[v5["game_id"].isin(holdout_ids)].dropna(subset=["predicted_home_margin", "actual_home_margin"])
    print("V5 CONTROL", evaluate_margin_predictions(control))

    for name, features in feature_sets.items():
        usable = frame.dropna(subset=[*features, "actual_home_margin"]).copy()
        train = usable[usable["season"].between(2015, 2022)]
        holdout = usable[usable["season"].between(2023, 2025)].copy()
        if train.empty or holdout.empty:
            print(name, "SKIPPED insufficient data")
            continue
        rows = []
        for alpha in [0.0, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]:
            model = Ridge(alpha=alpha).fit(train[features], train["actual_home_margin"])
            pred = model.predict(train[features])
            rows.append((alpha, float(np.mean(np.abs(pred - train["actual_home_margin"])))))
        alpha = min(rows, key=lambda row: row[1])[0]
        model = Ridge(alpha=alpha).fit(train[features], train["actual_home_margin"])
        holdout["predicted_home_margin"] = model.predict(holdout[features])
        print(name, evaluate_margin_predictions(holdout), "alpha", alpha)
        print("coefficients", dict(zip(features, np.round(model.coef_, 4))))


if __name__ == "__main__":
    run()
