"""Test whether wind effects vary with pregame offensive style."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from oak_nfl.data.nflverse import load_pbp, load_schedules

TRAIN_START = 2014
TRAIN_END = 2022
HOLDOUT_START = 2023
HOLDOUT_END = 2025


def _metrics(actual: pd.Series, predicted: np.ndarray) -> dict[str, float]:
    return {
        "games": float(len(actual)),
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
    }


def _build_pregame_style() -> pd.DataFrame:
    pbp = pd.concat(
        [load_pbp(season) for season in range(TRAIN_START, HOLDOUT_END + 1)],
        ignore_index=True,
    )
    required = {"game_id", "season", "week", "posteam", "pass", "rush", "yards_gained"}
    missing = required.difference(pbp.columns)
    if missing:
        raise ValueError(f"play-by-play missing style columns: {sorted(missing)}")

    plays = pbp[pbp["posteam"].notna()].copy()
    plays = plays[plays["pass"].eq(1) | plays["rush"].eq(1)].copy()
    plays["pass_play"] = plays["pass"].eq(1).astype(int)
    plays["explosive_pass"] = (
        plays["pass"].eq(1) & pd.to_numeric(plays["yards_gained"], errors="coerce").ge(20)
    ).astype(int)

    team_games = (
        plays.groupby(["game_id", "season", "week", "posteam"], as_index=False)
        .agg(
            plays=("pass_play", "size"),
            pass_plays=("pass_play", "sum"),
            explosive_passes=("explosive_pass", "sum"),
        )
    )
    team_games["pass_rate"] = team_games["pass_plays"] / team_games["plays"]
    team_games["explosive_pass_rate"] = (
        team_games["explosive_passes"] / team_games["pass_plays"].replace(0, np.nan)
    )

    if "field_goal_attempt" in pbp.columns:
        fg = (
            pbp[pbp["posteam"].notna()]
            .assign(fg_attempt=pd.to_numeric(pbp["field_goal_attempt"], errors="coerce").fillna(0.0))
            .groupby(["game_id", "posteam"], as_index=False)["fg_attempt"]
            .sum()
        )
        team_games = team_games.merge(fg, on=["game_id", "posteam"], how="left")
    else:
        team_games["fg_attempt"] = np.nan

    team_games = team_games.sort_values(["posteam", "season", "week", "game_id"])
    for metric in ["pass_rate", "explosive_pass_rate", "fg_attempt"]:
        team_games[f"pregame_{metric}"] = team_games.groupby(["posteam", "season"])[metric].transform(
            lambda values: values.expanding().mean().shift(1)
        )
    return team_games[
        [
            "game_id",
            "posteam",
            "pregame_pass_rate",
            "pregame_explosive_pass_rate",
            "pregame_fg_attempt",
        ]
    ]


def _prepare_games() -> pd.DataFrame:
    games = load_schedules(list(range(TRAIN_START, HOLDOUT_END + 1))).copy()
    games = games[games["game_type"].isin(["REG", "WC", "DIV", "CON", "SB"])].copy()
    for column in ["home_score", "away_score", "total_line", "wind"]:
        games[column] = pd.to_numeric(games[column], errors="coerce")
    games["actual_total"] = games["home_score"] + games["away_score"]
    roof = games["roof"].fillna("").astype(str).str.lower()
    games["outdoor"] = ~roof.isin(["dome", "closed"])
    games["wind_over_10"] = np.where(
        games["outdoor"], np.maximum(games["wind"].fillna(0.0) - 10.0, 0.0), 0.0
    )
    games["wind_over_15"] = np.where(
        games["outdoor"], np.maximum(games["wind"].fillna(0.0) - 15.0, 0.0), 0.0
    )

    style = _build_pregame_style()
    home = style.rename(
        columns={
            "posteam": "home_team",
            "pregame_pass_rate": "home_pass_rate",
            "pregame_explosive_pass_rate": "home_explosive_pass_rate",
            "pregame_fg_attempt": "home_fg_attempt",
        }
    )
    away = style.rename(
        columns={
            "posteam": "away_team",
            "pregame_pass_rate": "away_pass_rate",
            "pregame_explosive_pass_rate": "away_explosive_pass_rate",
            "pregame_fg_attempt": "away_fg_attempt",
        }
    )
    games = games.merge(home, on=["game_id", "home_team"], how="left").merge(
        away, on=["game_id", "away_team"], how="left"
    )
    games["combined_pass_rate"] = games["home_pass_rate"] + games["away_pass_rate"]
    games["combined_explosive_pass_rate"] = (
        games["home_explosive_pass_rate"] + games["away_explosive_pass_rate"]
    )
    games["combined_fg_attempt"] = games["home_fg_attempt"] + games["away_fg_attempt"]
    for metric in ["combined_pass_rate", "combined_explosive_pass_rate", "combined_fg_attempt"]:
        games[f"wind15_x_{metric}"] = games["wind_over_15"] * games[metric]
    return games


def _fit(train: pd.DataFrame, holdout: pd.DataFrame, features: list[str]) -> np.ndarray:
    target = train["actual_total"] - train["total_line"]
    model = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("ridge", Ridge(alpha=10.0)),
        ]
    )
    model.fit(train[features], target)
    return holdout["total_line"].to_numpy() + model.predict(holdout[features])


def run() -> None:
    games = _prepare_games().dropna(subset=["actual_total", "total_line"])
    train = games[games["season"].le(TRAIN_END)].copy()
    holdout = games[games["season"].between(HOLDOUT_START, HOLDOUT_END)].copy()
    print("=== OAK V11 WIND x OFFENSIVE STYLE ===")
    print(f"train games: {len(train)} | holdout games: {len(holdout)}")
    print("MARKET CONTROL", _metrics(holdout["actual_total"], holdout["total_line"].to_numpy()))

    feature_sets = {
        "wind": ["wind_over_10", "wind_over_15"],
        "wind_x_pass_rate": ["wind_over_10", "wind_over_15", "wind15_x_combined_pass_rate"],
        "wind_x_explosive_pass": [
            "wind_over_10",
            "wind_over_15",
            "wind15_x_combined_explosive_pass_rate",
        ],
        "wind_x_kicking": ["wind_over_10", "wind_over_15", "wind15_x_combined_fg_attempt"],
        "wind_x_all_style": [
            "wind_over_10",
            "wind_over_15",
            "wind15_x_combined_pass_rate",
            "wind15_x_combined_explosive_pass_rate",
            "wind15_x_combined_fg_attempt",
        ],
    }
    rows = []
    for name, features in feature_sets.items():
        prediction = _fit(train, holdout, features)
        rows.append({"model": name, **_metrics(holdout["actual_total"], prediction)})
    print(pd.DataFrame(rows).sort_values(["mae", "rmse"]).to_string(index=False))


if __name__ == "__main__":
    run()
