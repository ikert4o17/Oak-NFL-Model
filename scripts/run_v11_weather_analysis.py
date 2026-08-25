"""Historical V11 weather/acclimation diagnostic.

This is intentionally a hard diagnostic rather than Oak's final totals model:
it asks whether weather and climate mismatch explain residual scoring beyond the
closing market total. If a feature survives here, it is especially interesting;
if it does not, that does not by itself prove weather is useless to an independent
Oak totals model because the closing market has already incorporated forecasts.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from oak_nfl.data.nflverse import load_schedules
from oak_nfl.weather import build_temperature_acclimation, build_weather_features

TRAIN_END = 2022
HOLDOUT_START = 2023
HOLDOUT_END = 2025


def _prepare_games() -> pd.DataFrame:
    games = load_schedules(list(range(2000, HOLDOUT_END + 1))).copy()
    games = games[games["game_type"].isin(["REG", "WC", "DIV", "CON", "SB"])].copy()
    games["home_score"] = pd.to_numeric(games["home_score"], errors="coerce")
    games["away_score"] = pd.to_numeric(games["away_score"], errors="coerce")
    games["total_line"] = pd.to_numeric(games["total_line"], errors="coerce")
    games["temperature_f"] = pd.to_numeric(games["temp"], errors="coerce")
    games["wind_mph"] = pd.to_numeric(games["wind"], errors="coerce")
    games["precipitation_in"] = 0.0
    roof = games["roof"].fillna("").astype(str).str.lower()
    games["is_dome"] = roof.isin(["dome", "closed"])
    games["actual_total"] = games["home_score"] + games["away_score"]

    weather = build_weather_features(
        games[["game_id", "temperature_f", "wind_mph", "precipitation_in", "is_dome"]]
    )
    acclimation = build_temperature_acclimation(
        games[
            [
                "game_id",
                "season",
                "week",
                "home_team",
                "away_team",
                "temperature_f",
                "is_dome",
            ]
        ]
    )
    return games.merge(weather, on="game_id", suffixes=("", "_weather")).merge(
        acclimation, on="game_id", how="left"
    )


def _metrics(actual: pd.Series, predicted: np.ndarray | pd.Series) -> dict[str, float]:
    return {
        "games": float(len(actual)),
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
    }


def _fit_residual_correction(
    train: pd.DataFrame,
    holdout: pd.DataFrame,
    features: list[str],
) -> tuple[np.ndarray, Ridge]:
    target = train["actual_total"] - train["total_line"]
    preprocessing = ColumnTransformer(
        [
            (
                "numeric",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                    ]
                ),
                features,
            )
        ],
        remainder="drop",
    )
    model = Pipeline([("prep", preprocessing), ("ridge", Ridge(alpha=10.0))])
    model.fit(train, target)
    correction = model.predict(holdout)
    return holdout["total_line"].to_numpy() + correction, model.named_steps["ridge"]


def _bucket_report(frame: pd.DataFrame) -> None:
    outdoor = frame[frame["outdoor"].eq(1) & frame["wind_mph"].notna()].copy()
    outdoor["wind_bucket"] = pd.cut(
        outdoor["wind_mph"],
        bins=[-0.1, 9.99, 14.99, 19.99, 100],
        labels=["<10", "10-14", "15-19", "20+"],
    )
    report = outdoor.groupby("wind_bucket", observed=True).agg(
        games=("game_id", "size"),
        avg_total=("actual_total", "mean"),
        avg_market_total=("total_line", "mean"),
        avg_market_residual=("actual_total", lambda s: float("nan")),
    )
    for bucket in report.index:
        rows = outdoor[outdoor["wind_bucket"].eq(bucket)].dropna(
            subset=["actual_total", "total_line"]
        )
        report.loc[bucket, "avg_market_residual"] = (
            rows["actual_total"] - rows["total_line"]
        ).mean()
    print("\n=== WIND BUCKET DIAGNOSTIC ===")
    print(report.to_string())


def run() -> None:
    games = _prepare_games().dropna(subset=["actual_total", "total_line"])
    train = games[games["season"].le(TRAIN_END)].copy()
    holdout = games[games["season"].between(HOLDOUT_START, HOLDOUT_END)].copy()

    print("=== OAK V11 WEATHER / ACCLIMATION DIAGNOSTIC ===")
    print(f"train games: {len(train)} | holdout games: {len(holdout)}")
    print("MARKET TOTAL CONTROL", _metrics(holdout["actual_total"], holdout["total_line"]))

    feature_sets = {
        "wind": ["wind_over_10", "wind_over_15"],
        "absolute_temperature": ["cold_below_32", "heat_above_85"],
        "team_acclimation": [
            "home_cold_shock",
            "away_cold_shock",
            "home_heat_shock",
            "away_heat_shock",
            "away_indoor_to_outdoor",
        ],
        "wind_plus_acclimation": [
            "wind_over_10",
            "wind_over_15",
            "home_cold_shock",
            "away_cold_shock",
            "home_heat_shock",
            "away_heat_shock",
            "away_indoor_to_outdoor",
        ],
        "all_weather": [
            "wind_over_10",
            "wind_over_15",
            "cold_below_32",
            "heat_above_85",
            "home_cold_shock",
            "away_cold_shock",
            "home_heat_shock",
            "away_heat_shock",
            "away_indoor_to_outdoor",
        ],
    }
    rows = []
    for name, features in feature_sets.items():
        prediction, _ = _fit_residual_correction(train, holdout, features)
        rows.append({"model": name, **_metrics(holdout["actual_total"], prediction)})
    print(pd.DataFrame(rows).sort_values(["mae", "rmse"]).to_string(index=False))

    _bucket_report(games)


if __name__ == "__main__":
    run()
