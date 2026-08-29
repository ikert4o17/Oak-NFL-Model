"""Locked Oak V12 totals model for production prediction.

Specification: market total + Ridge residual model using PPA5, DPA3,
pregame EPA/play, pregame explosive rate, and rest differential.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from oak_nfl.features import build_team_game_features, build_team_weekly_ratings

V12_FEATURES = [
    "home_ppa5", "away_ppa5", "home_dpa3", "away_dpa3",
    "home_pregame_off_epa_per_play", "home_pregame_def_epa_per_play_allowed",
    "away_pregame_off_epa_per_play", "away_pregame_def_epa_per_play_allowed",
    "home_pregame_off_explosive_rate", "home_pregame_def_explosive_rate_allowed",
    "away_pregame_off_explosive_rate", "away_pregame_def_explosive_rate_allowed",
    "rest_diff",
]


def _game_date_col(frame: pd.DataFrame) -> str:
    for col in ("gameday", "game_date"):
        if col in frame.columns:
            return col
    raise ValueError("data missing gameday/game_date")


def _completed_games(pbp: pd.DataFrame) -> pd.DataFrame:
    cols = ["game_id", "season", "week", "home_team", "away_team", "home_score", "away_score", "total_line"]
    date_col = _game_date_col(pbp)
    base = pbp[[*cols, date_col]].drop_duplicates("game_id", keep="last").copy()
    plays = pbp[pbp["posteam"].notna()].copy()
    plays["yards_gained"] = pd.to_numeric(plays["yards_gained"], errors="coerce").fillna(0)
    touchdown = pd.to_numeric(plays.get("touchdown", 0), errors="coerce").fillna(0).eq(1)
    plays["off_td"] = (touchdown & plays["play_type"].isin(["pass", "run", "qb_kneel"])).astype(int)
    plays["fg_made"] = plays.get("field_goal_result", "").fillna("").astype(str).str.lower().eq("made").astype(int)
    team = plays.groupby(["game_id", "posteam"], as_index=False).agg(
        yards=("yards_gained", "sum"), off_td=("off_td", "sum"), fg_made=("fg_made", "sum")
    )
    team["off_points"] = 7 * team["off_td"] + 3 * team["fg_made"]
    home = team.rename(columns={"posteam": "home_team", "yards": "home_yards", "off_points": "home_off_points"})[
        ["game_id", "home_team", "home_yards", "home_off_points"]
    ]
    away = team.rename(columns={"posteam": "away_team", "yards": "away_yards", "off_points": "away_off_points"})[
        ["game_id", "away_team", "away_yards", "away_off_points"]
    ]
    games = base.merge(home, on=["game_id", "home_team"], how="left").merge(away, on=["game_id", "away_team"], how="left")
    games["actual_total"] = pd.to_numeric(games["home_score"], errors="coerce") + pd.to_numeric(games["away_score"], errors="coerce")
    return games


def _rolling_scoring(games: pd.DataFrame) -> pd.DataFrame:
    hist: defaultdict[str, list[tuple[float, float, float, float]]] = defaultdict(list)
    rows = []
    for row in games.sort_values(["season", "week", "game_id"]).itertuples(index=False):
        out = {"game_id": row.game_id}
        for side, team in (("home", row.home_team), ("away", row.away_team)):
            h = hist[str(team)]
            s5, s3 = h[-5:], h[-3:]
            off_yards = sum(x[1] for x in s5)
            def_yards = sum(x[3] for x in s3)
            out[f"{side}_ppa5"] = 100 * sum(x[0] for x in s5) / off_yards if off_yards else np.nan
            out[f"{side}_dpa3"] = 100 * sum(x[2] for x in s3) / def_yards if def_yards else np.nan
        rows.append(out)
        if pd.notna(row.actual_total) and pd.notna(row.home_yards) and pd.notna(row.away_yards):
            hist[str(row.home_team)].append((row.home_off_points, row.home_yards, row.away_off_points, row.away_yards))
            hist[str(row.away_team)].append((row.away_off_points, row.away_yards, row.home_off_points, row.home_yards))
    return pd.DataFrame(rows)


def _rest_features(games: pd.DataFrame) -> pd.DataFrame:
    date_col = _game_date_col(games)
    base = games[["game_id", "season", "home_team", "away_team", date_col]].copy()
    base["game_date"] = pd.to_datetime(base[date_col], errors="coerce")
    home = base[["game_id", "season", "game_date", "home_team"]].rename(columns={"home_team": "team"})
    away = base[["game_id", "season", "game_date", "away_team"]].rename(columns={"away_team": "team"})
    app = pd.concat([home, away], ignore_index=True).sort_values(["team", "season", "game_date", "game_id"])
    app["prev"] = app.groupby(["team", "season"])["game_date"].shift(1)
    app["rest"] = (app["game_date"] - app["prev"]).dt.days
    hr = app.rename(columns={"team": "home_team", "rest": "home_rest"})[["game_id", "home_team", "home_rest"]]
    ar = app.rename(columns={"team": "away_team", "rest": "away_rest"})[["game_id", "away_team", "away_rest"]]
    out = base.merge(hr, on=["game_id", "home_team"]).merge(ar, on=["game_id", "away_team"])
    out["rest_diff"] = out["home_rest"] - out["away_rest"]
    return out[["game_id", "rest_diff"]]


def build_v12_feature_frame(pbp: pd.DataFrame, slate: pd.DataFrame) -> pd.DataFrame:
    """Build locked V12 training rows plus pregame rows for an upcoming slate."""
    completed = _completed_games(pbp)
    slate_base = slate.copy()
    for col in ["home_score", "away_score", "home_yards", "away_yards", "home_off_points", "away_off_points", "actual_total"]:
        if col not in slate_base:
            slate_base[col] = np.nan
    date_col = _game_date_col(slate_base)
    if date_col not in completed.columns:
        completed[date_col] = completed[_game_date_col(completed)]
    keep = list(dict.fromkeys([*completed.columns, *slate_base.columns]))
    combined = pd.concat([completed.reindex(columns=keep), slate_base.reindex(columns=keep)], ignore_index=True)
    combined = combined.drop_duplicates("game_id", keep="last")
    combined = combined.merge(_rolling_scoring(combined), on="game_id", how="left")

    tg = build_team_game_features(pbp)
    dummy = []
    metric_cols = [c for c in tg.columns if c not in {"game_id", "season", "week", "posteam", "defteam"}]
    for row in slate.itertuples(index=False):
        for offense, defense in ((row.home_team, row.away_team), (row.away_team, row.home_team)):
            d = {"game_id": row.game_id, "season": row.season, "week": row.week, "posteam": offense, "defteam": defense}
            d.update({c: np.nan for c in metric_cols})
            dummy.append(d)
    ratings = build_team_weekly_ratings(pd.concat([tg, pd.DataFrame(dummy)], ignore_index=True))
    core = [c for c in ratings.columns if c.startswith("pregame_")]
    home = ratings.rename(columns={"team": "home_team", **{c: f"home_{c}" for c in core}})
    away = ratings.rename(columns={"team": "away_team", **{c: f"away_{c}" for c in core}})
    combined = combined.merge(home[["game_id", "home_team", *[f"home_{c}" for c in core]]], on=["game_id", "home_team"], how="left")
    combined = combined.merge(away[["game_id", "away_team", *[f"away_{c}" for c in core]]], on=["game_id", "away_team"], how="left")
    combined = combined.merge(_rest_features(combined), on="game_id", how="left")
    return combined


def predict_v12_totals(pbp: pd.DataFrame, slate: pd.DataFrame) -> pd.DataFrame:
    """Fit the frozen V12 residual model on completed history and predict slate totals."""
    frame = build_v12_feature_frame(pbp, slate)
    train = frame.dropna(subset=["actual_total", "total_line"]).copy()
    target = frame[frame["game_id"].isin(slate["game_id"])].copy()
    model = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("ridge", Ridge(alpha=10)),
    ])
    model.fit(train[V12_FEATURES], train["actual_total"] - train["total_line"])
    line = pd.to_numeric(target["total_line"], errors="coerce")
    target["predicted_total"] = line + model.predict(target[V12_FEATURES])
    return target
