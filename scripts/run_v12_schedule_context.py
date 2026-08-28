"""Schedule / rest / travel context experiment against locked V12.

Predetermined challengers:
- rest differential
- short-week flags (<= 6 days since prior game)
- extra-rest flags (>= 9 days since prior game)
- away-team travel distance between franchise home bases
- time-zone shift between franchise home bases
- international/neutral-site context when identifiable from PBP metadata
- combined schedule-context package

All inputs are known before kickoff. Rest is computed only from prior games in the
same season. Locked V12 remains unchanged and is the control.
"""
from __future__ import annotations

from math import asin, cos, radians, sin, sqrt

import numpy as np
import pandas as pd

from oak_nfl.data.nflverse import load_pbp
from oak_nfl.features import build_team_game_features, build_team_weekly_ratings
from run_v12_core_shootout import cols
from run_v12_incremental import HOLDOUT_END, START, metrics, model, prepare

TEST_SEASONS = range(2019, 2026)

# Approximate franchise home coordinates and standard UTC offsets. Historical
# aliases are included so the experiment works cleanly across 2014-2025.
TEAM_META = {
    "ARI": (33.45, -112.07, -7), "ATL": (33.75, -84.39, -5),
    "BAL": (39.29, -76.61, -5), "BUF": (42.89, -78.88, -5),
    "CAR": (35.23, -80.84, -5), "CHI": (41.88, -87.63, -6),
    "CIN": (39.10, -84.51, -5), "CLE": (41.50, -81.69, -5),
    "DAL": (32.78, -96.80, -6), "DEN": (39.74, -104.99, -7),
    "DET": (42.33, -83.05, -5), "GB": (44.51, -88.02, -6),
    "HOU": (29.76, -95.37, -6), "IND": (39.77, -86.16, -5),
    "JAX": (30.33, -81.66, -5), "KC": (39.10, -94.58, -6),
    "LV": (36.17, -115.14, -8), "OAK": (37.80, -122.27, -8),
    "LAC": (34.05, -118.24, -8), "SD": (32.72, -117.16, -8),
    "LAR": (34.05, -118.24, -8), "STL": (38.63, -90.20, -6),
    "MIA": (25.76, -80.19, -5), "MIN": (44.98, -93.27, -6),
    "NE": (42.09, -71.26, -5), "NO": (29.95, -90.07, -6),
    "NYG": (40.81, -74.07, -5), "NYJ": (40.81, -74.07, -5),
    "PHI": (39.95, -75.17, -5), "PIT": (40.44, -80.00, -5),
    "SEA": (47.61, -122.33, -8), "SF": (37.35, -121.95, -8),
    "TB": (27.95, -82.46, -5), "TEN": (36.16, -86.78, -6),
    "WAS": (38.91, -77.04, -5),
}

INTERNATIONAL_TERMS = (
    "wembley", "tottenham", "twickenham", "london", "allianz arena",
    "munich", "deutsche bank park", "frankfurt", "azteca", "mexico city",
    "neo quimica", "corinthians", "sao paulo", "são paulo", "brazil",
)


def haversine_miles(a: tuple[float, float], b: tuple[float, float]) -> float:
    lat1, lon1 = map(radians, a)
    lat2, lon2 = map(radians, b)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    h = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    return 3958.8 * 2 * asin(sqrt(h))


def _game_date_col(pbp: pd.DataFrame) -> str:
    for c in ("game_date", "gameday"):
        if c in pbp.columns:
            return c
    raise ValueError("PBP missing game date column (expected game_date or gameday)")


def build_schedule_context(pbp: pd.DataFrame) -> pd.DataFrame:
    date_col = _game_date_col(pbp)
    base_cols = ["game_id", "season", "week", "home_team", "away_team", date_col]
    optional = [c for c in ("stadium", "location") if c in pbp.columns]
    g = pbp[base_cols + optional].drop_duplicates("game_id", keep="last").copy()
    g["game_date"] = pd.to_datetime(g[date_col], errors="coerce")
    if g["game_date"].isna().all():
        raise ValueError(f"Could not parse PBP {date_col} as dates")

    # Build one row per team-game, then derive prior-game rest within season.
    home = g[["game_id", "season", "week", "game_date", "home_team"]].rename(
        columns={"home_team": "team"}
    )
    away = g[["game_id", "season", "week", "game_date", "away_team"]].rename(
        columns={"away_team": "team"}
    )
    appearances = pd.concat([home, away], ignore_index=True).sort_values(
        ["team", "season", "game_date", "game_id"]
    )
    appearances["prev_date"] = appearances.groupby(["team", "season"])["game_date"].shift(1)
    appearances["rest_days"] = (appearances["game_date"] - appearances["prev_date"]).dt.days

    hr = appearances.rename(columns={"team": "home_team", "rest_days": "home_rest_days"})[
        ["game_id", "home_team", "home_rest_days"]
    ]
    ar = appearances.rename(columns={"team": "away_team", "rest_days": "away_rest_days"})[
        ["game_id", "away_team", "away_rest_days"]
    ]
    g = g.merge(hr, on=["game_id", "home_team"], how="left").merge(
        ar, on=["game_id", "away_team"], how="left"
    )

    g["rest_diff"] = g["home_rest_days"] - g["away_rest_days"]
    g["home_short_week"] = g["home_rest_days"].le(6).astype(float)
    g["away_short_week"] = g["away_rest_days"].le(6).astype(float)
    g["home_extra_rest"] = g["home_rest_days"].ge(9).astype(float)
    g["away_extra_rest"] = g["away_rest_days"].ge(9).astype(float)

    def travel(row: pd.Series) -> float:
        h = TEAM_META.get(str(row.home_team))
        a = TEAM_META.get(str(row.away_team))
        if h is None or a is None:
            return np.nan
        return haversine_miles((a[0], a[1]), (h[0], h[1])) / 1000.0

    def tz_shift(row: pd.Series) -> float:
        h = TEAM_META.get(str(row.home_team))
        a = TEAM_META.get(str(row.away_team))
        if h is None or a is None:
            return np.nan
        return float(abs(a[2] - h[2]))

    g["away_travel_1000mi"] = g.apply(travel, axis=1)
    g["time_zone_shift"] = g.apply(tz_shift, axis=1)

    text = pd.Series("", index=g.index, dtype="object")
    for c in optional:
        text = text + " " + g[c].fillna("").astype(str).str.lower()
    pattern = "|".join(INTERNATIONAL_TERMS)
    g["international_game"] = text.str.contains(pattern, regex=True).astype(float)
    if "location" in g.columns:
        neutral = g["location"].fillna("").astype(str).str.lower().str.contains("neutral")
        g["neutral_site"] = neutral.astype(float)
    else:
        g["neutral_site"] = 0.0

    return g[[
        "game_id", "rest_diff", "home_short_week", "away_short_week",
        "home_extra_rest", "away_extra_rest", "away_travel_1000mi",
        "time_zone_shift", "international_game", "neutral_site",
    ]]


def run() -> None:
    pbp = pd.concat(
        [load_pbp(y) for y in range(START, HOLDOUT_END + 1)], ignore_index=True
    )
    g = prepare(pbp)

    tg = build_team_game_features(pbp)
    base = build_team_weekly_ratings(tg)
    core = [c for c in base if c.startswith("pregame_")]
    h = base.rename(columns={"team": "home_team", **{c: f"home_{c}" for c in core}})
    a = base.rename(columns={"team": "away_team", **{c: f"away_{c}" for c in core}})
    g = g.merge(
        h[["game_id", "home_team", *[f"home_{c}" for c in core]]],
        on=["game_id", "home_team"], how="left",
    ).merge(
        a[["game_id", "away_team", *[f"away_{c}" for c in core]]],
        on=["game_id", "away_team"], how="left",
    )
    g = g.merge(build_schedule_context(pbp), on="game_id", how="left")

    score = ["home_ppa5", "away_ppa5", "home_dpa3", "away_dpa3"]
    locked = score + cols(core, ["epa_per_play"]) + cols(core, ["explosive_rate"])
    sets = {
        "LOCKED_V12": locked,
        "REST_DIFF": locked + ["rest_diff"],
        "SHORT_WEEK": locked + ["home_short_week", "away_short_week"],
        "EXTRA_REST": locked + ["home_extra_rest", "away_extra_rest"],
        "TRAVEL_DISTANCE": locked + ["away_travel_1000mi"],
        "TIME_ZONE_SHIFT": locked + ["time_zone_shift"],
        "INTERNATIONAL_NEUTRAL": locked + ["international_game", "neutral_site"],
        "COMBINED_SCHEDULE_CONTEXT": locked + [
            "rest_diff", "home_short_week", "away_short_week",
            "home_extra_rest", "away_extra_rest", "away_travel_1000mi",
            "time_zone_shift", "international_game", "neutral_site",
        ],
    }

    allp = {k: [] for k in sets}
    print("=== V12 SCHEDULE / REST / TRAVEL EXPERIMENT ===")
    for y in TEST_SEASONS:
        tr = g[g.season.lt(y)].dropna(subset=["actual_total", "total_line"])
        te = g[g.season.eq(y)].dropna(subset=["actual_total", "total_line"])
        control = None
        print(f"\n=== {y} ===")
        for name, fs in sets.items():
            p = model(tr, te, fs)
            m = metrics(te.actual_total, p)
            if name == "LOCKED_V12":
                control = m
            row = dict(m)
            if name != "LOCKED_V12":
                row.update(
                    mae_vs_locked=round(m["mae"] - control["mae"], 4),
                    rmse_vs_locked=round(m["rmse"] - control["rmse"], 4),
                )
            print(name, row)
            z = te[["actual_total"]].copy()
            z["pred"] = p
            allp[name].append(z)

    print("\n=== OVERALL 2019-2025 WALK-FORWARD ===")
    cm = None
    for name, zs in allp.items():
        z = pd.concat(zs)
        m = metrics(z.actual_total, z.pred)
        if name == "LOCKED_V12":
            cm = m
        row = dict(m)
        if name != "LOCKED_V12":
            row.update(
                mae_vs_locked=round(m["mae"] - cm["mae"], 4),
                rmse_vs_locked=round(m["rmse"] - cm["rmse"], 4),
            )
        print(name, row)


if __name__ == "__main__":
    run()
