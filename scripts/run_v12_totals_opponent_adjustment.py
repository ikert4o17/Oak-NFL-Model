"""Totals-specific opponent adjustment challenger against promoted V12 + rest.

Single predetermined method, no parameter search:
- compute each team's raw pregame EPA/play and explosive-rate offense/defense;
- for each completed prior team-game, compare the game result to the opponent's
  pregame counterpart rating entering that game;
- season-to-date opponent adjustment is the expanding mean of those residuals,
  shifted one game;
- adjusted rating = incumbent raw pregame rating + prior residual adjustment.

This asks whether a team's efficiency was achieved against unusually strong or
weak opponents while keeping every input pregame-only and season-local.
"""
from __future__ import annotations

import pandas as pd

from oak_nfl.data.nflverse import load_pbp
from oak_nfl.features import build_team_game_features, build_team_weekly_ratings
from run_v12_core_shootout import cols
from run_v12_incremental import HOLDOUT_END, START, metrics, model, prepare
from run_v12_schedule_context import build_schedule_context

TEST_SEASONS = range(2019, 2026)


def build_adjusted_ratings(team_games: pd.DataFrame, ratings: pd.DataFrame) -> pd.DataFrame:
    rcols = [c for c in ratings.columns if c.startswith("pregame_")]
    r = ratings.copy()

    # Attach opponent pregame ratings to each offensive team-game row.
    opp = r.rename(columns={"team": "defteam", **{c: f"opp_{c}" for c in rcols}})
    x = team_games.merge(
        opp[["game_id", "defteam", *[f"opp_{c}" for c in rcols]]],
        on=["game_id", "defteam"], how="left"
    ).sort_values(["posteam", "season", "week", "game_id"])

    # Residual versus opponent expectation, using only metrics already in locked V12.
    x["epa_off_resid"] = x["epa_per_play"] - x["opp_pregame_def_epa_per_play_allowed"]
    x["expl_off_resid"] = x["explosive_rate"] - x["opp_pregame_def_explosive_rate_allowed"]

    # Defensive residual: what this defense allowed relative to opponent offense entering game.
    own = r.rename(columns={"team": "defteam", **{c: f"defteam_{c}" for c in rcols}})
    y = team_games.merge(
        r.rename(columns={"team": "posteam", **{c: f"offteam_{c}" for c in rcols}})[
            ["game_id", "posteam", *[f"offteam_{c}" for c in rcols]]
        ],
        on=["game_id", "posteam"], how="left"
    ).sort_values(["defteam", "season", "week", "game_id"])
    y["epa_def_resid"] = y["epa_per_play"] - y["offteam_pregame_off_epa_per_play"]
    y["expl_def_resid"] = y["explosive_rate"] - y["offteam_pregame_off_explosive_rate"]

    off_adj = x[["game_id", "season", "week", "posteam", "epa_off_resid", "expl_off_resid"]].rename(columns={"posteam": "team"})
    def_adj = y[["game_id", "season", "week", "defteam", "epa_def_resid", "expl_def_resid"]].rename(columns={"defteam": "team"})

    z = off_adj.merge(def_adj, on=["game_id", "season", "week", "team"], how="outer").sort_values(["team", "season", "week", "game_id"])
    for c in ["epa_off_resid", "expl_off_resid", "epa_def_resid", "expl_def_resid"]:
        z[f"pregame_{c}_adj"] = z.groupby(["team", "season"])[c].transform(lambda s: s.expanding().mean().shift(1))

    keep_adj = ["game_id", "team", *[f"pregame_{c}_adj" for c in ["epa_off_resid", "expl_off_resid", "epa_def_resid", "expl_def_resid"]]]
    out = r.merge(z[keep_adj], on=["game_id", "team"], how="left")
    out["pregame_adj_off_epa_per_play"] = out["pregame_off_epa_per_play"] + out["pregame_epa_off_resid_adj"]
    out["pregame_adj_def_epa_per_play_allowed"] = out["pregame_def_epa_per_play_allowed"] + out["pregame_epa_def_resid_adj"]
    out["pregame_adj_off_explosive_rate"] = out["pregame_off_explosive_rate"] + out["pregame_expl_off_resid_adj"]
    out["pregame_adj_def_explosive_rate_allowed"] = out["pregame_def_explosive_rate_allowed"] + out["pregame_expl_def_resid_adj"]
    return out


def run() -> None:
    pbp = pd.concat([load_pbp(y) for y in range(START, HOLDOUT_END + 1)], ignore_index=True)
    g = prepare(pbp)
    team_games = build_team_game_features(pbp)
    ratings = build_team_weekly_ratings(team_games)
    adjusted = build_adjusted_ratings(team_games, ratings)

    base_core = [c for c in ratings.columns if c.startswith("pregame_")]
    adj_cols = [
        "pregame_adj_off_epa_per_play", "pregame_adj_def_epa_per_play_allowed",
        "pregame_adj_off_explosive_rate", "pregame_adj_def_explosive_rate_allowed",
    ]

    h = adjusted.rename(columns={"team": "home_team", **{c: f"home_{c}" for c in [*base_core, *adj_cols]}})
    a = adjusted.rename(columns={"team": "away_team", **{c: f"away_{c}" for c in [*base_core, *adj_cols]}})
    g = g.merge(h[["game_id", "home_team", *[f"home_{c}" for c in [*base_core, *adj_cols]]]], on=["game_id", "home_team"], how="left")
    g = g.merge(a[["game_id", "away_team", *[f"away_{c}" for c in [*base_core, *adj_cols]]]], on=["game_id", "away_team"], how="left")
    g = g.merge(build_schedule_context(pbp)[["game_id", "rest_diff"]], on="game_id", how="left")

    score = ["home_ppa5", "away_ppa5", "home_dpa3", "away_dpa3"]
    promoted = score + cols(base_core, ["epa_per_play"]) + cols(base_core, ["explosive_rate"]) + ["rest_diff"]
    adj_model = score + [f"{s}_{c}" for s in ("home", "away") for c in adj_cols] + ["rest_diff"]
    augmented = promoted + [f"{s}_{c}" for s in ("home", "away") for c in adj_cols]
    sets = {
        "PROMOTED_V12_REST": promoted,
        "OPP_ADJUSTED_REPLACEMENT": adj_model,
        "RAW_PLUS_OPP_ADJUSTED": augmented,
    }

    allp = {k: [] for k in sets}
    print("=== V12 TOTALS-SPECIFIC OPPONENT ADJUSTMENT ===")
    for y in TEST_SEASONS:
        tr = g[g.season.lt(y)].dropna(subset=["actual_total", "total_line"])
        te = g[g.season.eq(y)].dropna(subset=["actual_total", "total_line"])
        control = None
        print(f"\n=== {y} ===")
        for name, fs in sets.items():
            pred = model(tr, te, fs)
            m = metrics(te.actual_total, pred)
            if name == "PROMOTED_V12_REST": control = m
            row = dict(m)
            if name != "PROMOTED_V12_REST":
                row.update(mae_vs_control=round(m["mae"]-control["mae"],4), rmse_vs_control=round(m["rmse"]-control["rmse"],4))
            print(name, row)
            z = te[["actual_total"]].copy(); z["pred"] = pred; allp[name].append(z)

    print("\n=== OVERALL 2019-2025 WALK-FORWARD ===")
    control = None
    for name, zs in allp.items():
        z = pd.concat(zs); m = metrics(z.actual_total, z.pred)
        if name == "PROMOTED_V12_REST": control = m
        row = dict(m)
        if name != "PROMOTED_V12_REST":
            row.update(mae_vs_control=round(m["mae"]-control["mae"],4), rmse_vs_control=round(m["rmse"]-control["rmse"],4))
        print(name, row)


if __name__ == "__main__":
    run()
