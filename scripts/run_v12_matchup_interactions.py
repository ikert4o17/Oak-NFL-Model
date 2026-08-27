"""Test leakage-safe matchup interactions against the locked V12 totals control.

The locked control remains PPA5 + DPA3 + EPA/play + explosive rate. This
experiment adds only pregame offense-vs-opponent-defense interaction terms and
uses expanding walk-forward evaluation so no current/future game data leaks.
"""
from __future__ import annotations

import pandas as pd

from oak_nfl.data.nflverse import load_pbp
from oak_nfl.features import build_team_game_features, build_team_weekly_ratings
from run_v12_core_shootout import cols
from run_v12_incremental import HOLDOUT_END, START, metrics, model, prepare

TEST_SEASONS = range(2019, 2026)


def add_matchup_interactions(g: pd.DataFrame) -> pd.DataFrame:
    out = g.copy()

    # A positive defensive "allowed" value is worse defense. Products let Ridge
    # represent cases where a strength meets a matching weakness rather than
    # assuming offense and defense contribute only additively.
    pairs = {
        "pass_epa": "pass_epa_per_play",
        "rush_epa": "rush_epa_per_play",
        "success": "success_rate",
        "explosive": "explosive_rate",
    }
    for label, metric in pairs.items():
        ho = f"home_pregame_off_{metric}"
        hd = f"home_pregame_def_{metric}_allowed"
        ao = f"away_pregame_off_{metric}"
        ad = f"away_pregame_def_{metric}_allowed"
        if not all(c in out.columns for c in [ho, hd, ao, ad]):
            continue
        out[f"home_{label}_matchup_product"] = out[ho] * out[ad]
        out[f"away_{label}_matchup_product"] = out[ao] * out[hd]
        out[f"{label}_matchup_sum"] = out[ho] + out[ad] + out[ao] + out[hd]
        out[f"{label}_matchup_imbalance"] = (out[ho] + out[ad]) - (out[ao] + out[hd])
    return out


def run() -> None:
    pbp = pd.concat([load_pbp(y) for y in range(START, HOLDOUT_END + 1)], ignore_index=True)
    g = prepare(pbp)
    ratings = build_team_weekly_ratings(build_team_game_features(pbp))
    core = [c for c in ratings.columns if c.startswith("pregame_")]

    home = ratings.rename(columns={"team": "home_team", **{c: f"home_{c}" for c in core}})
    away = ratings.rename(columns={"team": "away_team", **{c: f"away_{c}" for c in core}})
    g = g.merge(
        home[["game_id", "home_team", *[f"home_{c}" for c in core]]],
        on=["game_id", "home_team"], how="left",
    ).merge(
        away[["game_id", "away_team", *[f"away_{c}" for c in core]]],
        on=["game_id", "away_team"], how="left",
    )
    g = add_matchup_interactions(g)

    score = ["home_ppa5", "away_ppa5", "home_dpa3", "away_dpa3"]
    locked = score + cols(core, ["epa_per_play"]) + cols(core, ["explosive_rate"])

    families = {
        "PASS_MATCHUP": ["home_pass_epa_matchup_product", "away_pass_epa_matchup_product",
                         "pass_epa_matchup_sum", "pass_epa_matchup_imbalance"],
        "RUSH_MATCHUP": ["home_rush_epa_matchup_product", "away_rush_epa_matchup_product",
                         "rush_epa_matchup_sum", "rush_epa_matchup_imbalance"],
        "SUCCESS_MATCHUP": ["home_success_matchup_product", "away_success_matchup_product",
                            "success_matchup_sum", "success_matchup_imbalance"],
        "EXPLOSIVE_MATCHUP": ["home_explosive_matchup_product", "away_explosive_matchup_product",
                              "explosive_matchup_sum", "explosive_matchup_imbalance"],
    }
    families = {k: [c for c in v if c in g.columns] for k, v in families.items()}
    sets = {"LOCKED_V12": locked}
    for name, extra in families.items():
        if extra:
            sets[f"LOCKED_PLUS_{name}"] = locked + extra
    all_extra = [c for values in families.values() for c in values]
    sets["LOCKED_PLUS_ALL_MATCHUPS"] = locked + all_extra

    print("=== V12 MATCHUP INTERACTION EXPERIMENT ===")
    print("Control is frozen locked V12; matchup terms are pregame-only.")
    all_rows = {name: [] for name in sets}

    for year in TEST_SEASONS:
        tr = g[g.season.lt(year)].dropna(subset=["actual_total", "total_line"])
        te = g[g.season.eq(year)].dropna(subset=["actual_total", "total_line"])
        market = metrics(te.actual_total, te.total_line)
        print(f"\n=== {year} ===")
        print("MARKET", market)
        for name, features in sets.items():
            pred = model(tr, te, features)
            m = metrics(te.actual_total, pred)
            print(name, {
                **m,
                "mae_vs_market": round(m["mae"] - market["mae"], 4),
                "rmse_vs_market": round(m["rmse"] - market["rmse"], 4),
            })
            x = te[["game_id", "season", "actual_total", "total_line"]].copy()
            x["pred"] = pred
            all_rows[name].append(x)

    print("\n=== OVERALL 2019-2025 WALK-FORWARD ===")
    control = None
    for name, pieces in all_rows.items():
        x = pd.concat(pieces, ignore_index=True)
        m = metrics(x.actual_total, x.pred)
        market = metrics(x.actual_total, x.total_line)
        if name == "LOCKED_V12":
            control = m
        row = {
            **m,
            "mae_vs_market": round(m["mae"] - market["mae"], 4),
            "rmse_vs_market": round(m["rmse"] - market["rmse"], 4),
        }
        if control is not None and name != "LOCKED_V12":
            row["mae_vs_locked"] = round(m["mae"] - control["mae"], 4)
            row["rmse_vs_locked"] = round(m["rmse"] - control["rmse"], 4)
        print(name, row)


if __name__ == "__main__":
    run()
