"""Frozen validation of rest differential against locked V12.

This is a confirmation test, not a tuning pass. The rest differential definition
is copied unchanged from the successful schedule-context experiment. We test:
1. the original 2019-2025 walk-forward result for reproducibility;
2. early (2019-2022) versus late (2023-2025) eras;
3. performance excluding 2025 to ensure the aggregate result is not dependent on
   the strongest single season;
4. direct head-to-head game-level absolute-error wins/ties/losses.

No alternate rest cutoffs, transforms, interactions, or weights are searched.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from oak_nfl.data.nflverse import load_pbp
from oak_nfl.features import build_team_game_features, build_team_weekly_ratings
from run_v12_core_shootout import cols
from run_v12_incremental import HOLDOUT_END, START, metrics, model, prepare
from run_v12_schedule_context import build_schedule_context

TEST_SEASONS = range(2019, 2026)


def summarize(label: str, z: pd.DataFrame) -> None:
    c = metrics(z.actual_total, z.locked_pred)
    r = metrics(z.actual_total, z.rest_pred)
    locked_abs = (z.actual_total - z.locked_pred).abs()
    rest_abs = (z.actual_total - z.rest_pred).abs()
    print(label, {
        "games": len(z),
        "locked_mae": c["mae"], "rest_mae": r["mae"],
        "mae_delta": round(r["mae"] - c["mae"], 4),
        "locked_rmse": c["rmse"], "rest_rmse": r["rmse"],
        "rmse_delta": round(r["rmse"] - c["rmse"], 4),
        "rest_wins": int((rest_abs < locked_abs).sum()),
        "ties": int(np.isclose(rest_abs, locked_abs).sum()),
        "rest_losses": int((rest_abs > locked_abs).sum()),
    })


def run() -> None:
    pbp = pd.concat([load_pbp(y) for y in range(START, HOLDOUT_END + 1)], ignore_index=True)
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
    g = g.merge(build_schedule_context(pbp)[["game_id", "rest_diff"]], on="game_id", how="left")

    score = ["home_ppa5", "away_ppa5", "home_dpa3", "away_dpa3"]
    locked = score + cols(core, ["epa_per_play"]) + cols(core, ["explosive_rate"])
    rest = locked + ["rest_diff"]

    rows = []
    print("=== V12 REST DIFFERENTIAL FROZEN VALIDATION ===")
    for y in TEST_SEASONS:
        tr = g[g.season.lt(y)].dropna(subset=["actual_total", "total_line"])
        te = g[g.season.eq(y)].dropna(subset=["actual_total", "total_line"]).copy()
        te["locked_pred"] = model(tr, te, locked)
        te["rest_pred"] = model(tr, te, rest)
        te["season"] = y
        rows.append(te[["season", "actual_total", "locked_pred", "rest_pred"]])
        summarize(str(y), rows[-1])

    z = pd.concat(rows, ignore_index=True)
    print("\n=== STABILITY CHECKS ===")
    summarize("OVERALL_2019_2025", z)
    summarize("EARLY_2019_2022", z[z.season.le(2022)])
    summarize("LATE_2023_2025", z[z.season.ge(2023)])
    summarize("EXCLUDING_2025", z[z.season.ne(2025)])

    # Predetermined promotion gate: both MAE and RMSE must improve overall and
    # excluding 2025, and neither early nor late era may materially regress.
    def deltas(x: pd.DataFrame) -> tuple[float, float]:
        c = metrics(x.actual_total, x.locked_pred)
        r = metrics(x.actual_total, x.rest_pred)
        return r["mae"] - c["mae"], r["rmse"] - c["rmse"]

    overall = deltas(z)
    ex25 = deltas(z[z.season.ne(2025)])
    early = deltas(z[z.season.le(2022)])
    late = deltas(z[z.season.ge(2023)])
    passes = (
        overall[0] < 0 and overall[1] < 0
        and ex25[0] < 0 and ex25[1] < 0
        and early[0] <= 0.01 and early[1] <= 0.01
        and late[0] <= 0.01 and late[1] <= 0.01
    )
    print("\nPROMOTION_GATE", "PASS" if passes else "FAIL", {
        "overall": tuple(round(v, 4) for v in overall),
        "excluding_2025": tuple(round(v, 4) for v in ex25),
        "early": tuple(round(v, 4) for v in early),
        "late": tuple(round(v, 4) for v in late),
    })


if __name__ == "__main__":
    run()
