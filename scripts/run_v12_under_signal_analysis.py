"""Robustness analysis for integrated V12 Under signals."""
from __future__ import annotations

import numpy as np
import pandas as pd

from run_v12_edge_validation import build_games
from run_v12_integration_phase2 import TEST_SEASONS
from run_v12_incremental import model


def summarize(g: pd.DataFrame, label: str) -> None:
    n = len(g)
    if not n:
        return
    decided = g[g.result.ne("push")]
    wins = int((decided.result == "win").sum())
    losses = int((decided.result == "loss").sum())
    pushes = n - wins - losses
    win_rate = wins / (wins + losses) if wins + losses else np.nan
    roi_110 = (wins * (100 / 110) - losses) / (wins + losses) if wins + losses else np.nan
    print(label, {
        "games": n,
        "wins": wins,
        "losses": losses,
        "pushes": pushes,
        "win_rate": round(win_rate, 4),
        "roi_-110": round(roi_110, 4),
        "avg_edge": round(g.edge.mean(), 3),
        "avg_total": round(g.total_line.mean(), 3),
    })


def run():
    games, features = build_games()
    preds = []
    for year in TEST_SEASONS:
        train = games[games.season.lt(year)].dropna(subset=["actual_total", "total_line"])
        test = games[games.season.eq(year)].dropna(subset=["actual_total", "total_line"]).copy()
        test["pred"] = model(train, test, features)
        preds.append(test)
    r = pd.concat(preds, ignore_index=True)
    r["edge"] = r.pred - r.total_line
    r["abs_edge"] = r.edge.abs()
    r["bet_under"] = r.edge.lt(0)
    r["result"] = np.where(
        r.actual_total.eq(r.total_line),
        "push",
        np.where(
            r.actual_total.lt(r.total_line),
            np.where(r.bet_under, "win", "loss"),
            np.where(r.bet_under, "loss", "win"),
        ),
    )

    for threshold in [2, 3]:
        u = r[(r.bet_under) & (r.abs_edge >= threshold)].copy()
        print(f"=== UNDER {threshold}+ ROBUSTNESS ===")
        summarize(u, "ALL")

        print("-- WEATHER --")
        summarize(u[u.high_wind_15.eq(1)], "HIGH_WIND_15")
        summarize(u[u.cold_40.eq(1)], "COLD_40")
        summarize(u[u.precip.eq(1)], "PRECIP")
        summarize(u[(u.high_wind_15.eq(0)) & (u.cold_40.eq(0)) & (u.precip.eq(0))], "NO_EXTREME_WEATHER")

        print("-- TOTAL RANGE --")
        for lo, hi in [(0, 42), (42, 46), (46, 50), (50, 99)]:
            summarize(u[(u.total_line >= lo) & (u.total_line < hi)], f"TOTAL_{lo}_{hi}")

        print("-- WEEK --")
        summarize(u[u.week <= 6], "WEEKS_1_6")
        summarize(u[(u.week >= 7) & (u.week <= 12)], "WEEKS_7_12")
        summarize(u[u.week >= 13], "WEEKS_13_PLUS")

        print("-- QB/PERSONNEL CONFIRMATION --")
        # Negative home+away personnel points means more aggregate absences; split by median severity.
        u["personnel_sum"] = u.home_personnel + u.away_personnel
        med = u.personnel_sum.median()
        summarize(u[u.personnel_sum <= med], "MORE_PERSONNEL_ABSENCE")
        summarize(u[u.personnel_sum > med], "LESS_PERSONNEL_ABSENCE")

        print("-- SEASON --")
        for y in TEST_SEASONS:
            summarize(u[u.season.eq(y)], str(y))


if __name__ == "__main__":
    run()
