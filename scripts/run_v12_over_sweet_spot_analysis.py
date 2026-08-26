"""Search for stable sweet spots in Oak V12 Over edges."""
from __future__ import annotations

import numpy as np
import pandas as pd

from run_v12_edge_validation import build_games
from run_v12_integration_phase2 import TEST_SEASONS
from run_v12_incremental import model


def summarize(g: pd.DataFrame, label: str) -> None:
    if g.empty:
        return
    decided = g[g.result.ne("push")]
    wins = int((decided.result == "win").sum())
    losses = int((decided.result == "loss").sum())
    pushes = len(g) - wins - losses
    win_rate = wins / (wins + losses) if wins + losses else np.nan
    roi = (wins * (100 / 110) - losses) / (wins + losses) if wins + losses else np.nan
    print(label, {
        "games": len(g), "wins": wins, "losses": losses, "pushes": pushes,
        "win_rate": round(win_rate, 4), "roi_-110": round(roi, 4),
        "avg_edge": round(g.edge.mean(), 3), "avg_total": round(g.total_line.mean(), 3),
    })


def run() -> None:
    games, features = build_games()
    preds = []
    for year in TEST_SEASONS:
        train = games[games.season.lt(year)].dropna(subset=["actual_total", "total_line"])
        test = games[games.season.eq(year)].dropna(subset=["actual_total", "total_line"]).copy()
        test["pred"] = model(train, test, features)
        preds.append(test)

    r = pd.concat(preds, ignore_index=True)
    r["edge"] = r.pred - r.total_line
    r = r[r.edge.gt(0)].copy()
    r["result"] = np.where(
        r.actual_total.eq(r.total_line), "push",
        np.where(r.actual_total.gt(r.total_line), "win", "loss"),
    )
    adverse_weather = (
        r.high_wind_15.fillna(0).gt(0)
        | r.cold_40.fillna(0).gt(0)
        | r.precip.fillna(0).gt(0)
    )
    personnel_absence = r.home_personnel.abs().add(r.away_personnel.abs()).ge(0.75)

    print("=== V12 OVER SWEET-SPOT ANALYSIS ===")
    print("=== EDGE BANDS ===")
    bands = [(0, 1), (1, 1.5), (1.5, 2), (2, 2.5), (2.5, 3), (3, 3.5), (3.5, 4), (4, 99)]
    for lo, hi in bands:
        summarize(r[r.edge.ge(lo) & r.edge.lt(hi)], f"EDGE_{lo:g}_{hi:g}")

    print("=== CANDIDATE EDGE WINDOWS ===")
    windows = [(1, 2), (1, 2.5), (1.5, 2.5), (1.5, 3), (2, 3), (2, 3.5), (2.5, 3.5)]
    for lo, hi in windows:
        summarize(r[r.edge.ge(lo) & r.edge.lt(hi)], f"WINDOW_{lo:g}_{hi:g}")

    print("=== MARKET TOTAL RANGE ===")
    for lo, hi in [(0, 42), (42, 46), (46, 50), (50, 99)]:
        summarize(r[r.total_line.ge(lo) & r.total_line.lt(hi)], f"TOTAL_{lo:g}_{hi:g}")

    print("=== ENVIRONMENT / PERSONNEL ===")
    summarize(r[~adverse_weather], "CLEAN_WEATHER")
    summarize(r[adverse_weather], "ADVERSE_WEATHER")
    summarize(r[~personnel_absence], "HEALTHIER_PERSONNEL")
    summarize(r[personnel_absence], "MORE_PERSONNEL_ABSENCE")
    summarize(r[(~adverse_weather) & (~personnel_absence)], "CLEAN_AND_HEALTHIER")

    print("=== WEEK ===")
    summarize(r[r.week.le(6)], "WEEKS_1_6")
    summarize(r[r.week.between(7, 12)], "WEEKS_7_12")
    summarize(r[r.week.ge(13)], "WEEKS_13_PLUS")

    print("=== SEASON X CANDIDATE WINDOW ===")
    for year in TEST_SEASONS:
        yr = r[r.season.eq(year)]
        for lo, hi in windows:
            summarize(yr[yr.edge.ge(lo) & yr.edge.lt(hi)], f"{year}_WINDOW_{lo:g}_{hi:g}")

    print("=== EDGE WINDOW X CONTEXT ===")
    for lo, hi in [(1, 2.5), (1.5, 3), (2, 3)]:
        w = r.edge.ge(lo) & r.edge.lt(hi)
        summarize(r[w & ~adverse_weather], f"WINDOW_{lo:g}_{hi:g}_CLEAN_WEATHER")
        summarize(r[w & ~personnel_absence], f"WINDOW_{lo:g}_{hi:g}_HEALTHIER")
        summarize(r[w & (~adverse_weather) & (~personnel_absence)], f"WINDOW_{lo:g}_{hi:g}_CLEAN_HEALTHIER")


if __name__ == "__main__":
    run()
