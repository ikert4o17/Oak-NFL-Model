"""Test whether Oak Over edges exploit market over-suppression."""
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

    adverse = (
        r.high_wind_15.fillna(0).gt(0)
        | r.cold_40.fillna(0).gt(0)
        | r.precip.fillna(0).gt(0)
    )
    personnel = r.home_personnel.abs().add(r.away_personnel.abs()).ge(0.75)
    edge_2_3 = r.edge.ge(2) & r.edge.lt(3)
    low_total = r.total_line.lt(42)
    late = r.week.ge(13)

    # Simple overreaction score: conditions likely to push market totals down,
    # plus Oak's empirically stable 2-3 point Over disagreement.
    r["overreaction_score"] = (
        edge_2_3.astype(int)
        + low_total.astype(int)
        + adverse.astype(int)
        + personnel.astype(int)
        + late.astype(int)
    )

    print("=== V12 OVER MARKET-OVERREACTION ANALYSIS ===")
    summarize(r, "ALL_OVERS")

    print("=== INDIVIDUAL FLAGS ===")
    summarize(r[edge_2_3], "EDGE_2_3")
    summarize(r[low_total], "TOTAL_LT_42")
    summarize(r[adverse], "ADVERSE_WEATHER")
    summarize(r[personnel], "MORE_PERSONNEL_ABSENCE")
    summarize(r[late], "WEEKS_13_PLUS")

    print("=== FLAG COMBINATIONS ===")
    combos = {
        "EDGE_2_3_AND_LOW_TOTAL": edge_2_3 & low_total,
        "EDGE_2_3_AND_ADVERSE": edge_2_3 & adverse,
        "EDGE_2_3_AND_PERSONNEL": edge_2_3 & personnel,
        "EDGE_2_3_AND_LATE": edge_2_3 & late,
        "LOW_TOTAL_AND_ADVERSE": low_total & adverse,
        "LOW_TOTAL_AND_PERSONNEL": low_total & personnel,
        "LOW_TOTAL_AND_LATE": low_total & late,
        "ADVERSE_AND_PERSONNEL": adverse & personnel,
        "ADVERSE_AND_LATE": adverse & late,
        "PERSONNEL_AND_LATE": personnel & late,
        "EDGE_2_3_LOW_TOTAL_LATE": edge_2_3 & low_total & late,
        "EDGE_2_3_ADVERSE_PERSONNEL": edge_2_3 & adverse & personnel,
    }
    for label, mask in combos.items():
        summarize(r[mask], label)

    print("=== OVERREACTION SCORE ===")
    for score in range(0, 6):
        summarize(r[r.overreaction_score.eq(score)], f"SCORE_{score}")
    for threshold in range(1, 6):
        summarize(r[r.overreaction_score.ge(threshold)], f"SCORE_{threshold}_PLUS")

    print("=== SEASON X SCORE / KEY FILTERS ===")
    for year in TEST_SEASONS:
        yr = r[r.season.eq(year)]
        summarize(yr[yr.edge.ge(2) & yr.edge.lt(3)], f"{year}_EDGE_2_3")
        summarize(yr[yr.total_line.lt(42)], f"{year}_TOTAL_LT_42")
        summarize(yr[yr.week.ge(13)], f"{year}_WEEKS_13_PLUS")
        for threshold in [2, 3, 4]:
            summarize(yr[yr.overreaction_score.ge(threshold)], f"{year}_SCORE_{threshold}_PLUS")


if __name__ == "__main__":
    run()
