"""Validate simple Oak totals bet grades using edge + context confirmation."""
from __future__ import annotations

import numpy as np
import pandas as pd

from run_v12_edge_validation import build_games
from run_v12_integration_phase2 import TEST_SEASONS
from run_v12_incremental import model


def summarize(g, label):
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
        "avg_abs_edge": round(g.abs_edge.mean(), 3), "avg_score": round(g.confirm_score.mean(), 3),
    })


def grade_rows(r: pd.DataFrame) -> pd.DataFrame:
    out = r.copy()
    # Context flags are intentionally simple and pregame-only.
    out["weather_confirm"] = (
        out.high_wind_15.fillna(0).gt(0)
        | out.cold_40.fillna(0).gt(0)
        | out.precip.fillna(0).gt(0)
    ).astype(int)
    out["personnel_confirm"] = (
        out.home_personnel.abs().add(out.away_personnel.abs()).ge(0.75)
    ).astype(int)

    # Edge points: 1 point for >=1, 2 for >=2, 3 for >=3, capped at 4 for >=4.
    out["edge_points"] = np.select(
        [out.abs_edge.ge(4), out.abs_edge.ge(3), out.abs_edge.ge(2), out.abs_edge.ge(1)],
        [4, 3, 2, 1],
        default=0,
    )
    out["confirm_score"] = out.edge_points + out.weather_confirm + out.personnel_confirm

    # Grade hierarchy chosen before seeing this run's results.
    out["grade"] = np.select(
        [out.confirm_score.ge(5), out.confirm_score.eq(4), out.confirm_score.eq(3)],
        ["STRONG", "MEDIUM", "LEAN"],
        default="PASS",
    )
    return out


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
    r["bet_over"] = r.edge.gt(0)
    r["result"] = np.where(
        r.actual_total.eq(r.total_line),
        "push",
        np.where(
            (r.actual_total.gt(r.total_line) & r.bet_over)
            | (r.actual_total.lt(r.total_line) & ~r.bet_over),
            "win",
            "loss",
        ),
    )
    r = grade_rows(r)

    print("=== V12 CONFIRMATION SCORE GRADE VALIDATION ===")
    for grade in ["STRONG", "MEDIUM", "LEAN", "PASS"]:
        summarize(r[r.grade.eq(grade)], grade)

    print("=== ACTIONABLE ONLY ===")
    summarize(r[r.grade.ne("PASS")], "LEAN_PLUS")
    summarize(r[r.grade.isin(["MEDIUM", "STRONG"])], "MEDIUM_PLUS")
    summarize(r[r.grade.eq("STRONG")], "STRONG_ONLY")

    print("=== DIRECTION X GRADE ===")
    for grade in ["STRONG", "MEDIUM", "LEAN"]:
        summarize(r[(r.grade.eq(grade)) & r.edge.gt(0)], f"OVER_{grade}")
        summarize(r[(r.grade.eq(grade)) & r.edge.lt(0)], f"UNDER_{grade}")

    print("=== SEASON X GRADE ===")
    for year in TEST_SEASONS:
        for grade in ["STRONG", "MEDIUM", "LEAN"]:
            summarize(r[(r.season.eq(year)) & r.grade.eq(grade)], f"{year}_{grade}")


if __name__ == "__main__":
    run()
