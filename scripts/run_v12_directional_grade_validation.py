"""Validate direction-aware Oak totals grades for Overs and Unders."""
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
        "avg_abs_edge": round(g.abs_edge.mean(), 3), "avg_score": round(g.direction_score.mean(), 3),
    })


def grade_rows(r: pd.DataFrame) -> pd.DataFrame:
    out = r.copy()
    adverse_weather = (
        out.high_wind_15.fillna(0).gt(0)
        | out.cold_40.fillna(0).gt(0)
        | out.precip.fillna(0).gt(0)
    )
    personnel_absence = out.home_personnel.abs().add(out.away_personnel.abs()).ge(0.75)

    edge_points = np.select(
        [out.abs_edge.ge(4), out.abs_edge.ge(3), out.abs_edge.ge(2), out.abs_edge.ge(1)],
        [4, 3, 2, 1], default=0,
    )

    # Unders: adverse weather and personnel absence are positive confirmations.
    under_score = edge_points + adverse_weather.astype(int) + personnel_absence.astype(int)

    # Overs: do not reward suppression contexts. Instead reward clean environment
    # and relatively healthy personnel. This is intentionally simple for validation.
    over_score = edge_points + (~adverse_weather).astype(int) + (~personnel_absence).astype(int)

    out["direction_score"] = np.where(out.edge.lt(0), under_score, over_score)
    out["grade"] = np.select(
        [out.direction_score.ge(5), out.direction_score.eq(4), out.direction_score.eq(3)],
        ["STRONG", "MEDIUM", "LEAN"], default="PASS",
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
        r.actual_total.eq(r.total_line), "push",
        np.where(
            (r.actual_total.gt(r.total_line) & r.bet_over)
            | (r.actual_total.lt(r.total_line) & ~r.bet_over),
            "win", "loss",
        ),
    )
    r = grade_rows(r)

    print("=== V12 DIRECTION-AWARE GRADE VALIDATION ===")
    for direction, mask in [("OVER", r.edge.gt(0)), ("UNDER", r.edge.lt(0))]:
        print(f"=== {direction} GRADES ===")
        for grade in ["STRONG", "MEDIUM", "LEAN", "PASS"]:
            summarize(r[mask & r.grade.eq(grade)], f"{direction}_{grade}")

    print("=== ACTIONABLE BY DIRECTION ===")
    summarize(r[r.edge.gt(0) & r.grade.ne("PASS")], "OVER_LEAN_PLUS")
    summarize(r[r.edge.lt(0) & r.grade.ne("PASS")], "UNDER_LEAN_PLUS")
    summarize(r[r.edge.gt(0) & r.grade.isin(["MEDIUM", "STRONG"])], "OVER_MEDIUM_PLUS")
    summarize(r[r.edge.lt(0) & r.grade.isin(["MEDIUM", "STRONG"])], "UNDER_MEDIUM_PLUS")

    print("=== SEASON X DIRECTION X GRADE ===")
    for year in TEST_SEASONS:
        for direction, mask in [("OVER", r.edge.gt(0)), ("UNDER", r.edge.lt(0))]:
            for grade in ["STRONG", "MEDIUM", "LEAN"]:
                summarize(r[r.season.eq(year) & mask & r.grade.eq(grade)], f"{year}_{direction}_{grade}")


if __name__ == "__main__":
    run()
