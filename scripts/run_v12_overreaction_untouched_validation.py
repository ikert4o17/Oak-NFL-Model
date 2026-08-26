"""Validate the frozen Oak V12 Overreaction Score 2+ on untouched 2019-2022 seasons."""
from __future__ import annotations

import numpy as np
import pandas as pd

from run_v12_edge_validation import build_games
from run_v12_incremental import model

VALIDATION_SEASONS = range(2019, 2023)


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


def add_frozen_score(r: pd.DataFrame) -> pd.DataFrame:
    out = r.copy()
    adverse = (
        out.high_wind_15.fillna(0).gt(0)
        | out.cold_40.fillna(0).gt(0)
        | out.precip.fillna(0).gt(0)
    )
    personnel = out.home_personnel.abs().add(out.away_personnel.abs()).ge(0.75)
    edge_2_3 = out.edge.ge(2) & out.edge.lt(3)
    low_total = out.total_line.lt(42)
    late = out.week.ge(13)
    out["overreaction_score"] = (
        edge_2_3.astype(int)
        + low_total.astype(int)
        + adverse.astype(int)
        + personnel.astype(int)
        + late.astype(int)
    )
    return out


def run() -> None:
    games, features = build_games()
    preds = []
    for year in VALIDATION_SEASONS:
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
    r = add_frozen_score(r)

    print("=== V12 FROZEN OVERREACTION SCORE UNTOUCHED VALIDATION: 2019-2022 ===")
    summarize(r, "ALL_OVERS")
    summarize(r[r.overreaction_score.ge(2)], "FROZEN_SCORE_2_PLUS")
    summarize(r[r.overreaction_score.ge(3)], "REFERENCE_SCORE_3_PLUS")
    summarize(r[r.overreaction_score.ge(4)], "REFERENCE_SCORE_4_PLUS")

    print("=== YEAR X FROZEN SCORE 2+ ===")
    for year in VALIDATION_SEASONS:
        yr = r[r.season.eq(year)]
        summarize(yr, f"{year}_ALL_OVERS")
        summarize(yr[yr.overreaction_score.ge(2)], f"{year}_SCORE_2_PLUS")

    print("=== FROZEN SCORE DISTRIBUTION ===")
    for score in range(0, 6):
        summarize(r[r.overreaction_score.eq(score)], f"SCORE_{score}")


if __name__ == "__main__":
    run()
