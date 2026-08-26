"""Diagnostic autopsy for frozen V12 Overreaction Score 2+; no retuning."""
from __future__ import annotations

import numpy as np
import pandas as pd

from run_v12_edge_validation import build_games
from run_v12_incremental import model
from run_v12_overreaction_untouched_validation import add_frozen_score

SEASONS = (2019, 2020, 2021, 2022)


def summarize(g: pd.DataFrame, label: str) -> None:
    if g.empty:
        print(label, {"games": 0})
        return
    d = g[g.result.ne("push")]
    w = int((d.result == "win").sum()); l = int((d.result == "loss").sum())
    wr = w / (w + l) if w + l else np.nan
    roi = (w * (100 / 110) - l) / (w + l) if w + l else np.nan
    print(label, {"games": len(g), "wins": w, "losses": l, "pushes": len(g)-w-l,
                  "win_rate": round(wr,4), "roi_-110": round(roi,4),
                  "avg_edge": round(g.edge.mean(),3), "avg_total": round(g.total_line.mean(),3)})


def build_rows() -> pd.DataFrame:
    games, features = build_games(); out = []
    for year in SEASONS:
        train = games[games.season.lt(year)].dropna(subset=["actual_total", "total_line"])
        test = games[games.season.eq(year)].dropna(subset=["actual_total", "total_line"]).copy()
        test["pred"] = model(train, test, features); out.append(test)
    r = pd.concat(out, ignore_index=True); r["edge"] = r.pred - r.total_line
    r = r[r.edge.gt(0)].copy()
    r["result"] = np.where(r.actual_total.eq(r.total_line), "push",
                           np.where(r.actual_total.gt(r.total_line), "win", "loss"))
    r = add_frozen_score(r)
    r["low_total_flag"] = r.total_line.lt(42)
    r["late_season_flag"] = r.week.ge(13)
    r["personnel_absence_flag"] = r.home_personnel.abs().add(r.away_personnel.abs()).ge(0.75)
    r["adverse_weather_flag"] = (r.high_wind_15.fillna(0).gt(0) | r.cold_40.fillna(0).gt(0) | r.precip.fillna(0).gt(0))
    r["edge_2_3_flag"] = r.edge.ge(2) & r.edge.lt(3)
    return r[r.overreaction_score.ge(2)].copy()


def main() -> None:
    r = build_rows()
    print("=== V12 FROZEN SCORE 2+ — 2021 AUTOPSY (DIAGNOSTIC ONLY; NO RETUNING) ===")
    print("=== BY SEASON ===")
    for y in SEASONS: summarize(r[r.season.eq(y)], str(y))
    print("=== SEASON X PHASE ===")
    for y in SEASONS:
        yr=r[r.season.eq(y)]; summarize(yr[yr.week.le(8)], f"{y}_W1_8"); summarize(yr[yr.week.ge(9)], f"{y}_W9_PLUS")
    print("=== 2021 WEEK BANDS ===")
    y=r[r.season.eq(2021)]
    summarize(y[y.week.le(6)], "2021_W1_6"); summarize(y[y.week.between(7,12)], "2021_W7_12"); summarize(y[y.week.ge(13)], "2021_W13_PLUS")
    print("=== 2021 MARKET TOTAL ===")
    for lo,hi in [(0,42),(42,46),(46,50),(50,99)]: summarize(y[y.total_line.ge(lo)&y.total_line.lt(hi)], f"2021_TOTAL_{lo}_{hi}")
    print("=== 2021 FROZEN SCORE ===")
    for s in range(2,6): summarize(y[y.overreaction_score.eq(s)], f"2021_SCORE_{s}")
    print("=== 2021 OAK EDGE ===")
    for lo,hi in [(0,2),(2,3),(3,4),(4,99)]: summarize(y[y.edge.ge(lo)&y.edge.lt(hi)], f"2021_EDGE_{lo}_{hi}")
    print("=== COMPONENT X SEASON ===")
    for c in ["low_total_flag","late_season_flag","personnel_absence_flag","adverse_weather_flag","edge_2_3_flag"]:
        print("COMPONENT", c)
        for yr in SEASONS:
            z=r[r.season.eq(yr)]; summarize(z[z[c]], f"{yr}_{c}_ON"); summarize(z[~z[c]], f"{yr}_{c}_OFF")

if __name__ == "__main__": main()
