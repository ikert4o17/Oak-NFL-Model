"""Diagnostic-only scoring-regime test for Oak V12 frozen Overreaction Score 2+.

No thresholds in the frozen score are changed. This asks whether lagged league scoring
conditions known before each game explain when the existing Over signal succeeds/fails.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from run_v12_2021_overreaction_autopsy import build_rows, summarize
from run_v12_edge_validation import build_games

SEASONS = range(2019, 2025)


def add_regime_features(r: pd.DataFrame) -> pd.DataFrame:
    games, _ = build_games()
    base = games[games.season.isin(SEASONS)].dropna(subset=["actual_total", "total_line"]).copy()
    weekly = (base.groupby(["season", "week"], as_index=False)
              .agg(league_actual_total=("actual_total", "mean"),
                   league_market_total=("total_line", "mean")))
    weekly = weekly.sort_values(["season", "week"])
    # Strictly lagged: only prior completed weeks can define the regime for a game.
    weekly["actual_4w"] = weekly.groupby("season")["league_actual_total"].transform(
        lambda s: s.shift(1).rolling(4, min_periods=3).mean())
    weekly["actual_8w"] = weekly.groupby("season")["league_actual_total"].transform(
        lambda s: s.shift(1).rolling(8, min_periods=4).mean())
    weekly["market_4w"] = weekly.groupby("season")["league_market_total"].transform(
        lambda s: s.shift(1).rolling(4, min_periods=3).mean())
    weekly["market_8w"] = weekly.groupby("season")["league_market_total"].transform(
        lambda s: s.shift(1).rolling(8, min_periods=4).mean())
    weekly["scoring_delta"] = weekly.actual_4w - weekly.actual_8w
    weekly["market_delta"] = weekly.market_4w - weekly.market_8w
    weekly["market_gap"] = weekly.actual_4w - weekly.market_4w
    return r.merge(weekly[["season","week","actual_4w","actual_8w","market_4w","market_8w",
                           "scoring_delta","market_delta","market_gap"]], on=["season","week"], how="left")


def main() -> None:
    # Rebuild the frozen 2019-22 rows, then extend the same diagnostic construction to 2023-24
    # without changing the score. build_rows supplies 2019-22; later seasons are intentionally
    # reported separately by the existing development analysis, so this script focuses first on
    # whether a pregame regime variable explains the untouched failure pattern.
    r = add_regime_features(build_rows())
    print("=== V12 SCORING REGIME DIAGNOSTIC — FROZEN SCORE 2+; NO RETUNING ===")
    print("Regime inputs are strictly lagged rolling league values (no current-week leakage).")
    valid = r.dropna(subset=["scoring_delta", "market_gap"])
    print("=== BY SEASON, REGIME-ELIGIBLE ===")
    for y in sorted(valid.season.unique()): summarize(valid[valid.season.eq(y)], str(y))

    print("=== SCORING MOMENTUM BANDS (4W actual minus 8W actual) ===")
    for lo, hi in [(-99,-2),(-2,-1),(-1,0),(0,1),(1,2),(2,99)]:
        summarize(valid[valid.scoring_delta.ge(lo)&valid.scoring_delta.lt(hi)], f"SCORING_DELTA_{lo}_{hi}")

    print("=== MARKET GAP BANDS (4W actual minus 4W market) ===")
    for lo, hi in [(-99,-3),(-3,-1.5),(-1.5,0),(0,1.5),(1.5,3),(3,99)]:
        summarize(valid[valid.market_gap.ge(lo)&valid.market_gap.lt(hi)], f"MARKET_GAP_{lo}_{hi}")

    print("=== 2021 REGIME TIMELINE ===")
    y = valid[valid.season.eq(2021)]
    for lo, hi in [(1,7),(7,13),(13,99)]:
        z=y[y.week.ge(lo)&y.week.lt(hi)]
        summarize(z, f"2021_W{lo}_{hi-1 if hi<99 else 'PLUS'}")
        if not z.empty:
            print("REGIME", {"phase": f"W{lo}_{hi-1 if hi<99 else 'PLUS'}",
                  "avg_scoring_delta": round(z.scoring_delta.mean(),3),
                  "avg_market_delta": round(z.market_delta.mean(),3),
                  "avg_market_gap": round(z.market_gap.mean(),3)})

    print("=== NEGATIVE SCORING REGIME X SEASON ===")
    # Diagnostic sign split only, not a proposed production threshold.
    for yr in sorted(valid.season.unique()):
        z=valid[valid.season.eq(yr)]
        summarize(z[z.scoring_delta.lt(0)], f"{yr}_DECLINING")
        summarize(z[z.scoring_delta.ge(0)], f"{yr}_STABLE_OR_RISING")


if __name__ == "__main__":
    main()
