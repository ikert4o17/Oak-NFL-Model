"""Diagnostic autopsy for the frozen V12 Overreaction Score 2+ rule.

This script is intentionally diagnostic only: it does not retune thresholds or
change the frozen rule. It compares the 2021 failure season with 2019, 2020,
and 2022 across pre-specified buckets so we can identify regime sensitivity.
"""

from collections import defaultdict

from scripts.run_v12_overreaction_oos_validation import build_rows

SEASONS = (2019, 2020, 2021, 2022)


def roi(w, l):
    return ((w * 100.0 - l * 110.0) / ((w + l) * 110.0) * 100.0) if w + l else 0.0


def summarize(rows, label):
    w = sum(r["result"] == "W" for r in rows)
    l = sum(r["result"] == "L" for r in rows)
    p = sum(r["result"] == "P" for r in rows)
    n = w + l
    print(f"{label:32s} {w:3d}-{l:3d}-{p:2d}  WR={100*w/n if n else 0:5.1f}%  ROI={roi(w,l):6.1f}%  N={len(rows):3d}")


def bucket(rows, name, fn):
    groups = defaultdict(list)
    for r in rows:
        groups[fn(r)].append(r)
    print(f"\n{name}")
    for key in sorted(groups, key=str):
        summarize(groups[key], str(key))


def main():
    rows = [r for r in build_rows(SEASONS) if r["direction"] == "OVER" and r["overreaction_score"] >= 2]
    print("=" * 88)
    print("V12 FROZEN SCORE 2+ — 2021 AUTOPSY (DIAGNOSTIC ONLY; NO RETUNING)")
    print("=" * 88)

    bucket(rows, "BY SEASON", lambda r: r["season"])
    bucket(rows, "BY WEEK BAND", lambda r: "W1-6" if r["week"] <= 6 else ("W7-12" if r["week"] <= 12 else "W13+"))
    bucket(rows, "BY MARKET TOTAL", lambda r: "<42" if r["market_total"] < 42 else ("42-46" if r["market_total"] < 46 else ("46-50" if r["market_total"] < 50 else "50+")))
    bucket(rows, "BY FROZEN SCORE", lambda r: r["overreaction_score"])
    bucket(rows, "BY OAK OVER EDGE", lambda r: "<2" if r["edge"] < 2 else ("2-3" if r["edge"] < 3 else ("3-4" if r["edge"] < 4 else "4+")))

    # Compare 2021 vs the surrounding untouched seasons for each frozen component.
    for component in ("low_total_flag", "late_season_flag", "personnel_absence_flag", "adverse_weather_flag", "edge_2_3_flag"):
        print(f"\nCOMPONENT: {component}")
        for season in SEASONS:
            sr = [r for r in rows if r["season"] == season]
            summarize([r for r in sr if r.get(component)], f"{season} flag=1")
            summarize([r for r in sr if not r.get(component)], f"{season} flag=0")

    # Early/late season x year is the cleanest regime-transition diagnostic.
    print("\nSEASON x PHASE")
    for season in SEASONS:
        sr = [r for r in rows if r["season"] == season]
        summarize([r for r in sr if r["week"] <= 8], f"{season} W1-8")
        summarize([r for r in sr if r["week"] >= 9], f"{season} W9+")


if __name__ == "__main__":
    main()
