"""Build rolling Oak survivor paths and website JSON."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from oak_nfl.data.schedules import load_schedules
from oak_nfl.survivor import future_value_by_team, pick_two, rank_candidates


def _load_json(path: Path, default):
    if not path.exists():
        return default
    return json.loads(path.read_text())


def _grade_pick(results: pd.DataFrame, season: int, week: int, team: str) -> str:
    rows = results.loc[(results.season.eq(season)) & (results.week.eq(week)) & ((results.home_team.eq(team)) | (results.away_team.eq(team)))]
    if rows.empty:
        return "PENDING"
    row = rows.iloc[0]
    if pd.isna(row.get("home_score")) or pd.isna(row.get("away_score")):
        return "PENDING"
    hs, as_ = float(row.home_score), float(row.away_score)
    won = hs > as_ if row.home_team == team else as_ > hs
    return "WIN" if won else "LOSS"


def _advance_path(path_state: dict, season: int, week: int, results: pd.DataFrame) -> None:
    rec = path_state.get("current")
    if not rec or int(rec.get("week", week)) >= week:
        return
    team = rec["primary"]["team"]
    result = _grade_pick(results, season, int(rec["week"]), team)
    if result == "PENDING":
        return
    path_state.setdefault("history", []).append({"week": int(rec["week"]), "team": team, "result": result})
    path_state.setdefault("used", []).append(team)
    path_state["alive"] = bool(path_state.get("alive", True) and result == "WIN")
    path_state["current"] = None


def _reason(path: str, primary: dict, alternate: dict | None) -> str:
    if path == "a":
        if primary["future_value"] < 0.65:
            return "Strong current spot with limited premium future value — a good team to use now."
        return "Best balance of current survival strength and preserving stronger future options."
    if alternate and primary["win_confidence"] - alternate["win_confidence"] >= 0.05:
        return "Oak's strongest current-week survival position."
    return "Highest current-week survival confidence, with future value used only as a tiebreaker."


def _as_pick(c):
    return {
        "team": c.team, "opponent": c.opponent, "location": c.location,
        "oak_margin": round(c.margin, 2),
        "survival_confidence": round(c.win_confidence, 4),
        "future_value": round(c.future_value, 4),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("card", type=Path)
    p.add_argument("--power", type=Path, default=Path("site/data/power.json"))
    p.add_argument("--results", type=Path, default=Path("data/results/oak_results.csv"))
    p.add_argument("--state", type=Path, default=Path("data/survivor/state.json"))
    p.add_argument("--output", type=Path, default=Path("site/data/survivor.json"))
    args = p.parse_args()

    card = pd.read_csv(args.card)
    season, week = int(card.iloc[0].season), int(card.iloc[0].week)
    power = _load_json(args.power, {"teams": []})
    ratings = {x["team"]: float(x["rating"]) for x in power.get("teams", [])}
    if len(ratings) != 32:
        raise RuntimeError("survivor requires 32 Oak power ratings")
    schedule = load_schedules()
    schedule = schedule.loc[schedule.season.eq(season)].copy()
    results = pd.read_csv(args.results) if args.results.exists() else pd.DataFrame(columns=["season","week","home_team","away_team"])

    state = _load_json(args.state, {
        "season": season,
        "path_a": {"alive": True, "used": [], "history": [], "current": None},
        "path_b": {"alive": True, "used": [], "history": [], "current": None},
    })
    if int(state.get("season", season)) != season:
        state = {"season": season, "path_a": {"alive": True, "used": [], "history": [], "current": None}, "path_b": {"alive": True, "used": [], "history": [], "current": None}}

    future = future_value_by_team(schedule, ratings, current_week=week)
    paths = {}
    for key, code in (("path_a", "a"), ("path_b", "b")):
        ps = state[key]
        _advance_path(ps, season, week, results)
        used = set(ps.get("used", []))
        ranked = rank_candidates(card, future, used)
        picks = pick_two(ranked, code) if ps.get("alive", True) else []
        if picks:
            primary = _as_pick(picks[0])
            alternate = _as_pick(picks[1]) if len(picks) > 1 else None
            ps["current"] = {"week": week, "primary": primary, "alternate": alternate}
            reason = _reason(code, primary, alternate)
        else:
            primary = alternate = None
            reason = "Path eliminated." if not ps.get("alive", True) else "No eligible positive-margin survivor pick."
        paths[key] = {
            "name": "Best Survival Path" if code == "a" else "Best Pick Now",
            "alive": bool(ps.get("alive", True)), "primary": primary, "alternate": alternate,
            "reason": reason, "used_teams": ps.get("used", []), "history": ps.get("history", []),
        }

    state["season"] = season
    args.state.parent.mkdir(parents=True, exist_ok=True)
    args.state.write_text(json.dumps(state, indent=2) + "\n")
    payload = {
        "season": season, "week": week,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "method_note": "Survival confidence is a ranking transform of Oak projected margin, not a calibrated betting probability.",
        "paths": paths,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"Saved survivor recommendations for {season} Week {week}")


if __name__ == "__main__":
    main()
