"""Build the dashboard's V5-derived neutral-field power ratings."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from oak_nfl.data.nflverse import load_pbp
from oak_nfl.features import build_team_game_features
from oak_nfl.power import build_power_ratings
from oak_nfl.ratings.v5 import build_v5_pregame_ratings


def _load_history(season: int, start: int = 2014) -> pd.DataFrame:
    frames = []
    for year in range(start, season + 1):
        try:
            frames.append(load_pbp(year))
        except Exception:
            if year != season:
                raise
    if not frames:
        raise RuntimeError("no historical play-by-play data available")
    return pd.concat(frames, ignore_index=True)


def _snapshot(team_games: pd.DataFrame, season: int, week: int) -> pd.DataFrame:
    teams = sorted(set(team_games["posteam"].dropna()) | set(team_games["defteam"].dropna()))
    metric_cols = [c for c in team_games.columns if c not in {"game_id", "season", "week", "posteam", "defteam"}]
    dummies = []
    for team in teams:
        row = {"game_id": f"POWER_{season}_{week}_{team}", "season": season, "week": week, "posteam": team, "defteam": team}
        row.update({c: np.nan for c in metric_cols})
        dummies.append(row)
    ratings = build_v5_pregame_ratings(pd.concat([team_games, pd.DataFrame(dummies)], ignore_index=True))
    return ratings.loc[(ratings["season"].eq(season)) & (ratings["week"].eq(week))].drop_duplicates("team")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--week", type=int, required=True)
    parser.add_argument("--output", default="site/data/power.json")
    args = parser.parse_args()

    pbp = _load_history(args.season)
    team_games = build_team_game_features(pbp)
    current = build_power_ratings(_snapshot(team_games, args.season, args.week))

    previous = None
    if args.week > 1:
        previous = build_power_ratings(_snapshot(team_games, args.season, args.week - 1))[["team", "rank"]].rename(columns={"rank": "previous_rank"})
        current = current.merge(previous, on="team", how="left")
    else:
        current["previous_rank"] = np.nan

    current["movement"] = current["previous_rank"] - current["rank"]
    rows = []
    for row in current.itertuples(index=False):
        rows.append({
            "rank": int(row.rank),
            "team": row.team,
            "rating": round(float(row.rating), 2),
            "movement": None if pd.isna(row.movement) else int(row.movement),
            "epa_points": round(float(row.epa_points), 2),
            "success_points": round(float(row.success_points), 2),
            "explosive_points": round(float(row.explosive_points), 2),
        })
    payload = {"season": args.season, "week": args.week, "generated_at": datetime.now(timezone.utc).isoformat(), "teams": rows}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"Saved {len(rows)} power ratings to {output}")


if __name__ == "__main__":
    main()
