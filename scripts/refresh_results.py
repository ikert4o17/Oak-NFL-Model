"""Refresh Oak's public results ledger from frozen prediction snapshots."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from oak_nfl.data.schedules import load_schedules
from oak_nfl.results import grade_predictions, summarize_results


PREDICTION_GLOB = "oak_*_week_*.csv"


def _clean(value):
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        return value.item()
    return value


def _load_frozen_predictions(directory: Path) -> pd.DataFrame:
    files = sorted(directory.glob(PREDICTION_GLOB))
    if not files:
        return pd.DataFrame()
    frames = [pd.read_csv(path) for path in files]
    out = pd.concat(frames, ignore_index=True)
    return out.drop_duplicates("game_id", keep="first")


def _finals_from_schedules(schedules: pd.DataFrame) -> pd.DataFrame:
    """Extract final scores and verified nflverse closing lines.

    nflverse documents schedule ``spread_line`` and ``total_line`` as closing
    lines sourced from Pro-Football-Reference. ``load_schedules`` already
    converts spread_line to Oak's conventional home-team sportsbook sign.
    """
    required = {"game_id", "home_score", "away_score"}
    missing = required.difference(schedules.columns)
    if missing:
        raise ValueError(f"schedule data missing required result columns: {sorted(missing)}")

    cols = ["game_id", "home_score", "away_score"]
    for col in ("spread_line", "total_line"):
        if col in schedules.columns:
            cols.append(col)
    finals = schedules[cols].copy()
    finals = finals.loc[finals["home_score"].notna() & finals["away_score"].notna()]
    finals = finals.rename(
        columns={
            "spread_line": "closing_spread_line",
            "total_line": "closing_total_line",
        }
    )
    return finals.drop_duplicates("game_id", keep="last")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions-dir", type=Path, default=Path("data/predictions"))
    parser.add_argument("--ledger", type=Path, default=Path("data/results/oak_results.csv"))
    parser.add_argument("--output", type=Path, default=Path("site/data/results.json"))
    parser.add_argument("--refresh-schedule", action="store_true")
    args = parser.parse_args()

    predictions = _load_frozen_predictions(args.predictions_dir)
    if predictions.empty:
        print("No frozen weekly predictions found; leaving results empty")
        return

    schedules = load_schedules(refresh=args.refresh_schedule)
    finals = _finals_from_schedules(schedules)
    graded = grade_predictions(predictions, finals)
    completed = graded.loc[graded["final_home_margin"].notna()].copy()
    weekly = summarize_results(completed) if not completed.empty else pd.DataFrame()

    args.ledger.parent.mkdir(parents=True, exist_ok=True)
    graded.to_csv(args.ledger, index=False)

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "closing_line_source": "nflverse schedules (Pro-Football-Reference)",
        "weekly": [{k: _clean(v) for k, v in row.items()} for row in weekly.to_dict("records")],
        "games": [{k: _clean(v) for k, v in row.items()} for row in graded.to_dict("records")],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    print(f"Saved {args.output}: {len(completed)} completed of {len(graded)} frozen games")


if __name__ == "__main__":
    main()
