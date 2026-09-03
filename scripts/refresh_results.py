"""Refresh Oak's public results ledger from frozen prediction snapshots."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from oak_nfl.data.schedules import load_schedules
from oak_nfl.results import grade_predictions, summarize_results
from oak_nfl.results_refresh import finals_from_schedules, load_frozen_predictions


def _clean(value):
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        return value.item()
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions-dir", type=Path, default=Path("data/predictions"))
    parser.add_argument("--ledger", type=Path, default=Path("data/results/oak_results.csv"))
    parser.add_argument("--output", type=Path, default=Path("site/data/results.json"))
    parser.add_argument("--refresh-schedule", action="store_true")
    args = parser.parse_args()

    predictions = load_frozen_predictions(args.predictions_dir)
    if predictions.empty:
        print("No frozen weekly predictions found; leaving results empty")
        return

    schedules = load_schedules(refresh=args.refresh_schedule)
    finals = finals_from_schedules(schedules)
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
