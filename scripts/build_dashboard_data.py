"""Convert an Oak weekly prediction CSV into website JSON."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


def _clean(value):
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        return value.item()
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", type=Path)
    parser.add_argument("--output", type=Path, default=Path("site/data/latest.json"))
    args = parser.parse_args()

    card = pd.read_csv(args.csv)
    if card.empty:
        raise RuntimeError("weekly prediction card is empty")

    season = int(card.iloc[0]["season"]) if "season" in card else None
    week = int(card.iloc[0]["week"]) if "week" in card else None
    games = [{k: _clean(v) for k, v in row.items()} for row in card.to_dict("records")]
    payload = {
        "season": season,
        "week": week,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "games": games,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    print(f"Saved {args.output} with {len(games)} games")


if __name__ == "__main__":
    main()
