"""Generate Oak's unified weekly spread and total prediction card."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from oak_nfl.data.depth_charts import load_depth_charts
from oak_nfl.data.espn_injuries import fetch_espn_injuries
from oak_nfl.data.injuries import CANONICAL_COLUMNS
from oak_nfl.data.nflverse import load_pbp
from oak_nfl.data.schedules import load_next_slate, load_week
from oak_nfl.injury_context import build_game_injury_context
from oak_nfl.live_qb import build_live_qb_inputs
from oak_nfl.weekly import run_weekly_predictions


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--season", type=int)
    parser.add_argument("--week", type=int)
    parser.add_argument("--auto", action="store_true")
    parser.add_argument("--refresh-schedule", action="store_true")
    parser.add_argument(
        "--live-qb",
        action="store_true",
        help="Apply current nflverse depth-chart QB context",
    )
    parser.add_argument(
        "--live-injuries",
        action="store_true",
        help="Add informational ESPN injury-report context without changing model points",
    )
    parser.add_argument(
        "--freeze",
        action="store_true",
        help="Never overwrite an existing season/week snapshot",
    )
    parser.add_argument("--output-dir", default="data/predictions")
    parser.add_argument("--context-output-dir", default="data/context")
    parser.add_argument("--preview-output-dir", default="data/previews")
    args = parser.parse_args()

    if args.auto:
        slate = load_next_slate(refresh=args.refresh_schedule)
    else:
        if args.season is None or args.week is None:
            parser.error("provide --season and --week, or use --auto")
        slate = load_week(args.season, args.week, refresh=args.refresh_schedule)

    if slate.empty:
        raise RuntimeError("selected slate has no games")
    season = int(slate.iloc[0]["season"])
    week = int(slate.iloc[0]["week"])

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"oak_{season}_week_{week}.csv"

    # Live context is refreshed even after an official card has been frozen. That
    # gives Oak a current day-by-day preview without rewriting the auditable
    # published snapshot used for results grading.
    pbp: pd.DataFrame | None = None
    live_card: pd.DataFrame | None = None
    qb_inputs: pd.DataFrame | None = None
    live_context_requested = args.live_qb or args.live_injuries

    if live_context_requested:
        pbp = _load_history(season)
        context_dir = Path(args.context_output_dir)
        context_dir.mkdir(parents=True, exist_ok=True)

        if args.live_qb:
            depth = load_depth_charts(season)
            qb_inputs = build_live_qb_inputs(pbp, slate, depth)
            qb_output = context_dir / f"oak_{season}_week_{week}_qb.csv"
            qb_inputs.to_csv(qb_output, index=False)
            print(f"Saved live QB context {qb_output}")

        live_card = run_weekly_predictions(pbp, slate, qb_inputs=qb_inputs)

        if args.live_injuries:
            try:
                injury_report = fetch_espn_injuries(season=season, week=week)
            except Exception as exc:
                # Injury context is supplemental. A provider outage must never
                # block Oak's core prediction run or fabricate an adjustment.
                print(f"Live injury context unavailable; continuing safely: {exc}")
                injury_report = pd.DataFrame(columns=CANONICAL_COLUMNS)

            injury_output = context_dir / f"oak_{season}_week_{week}_injuries.csv"
            injury_report.to_csv(injury_output, index=False)
            print(f"Saved live injury context {injury_output}")

            game_injuries = build_game_injury_context(injury_report, slate)
            injury_cols = [
                c for c in game_injuries.columns if c not in {"home_team", "away_team"}
            ]
            live_card = live_card.merge(
                game_injuries[injury_cols], on="game_id", how="left", validate="one_to_one"
            )

        preview_dir = Path(args.preview_output_dir)
        preview_dir.mkdir(parents=True, exist_ok=True)
        preview_output = preview_dir / f"oak_{season}_week_{week}_live.csv"
        live_card.to_csv(preview_output, index=False)
        print(f"Saved current adjusted preview {preview_output}")

    if args.freeze and output.exists():
        card = pd.read_csv(output)
        print(f"Using frozen weekly snapshot {output}")
    else:
        if live_card is not None:
            card = live_card
        else:
            if pbp is None:
                pbp = _load_history(season)
            card = run_weekly_predictions(pbp, slate)
        card.to_csv(output, index=False)
        print(f"Saved {output}")

    display = [
        "away_team",
        "home_team",
        "predicted_home_margin",
        "spread_line",
        "spread_edge",
        "spread_side",
        "predicted_total",
        "total_line",
        "total_edge",
        "total_side",
    ]
    print(card[[c for c in display if c in card.columns]].to_string(index=False))


if __name__ == "__main__":
    main()
