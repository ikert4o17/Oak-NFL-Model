"""Audit manifest helpers for Oak production runs."""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def build_run_manifest(
    *,
    season: int,
    week: int,
    frozen_card_path: str | Path,
    frozen_card_existed_before: bool,
    live_preview_path: str | Path | None,
    qb_context_path: str | Path | None,
    injury_context_path: str | Path | None,
    weather_context_path: str | Path | None,
    live_qb: bool,
    live_injuries: bool,
    live_weather: bool,
    freeze: bool,
    generated_at: datetime | None = None,
) -> dict[str, Any]:
    """Return a compact, serializable record of what a production run knew."""
    timestamp = (generated_at or datetime.now(UTC)).astimezone(UTC)
    frozen_action = "reused" if freeze and frozen_card_existed_before else "created_or_updated"
    return {
        "schema_version": 1,
        "generated_at_utc": timestamp.isoformat().replace("+00:00", "Z"),
        "season": season,
        "week": week,
        "frozen_card": {
            "path": str(frozen_card_path),
            "freeze_requested": freeze,
            "existed_before_run": frozen_card_existed_before,
            "action": frozen_action,
        },
        "live_context": {
            "qb_requested": live_qb,
            "injuries_requested": live_injuries,
            "weather_requested": live_weather,
            "preview_path": str(live_preview_path) if live_preview_path else None,
            "qb_context_path": str(qb_context_path) if qb_context_path else None,
            "injury_context_path": str(injury_context_path) if injury_context_path else None,
            "weather_context_path": str(weather_context_path) if weather_context_path else None,
        },
        "github": {
            "sha": os.getenv("GITHUB_SHA"),
            "run_id": os.getenv("GITHUB_RUN_ID"),
            "run_attempt": os.getenv("GITHUB_RUN_ATTEMPT"),
            "workflow": os.getenv("GITHUB_WORKFLOW"),
        },
    }


def write_run_manifest(path: str | Path, manifest: dict[str, Any]) -> Path:
    """Write a deterministic JSON sidecar for the production run."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output
