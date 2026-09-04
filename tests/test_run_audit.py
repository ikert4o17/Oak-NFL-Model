from datetime import UTC, datetime

from oak_nfl.run_audit import build_run_manifest, write_run_manifest


def test_manifest_records_frozen_reuse_and_context_paths(tmp_path):
    generated_at = datetime(2026, 9, 3, 22, 0, tzinfo=UTC)
    manifest = build_run_manifest(
        season=2026,
        week=1,
        frozen_card_path="data/predictions/oak_2026_week_1.csv",
        frozen_card_existed_before=True,
        live_preview_path="data/previews/oak_2026_week_1_live.csv",
        qb_context_path="data/context/oak_2026_week_1_qb.csv",
        injury_context_path="data/context/oak_2026_week_1_injuries.csv",
        weather_context_path="data/context/oak_2026_week_1_weather.csv",
        live_qb=True,
        live_injuries=True,
        live_weather=True,
        freeze=True,
        generated_at=generated_at,
    )

    assert manifest["generated_at_utc"] == "2026-09-03T22:00:00Z"
    assert manifest["frozen_card"]["action"] == "reused"
    assert manifest["live_context"]["weather_requested"] is True

    output = write_run_manifest(tmp_path / "manifest.json", manifest)
    text = output.read_text(encoding="utf-8")
    assert '"schema_version": 1' in text
    assert '"action": "reused"' in text


def test_manifest_marks_new_frozen_card_creation():
    manifest = build_run_manifest(
        season=2026,
        week=1,
        frozen_card_path="card.csv",
        frozen_card_existed_before=False,
        live_preview_path=None,
        qb_context_path=None,
        injury_context_path=None,
        weather_context_path=None,
        live_qb=False,
        live_injuries=False,
        live_weather=False,
        freeze=True,
        generated_at=datetime(2026, 9, 3, tzinfo=UTC),
    )
    assert manifest["frozen_card"]["action"] == "created_or_updated"
