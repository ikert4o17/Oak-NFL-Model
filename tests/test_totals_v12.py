import pandas as pd

from oak_nfl.totals_v12 import _append_slate_dummy_team_games


def test_completed_slate_game_does_not_get_duplicate_dummy_rows():
    team_games = pd.DataFrame(
        [
            {
                "game_id": "2026_01_SF_LA",
                "season": 2026,
                "week": 1,
                "posteam": "SF",
                "defteam": "LA",
                "epa_per_play": 0.10,
            },
            {
                "game_id": "2026_01_SF_LA",
                "season": 2026,
                "week": 1,
                "posteam": "LA",
                "defteam": "SF",
                "epa_per_play": -0.05,
            },
        ]
    )
    slate = pd.DataFrame(
        [
            {
                "game_id": "2026_01_SF_LA",
                "season": 2026,
                "week": 1,
                "home_team": "LA",
                "away_team": "SF",
            },
            {
                "game_id": "2026_01_CLE_JAX",
                "season": 2026,
                "week": 1,
                "home_team": "JAX",
                "away_team": "CLE",
            },
        ]
    )

    out = _append_slate_dummy_team_games(team_games, slate)

    keys = list(out[["game_id", "posteam"]].itertuples(index=False, name=None))
    assert len(keys) == len(set(keys))
    assert keys.count(("2026_01_SF_LA", "SF")) == 1
    assert keys.count(("2026_01_SF_LA", "LA")) == 1
    assert ("2026_01_CLE_JAX", "CLE") in keys
    assert ("2026_01_CLE_JAX", "JAX") in keys
