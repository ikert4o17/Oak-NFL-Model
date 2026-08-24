import pytest

from oak_nfl.data.nflverse import pbp_url, roster_url


def test_pbp_url() -> None:
    assert pbp_url(2025).endswith("/pbp/play_by_play_2025.parquet")


def test_roster_url() -> None:
    assert roster_url(2025).endswith("/rosters/roster_2025.parquet")


def test_pbp_rejects_pre_nflfastr_seasons() -> None:
    with pytest.raises(ValueError):
        pbp_url(1998)
