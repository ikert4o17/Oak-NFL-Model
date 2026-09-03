"""Current NFL depth-chart inputs for live Oak context.

nflverse changed depth-chart providers/schema beginning in 2025. This module
normalizes the current ESPN-backed shape while retaining a few legacy aliases so
the production layer does not depend on one raw column spelling.
"""

from __future__ import annotations

import pandas as pd

DEPTH_CHART_URL = (
    "https://github.com/nflverse/nflverse-data/releases/download/"
    "depth_charts/depth_charts_{season}.parquet"
)


def load_depth_charts(season: int) -> pd.DataFrame:
    """Load the latest nflverse depth-chart archive for ``season``."""
    return pd.read_parquet(DEPTH_CHART_URL.format(season=int(season)))


def _first_present(frame: pd.DataFrame, names: tuple[str, ...]) -> str | None:
    return next((name for name in names if name in frame.columns), None)


def normalize_depth_charts(raw: pd.DataFrame) -> pd.DataFrame:
    """Normalize nflverse depth charts to date/team/player/position/rank fields."""
    if raw.empty:
        return pd.DataFrame(
            columns=["snapshot_date", "team", "player_name", "player_id", "position", "depth_rank"]
        )

    team_col = _first_present(raw, ("team", "club_code", "team_abbr"))
    name_col = _first_present(raw, ("player_name", "full_name", "football_name", "name"))
    id_col = _first_present(raw, ("gsis_id", "player_id", "espn_id"))
    position_col = _first_present(raw, ("pos_abb", "pos_name", "pos_grp", "position", "position_group"))
    rank_col = _first_present(raw, ("pos_rank", "depth_chart_order", "depth_team", "depth_position"))
    date_col = _first_present(raw, ("dt", "date", "snapshot_date"))

    required = {
        "team": team_col,
        "player_name": name_col,
        "position": position_col,
        "depth_rank": rank_col,
    }
    missing = [label for label, column in required.items() if column is None]
    if missing:
        raise ValueError(f"depth charts missing required normalized fields: {missing}")

    out = pd.DataFrame(
        {
            "snapshot_date": pd.to_datetime(raw[date_col], errors="coerce") if date_col else pd.NaT,
            "team": raw[team_col].astype("string").str.upper(),
            "player_name": raw[name_col].astype("string"),
            "player_id": raw[id_col].astype("string") if id_col else pd.Series(pd.NA, index=raw.index, dtype="string"),
            "position": raw[position_col].astype("string").str.upper(),
            "depth_rank": pd.to_numeric(raw[rank_col], errors="coerce"),
        }
    )
    return out.dropna(subset=["team", "player_name", "position"]).reset_index(drop=True)


def expected_starting_qbs(raw: pd.DataFrame) -> pd.DataFrame:
    """Return the most recent QB1 for every team in a depth-chart archive."""
    depth = normalize_depth_charts(raw)
    if depth.empty:
        return pd.DataFrame(
            columns=["team", "expected_qb_name", "expected_qb_id", "depth_chart_date", "depth_rank"]
        )

    position = depth["position"].fillna("")
    qbs = depth.loc[position.eq("QB") | position.str.contains("QUARTERBACK", regex=False)].copy()
    if qbs.empty:
        return pd.DataFrame(
            columns=["team", "expected_qb_name", "expected_qb_id", "depth_chart_date", "depth_rank"]
        )

    # Keep only each team's newest dated snapshot. If the source has no date,
    # all rows remain eligible and rank determines QB1.
    if qbs["snapshot_date"].notna().any():
        newest = qbs.groupby("team")["snapshot_date"].transform("max")
        qbs = qbs.loc[qbs["snapshot_date"].eq(newest)]

    qbs["depth_rank"] = qbs["depth_rank"].fillna(9999)
    qbs = qbs.sort_values(["team", "depth_rank", "player_name"])
    qbs = qbs.drop_duplicates("team", keep="first")
    return qbs.rename(
        columns={
            "player_name": "expected_qb_name",
            "player_id": "expected_qb_id",
            "snapshot_date": "depth_chart_date",
        }
    )[["team", "expected_qb_name", "expected_qb_id", "depth_chart_date", "depth_rank"]].reset_index(drop=True)
