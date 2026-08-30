import numpy as np
import pandas as pd

from oak_nfl.data.schedules import _normalize_market_lines


def test_nflverse_spread_line_is_normalized_to_sportsbook_convention() -> None:
    raw = pd.DataFrame(
        {
            "home_team": ["JAX", "DAL", "KC"],
            "spread_line": [7.5, -2.5, np.nan],
        }
    )

    out = _normalize_market_lines(raw)

    assert out.loc[0, "spread_line"] == -7.5
    assert out.loc[1, "spread_line"] == 2.5
    assert np.isnan(out.loc[2, "spread_line"])
    assert raw.loc[0, "spread_line"] == 7.5
