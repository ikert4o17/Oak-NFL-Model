# Oak NFL Model

NFL forecasting and analytics project for preseason ratings, in-season team strength, spreads, totals, injuries, backtesting, and a GitHub-hosted website.

## Core principles

- Point-in-time historical data only; no future-data leakage.
- Walk-forward backtesting.
- Separate spread and total models.
- Preseason priors that decay as current-season information accumulates.
- Explicit quarterback and injury handling.
- Immutable archived predictions after kickoff.
- Prefer simple models unless more complex approaches prove better out of sample.

## Initial roadmap

1. Repository and Python project foundation.
2. Historical nflverse data ingestion.
3. Baseline offensive and defensive efficiency ratings.
4. Preseason priors and home-field adjustment.
5. Spread and total baseline models.
6. Walk-forward backtesting and model evaluation.
7. Opponent adjustment, recency weighting, QB, roster continuity, injuries, weather, and matchup features.
8. Automated weekly predictions and GitHub Pages dashboard.
