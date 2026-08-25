# Oak V11 — Weather / Totals Validation Plan

## Goal
Test whether pregame weather adds out-of-sample signal to Oak's game-total projections without changing the validated spread model merely because football intuition says weather should matter.

## Order of testing
1. Wind only.
2. Absolute extreme cold / heat.
3. Team temperature acclimation / climate mismatch.
4. Indoor-team to outdoor-weather transitions.
5. Precipitation when a reliable historical pregame source is available.
6. Weather combinations that survived individually.

## Leakage controls
Every historical weather observation must represent conditions known or reasonably forecastable before kickoff. Team climate baselines use only games played before the matchup being predicted. The current game's conditions can be compared with that prior baseline, but can never be used to construct the baseline itself.

## Feature design
Prefer continuous severity features over arbitrary point penalties. V11 begins with wind above 10 mph and 15 mph, degrees below 32 F, degrees above 85 F, and indoor/outdoor status. Indoor or closed-roof games receive zero outdoor weather severity.

### Team acclimation
For each team, maintain a rolling pregame climate profile derived from its prior home games. Outdoor home games contribute historical temperature; all prior home games contribute to an indoor-home share. For each matchup, calculate:

- home and away expected home-climate temperature,
- cold shock: how many degrees colder the game is than the team's prior home climate,
- heat shock: how many degrees hotter the game is than the team's prior home climate,
- relative cold/heat shock between away and home teams,
- indoor-home team transitioning to an outdoor game.

These are hypotheses to validate, not assumed point adjustments. If warm-weather teams do not show a reproducible penalty in cold games, the feature does not survive V11.

## Historical data strategy
nflverse game/schedule data includes game ID, teams, scores, roof status, temperature, wind and closing total. That is sufficient to start the historical wind/temperature/acclimation diagnostic. Precipitation will remain provider-agnostic until a reliable historical pregame feed is selected.

## Validation
The first diagnostic uses seasons through 2022 for fitting and 2023-2025 as holdout. Because Oak does not yet have a frozen mature totals model, V11 initially tests weather against residual scoring beyond the closing market total. This is deliberately stringent because the closing market already incorporates some weather information. The final Oak totals model will later test the same feature families without relying on the market line as its forecast.

Evaluate total-points MAE and RMSE, performance by season, wind buckets, temperature-shock buckets, and whether gains are concentrated in a tiny number of extreme games.

## Promotion rule
No weather feature reaches production from in-sample fit alone. It must improve out-of-sample totals error and remain directionally sensible across seasons. If wind is robust but temperature acclimation is not, V11 should keep wind and drop acclimation rather than forcing the football narrative into the model.
