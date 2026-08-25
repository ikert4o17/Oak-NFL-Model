# Oak V11 — Weather / Totals Validation Plan

## Goal
Test whether pregame weather adds out-of-sample signal to Oak's game-total projections without changing the validated spread model merely because football intuition says weather should matter.

## Order of testing
1. Wind only.
2. Precipitation only.
3. Extreme cold / heat only.
4. Outdoor vs dome / closed-roof handling.
5. Weather combinations that survived individually.

## Leakage controls
Every historical weather observation must represent conditions known or reasonably forecastable before kickoff. Do not use postgame summaries that encode conditions occurring only after kickoff as if they were known pregame.

## Feature design
Prefer continuous severity features over arbitrary point penalties. V11 begins with wind above 10 mph and 15 mph, precipitation amount, degrees below 32 F, degrees above 85 F, and an indoor/outdoor flag. Indoor or closed-roof games receive zero outdoor weather severity.

## Validation
Keep the established historical train/holdout split. Evaluate total-points MAE and RMSE first. Also report calibration by predicted-total bucket and the direction/magnitude of each weather effect. A feature is promoted only if holdout improvement is repeatable and not driven by a tiny number of extreme games.

## Data strategy
Use a historical game-weather source with kickoff-time or pregame observations/forecasts and explicit venue/roof information. Normalize provider-specific columns before model fitting so the live-season provider can later be swapped without changing model logic.

## Promotion rule
No weather feature reaches production from in-sample fit alone. It must improve out-of-sample totals error and remain directionally sensible across seasons. If wind is the only robust signal, V11 should remain wind-only rather than forcing precipitation or temperature into the model.
