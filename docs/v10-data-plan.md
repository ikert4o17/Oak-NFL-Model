# Oak V10 — Non-QB Personnel Validation Plan

## Goal
Prove whether non-QB player availability adds out-of-sample spread signal beyond Oak's accepted team/QB layers.

## Historical data strategy
Use nflverse sources where coverage is reliable:

- Injury/practice reports through 2024.
- Weekly snap counts to estimate actual role/importance and confirm missed/limited games.
- Weekly/depth-chart roster data to map players to teams and position groups.
- Participation data as a secondary usage/role source where available.

Because nflverse's injury source ended after 2024, V10's primary injury-report validation window will stop at 2024 rather than silently treating missing 2025 reports as healthy players. A future live provider can be plugged into the same normalized availability schema.

## Leakage controls
Player value for game G must use only games before G. Current-game snaps, stats, and participation may be used only as outcome/validation information, never as a pregame feature.

## Player value prototype
Estimate pregame importance from trailing prior-game snap share, with sample-size regression and depth-chart role where available. Do not use QB; QB is handled by V9.

## Position groups
OT, IOL, WR, TE, RB, EDGE, IDL, LB, CB, S.

## Tournament
Test groups separately before combining:

1. OL availability
2. WR/TE availability
3. RB availability
4. pass-rush availability (EDGE/IDL)
5. LB availability
6. secondary availability (CB/S)
7. all non-QB personnel

For each group, test conservative scaling and team-level caps. Promote only if the holdout improves spread MAE/RMSE without an unacceptable winner-accuracy tradeoff.

## Live-season architecture
The model should support multiple weekly snapshots (early week, practice-report updates, final inactive update). Store timestamps and component adjustments so every projection move is auditable.
