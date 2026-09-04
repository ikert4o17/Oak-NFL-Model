# Oak live weather context

Oak treats pregame weather as supplemental, auditable context unless and until a weather effect is separately validated.

The live provider uses Open-Meteo hourly forecasts near scheduled kickoff. nflverse schedule metadata supplies roof, surface, stadium, location, and Eastern-time kickoff semantics. Home-site games use configured home-stadium coordinates. Neutral-site games fail safe to missing weather rather than silently using the designated home team's stadium.

The live weather snapshot includes temperature, sustained wind, gusts, precipitation probability, precipitation amount, provider weather code, forecast timestamp, roof/surface/stadium metadata, and source availability. `weather_auto_points` is always `0.0` in this layer.

Provider failures, unsupported venues, forecasts outside the provider horizon, or malformed responses do not block Oak's weekly prediction run and do not change V5/V12 or validated QB adjustments. Weather context refreshes in the live preview while an existing frozen prediction card remains immutable.
