# Weather validation roadmap

This PR intentionally stops at collection, normalization, and live-preview display. Any future weather-driven point adjustment must be validated separately against historical pregame-available weather data before being allowed to move Oak spread or total predictions.

Candidate validation dimensions include wind/gust thresholds, precipitation, temperature extremes, roof status, and their interaction with totals. Until that work clears Oak's validation standards, missing or extreme weather remains context only and `weather_auto_points` stays zero.
