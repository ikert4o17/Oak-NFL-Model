# Weather source semantics

- Kickoff times come from nflverse schedules, where `gametime` is Eastern time.
- Roof, surface, stadium, and location metadata are preserved from the schedule when available.
- Hourly forecasts come from Open-Meteo.
- Home-site forecasts use configured NFL home-stadium coordinates.
- Neutral-site games deliberately return missing forecast context until a verified neutral venue coordinate is available.

This conservative behavior prevents a designated home team from accidentally supplying the wrong physical location for an international or other neutral-site game.
