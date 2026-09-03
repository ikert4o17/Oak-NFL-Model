# Weather provider notes

Open-Meteo is used as a replaceable public forecast source for the live context layer. Oak requests hourly temperature, precipitation probability/amount, weather code, sustained wind, and gusts in US-friendly units and stores the nearest forecast hour to scheduled kickoff.

The provider is not model logic. Its output is normalized into Oak context fields and currently carries zero automatic point adjustment.
