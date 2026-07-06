"""
Weather tool — Open-Meteo (free, keyless; nothing to store, fits the no-secrets
rule). Geocodes a city name to coordinates, then fetches current conditions and
a short daily forecast, returned as structured data.

The source is deliberately hidden behind the Tool interface so it can be swapped
later without touching the agent. Network failure returns `{"error": ...}` —
never raises — so the model can give an honest "couldn't reach the weather
service" answer instead of the turn crashing.
"""
from __future__ import annotations

import logging

import httpx

from core.constants import (
    LOGGER_TOOLS,
    OPEN_METEO_FORECAST_URL,
    OPEN_METEO_GEOCODE_URL,
    OPEN_METEO_TIMEOUT_S,
    OPEN_METEO_UNITS,
    WEATHER_FORECAST_DAYS,
    WMO_WEATHER_CODES,
)
from core.tools.base import Tool

logger = logging.getLogger(LOGGER_TOOLS)


def _conditions(code) -> str:
    return WMO_WEATHER_CODES.get(code, f"weather code {code}")


class WeatherTool(Tool):
    name = "get_weather"
    description = (
        "Get the current weather and a short forecast. "
        "Omit city to use the user's default location."
    )
    parameters = {
        "type": "object",
        "properties": {
            "city": {
                "type": "string",
                "description": "City name, e.g. 'Denver' or 'Buffalo, NY'. "
                               "Omit for the user's default location.",
            }
        },
        "required": [],
    }
    status = "checking the weather"

    def __init__(self, default_location: str) -> None:
        self._default = default_location
        # Geocoding results are stable — cache per place name for the session.
        self._coords: dict[str, dict] = {}

    async def run(self, city: str = "") -> dict:
        place = (city or self._default).strip()
        if not place:
            return {"error": "no city given and no default location configured"}
        try:
            location = await self._geocode(place)
            if location is None:
                return {"error": f"no location found for {place!r}"}
            forecast = await self._forecast(location)
        except Exception as exc:
            logger.warning("weather lookup failed for %r: %s", place, exc)
            return {"error": f"weather service unreachable: {exc}"}
        return {"location": location["label"], **forecast}

    async def _geocode(self, place: str) -> dict | None:
        key = place.lower()
        if key not in self._coords:
            hit = None
            # Open-Meteo's geocoder matches plain names, not "City, ST" strings —
            # fall back to the part before the comma (top-ranked match wins; the
            # returned label states which city was picked, so it's transparent).
            candidates = [place]
            if "," in place:
                candidates.append(place.split(",")[0].strip())
            async with httpx.AsyncClient(timeout=OPEN_METEO_TIMEOUT_S) as client:
                for name in candidates:
                    resp = await client.get(
                        OPEN_METEO_GEOCODE_URL,
                        params={"name": name, "count": 1, "language": "en", "format": "json"},
                    )
                    resp.raise_for_status()
                    results = resp.json().get("results") or []
                    if results:
                        hit = results[0]
                        break
            if hit is None:
                return None
            label = ", ".join(
                part for part in (hit.get("name"), hit.get("admin1"), hit.get("country"))
                if part
            )
            self._coords[key] = {
                "label": label,
                "latitude": hit["latitude"],
                "longitude": hit["longitude"],
            }
        return self._coords.get(key)

    async def _forecast(self, location: dict) -> dict:
        async with httpx.AsyncClient(timeout=OPEN_METEO_TIMEOUT_S) as client:
            resp = await client.get(
                OPEN_METEO_FORECAST_URL,
                params={
                    "latitude": location["latitude"],
                    "longitude": location["longitude"],
                    "current": "temperature_2m,apparent_temperature,"
                               "relative_humidity_2m,precipitation,weather_code,"
                               "wind_speed_10m",
                    "daily": "weather_code,temperature_2m_max,temperature_2m_min,"
                             "precipitation_probability_max",
                    "forecast_days": WEATHER_FORECAST_DAYS,
                    "timezone": "auto",
                    **OPEN_METEO_UNITS,
                },
            )
            resp.raise_for_status()
        data = resp.json()

        current = data.get("current", {})
        daily = data.get("daily", {})
        days = []
        for i, date in enumerate(daily.get("time", [])):
            days.append({
                "date": date,
                "high_f": daily["temperature_2m_max"][i],
                "low_f": daily["temperature_2m_min"][i],
                "precip_chance_pct": daily["precipitation_probability_max"][i],
                "conditions": _conditions(daily["weather_code"][i]),
            })
        return {
            "current": {
                "temperature_f": current.get("temperature_2m"),
                "feels_like_f": current.get("apparent_temperature"),
                "humidity_pct": current.get("relative_humidity_2m"),
                "precipitation_in": current.get("precipitation"),
                "wind_mph": current.get("wind_speed_10m"),
                "conditions": _conditions(current.get("weather_code")),
            },
            "forecast": days,
        }
