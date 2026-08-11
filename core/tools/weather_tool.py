"""
Weather tool — the model's READER over WeatherService.

Fetching, caching and unit choice all live in the service (one owner, one
reading); this file is only the seam that turns that reading into a tool result
and a failure into data. Network failure returns `{"error": ...}` — never raises
— so the model can give an honest "couldn't reach the weather service" answer
instead of the turn crashing.

The result carries a `units` map beside the numbers. The model is meant to READ
the unit from it rather than infer one from a key name, which is what the old
`temperature_f` shape forced it to do.
"""
from __future__ import annotations

import logging

from core.constants import LOGGER_TOOLS
from core.tools.base import Tool
from core.tools.weather_service import WeatherService

logger = logging.getLogger(LOGGER_TOOLS)


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

    def __init__(self, service: WeatherService) -> None:
        self._service = service

    async def run(self, city: str = "") -> dict:
        try:
            reading = await self._service.reading(city)
        except LookupError as exc:
            # No such place, or nowhere to look — a definite answer, not an outage.
            return {"error": str(exc)}
        except Exception as exc:
            logger.warning("weather lookup failed for %r: %s", city, exc)
            return self._failure(city, exc)
        return reading.as_result()

    def _failure(self, city: str, exc: Exception) -> dict:
        """An outage, plus whatever was last known — nested and dated.

        The last reading is NOT spread into the top level: a stale number
        presented in the shape of a current one is exactly what the caller must
        not be able to mistake it for. Age travels with it so the model can say
        how old the figure is instead of implying it is now.
        """
        result = {"error": f"weather service unreachable: {exc}"}
        held = self._service.held(city)
        if held is not None:
            result["last_reading"] = {
                "age_minutes": round(held.age_seconds() / 60, 1),
                "is_current": False,
                **held.as_result(),
            }
        return result
