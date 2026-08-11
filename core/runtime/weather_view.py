"""weather_event() — a held WeatherReading as the weather tile needs to see it.

Read-only and derived, the same seam telemetry.sample() and profile_event() sit
behind: structured data out, no presentation. Degrees stay degrees and units stay
words here; turning 199 into SSW and `fahrenheit` into °F is the frontend's job,
because the model reads this same vocabulary aloud and "SSW" is not a word.

The frame carries `is_current` and `age_minutes` because a reading that could not
be refreshed is still worth showing — but only if the viewer can tell. Presenting
a held reading with no age is the one thing this must never do.
"""
from __future__ import annotations

from core.constants import WEATHER_DISPLAY_ROUNDED_FIELDS, WEATHER_EVENT_TYPE
from core.tools.weather_service import WeatherReading, WeatherService


def _rounded_for_display(current: dict) -> dict:
    """Whole degrees for the tile — a DISPLAY decision, made here and nowhere else.

    Two surfaces reading 75.5 and 76 look like disagreement even when both are
    right, and half a degree is neither perceptible nor inside Open-Meteo's own
    accuracy. The HELD reading keeps its full precision (the service never sees
    this), so a future trend or comparison still has real numbers to work with,
    and the tool's return value still carries the decimal for the model.

    Temperature only. Humidity, precipitation and wind are left exactly as
    measured — 0.02 in rounded to whole inches would be no reading at all.
    """
    rounded = dict(current)
    for key in WEATHER_DISPLAY_ROUNDED_FIELDS:
        value = rounded.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            rounded[key] = round(value)
    return rounded


def weather_frame(reading: WeatherReading, is_current: bool) -> dict:
    """One reading as a `weather` frame. An absent value omits its key, per the
    telemetry convention — a wind_speed of 0 is calm, a missing one is unknown."""
    frame = {
        "type": WEATHER_EVENT_TYPE,
        "place": reading.location,
        # Whole days, not a rendered subset: the detail view is then a rendering
        # change rather than a wire change.
        "current": _rounded_for_display(reading.current),
        "units": reading.units,
        "forecast": reading.forecast,
        "is_current": is_current,
        "age_minutes": round(reading.age_seconds() / 60, 1),
    }
    for key, value in (("lat", reading.latitude), ("lon", reading.longitude),
                       ("observed_at", reading.observed_at)):
        if value is not None:
            frame[key] = value
    return frame


async def weather_event(service: WeatherService, refresh: bool = False) -> dict:
    """The current frame, fetching per the service's rules.

    A failed fetch falls back to the last held reading marked NOT current. With
    nothing held there is nothing honest to send, so the exception propagates and
    the tile keeps reading "—" rather than inventing a first value.
    """
    try:
        reading = await (service.refresh() if refresh else service.reading())
    except Exception:
        held = service.held()
        if held is None:
            raise
        return weather_frame(held, is_current=False)
    return weather_frame(reading, is_current=True)
