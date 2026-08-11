"""
WeatherService — the ONE thing that fetches weather, and the holder of the last
reading everything else reads.

Before this, the fetch lived inside WeatherTool and there was no cache, so every
consumer was its own path to the same fact. A tile polling on its own clock
would have produced a second, independently-timed answer: 73.8 on screen while
JARVIS says 74.1 from a fetch two seconds later, with nothing to say which is
right. One owner and one held reading make that disagreement impossible to
express rather than merely unlikely.

The unit is DATA here, and it is the unit the API ANSWERED WITH (`current_units`
in the response), never the unit we asked for. Those two can disagree — a param
we spelled wrong, a value Open-Meteo declines — and only the answered one is
guaranteed to describe the number printed beside it.

The held reading also remembers WHICH units it was fetched with. Without that, a
unit change would serve the old payload under the new label: the same silent
mislabel as encoding the unit in a key name, moved one layer up.
"""
from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime, timedelta, timezone
from typing import Optional

import httpx
from pydantic import BaseModel

from core.constants import (
    LOGGER_TOOLS,
    OPEN_METEO_CURRENT_FIELDS,
    OPEN_METEO_CURRENT_KEYS,
    OPEN_METEO_DAILY_FIELDS,
    OPEN_METEO_DAILY_KEYS,
    OPEN_METEO_DAILY_UNIT_FIELDS,
    OPEN_METEO_FORECAST_URL,
    OPEN_METEO_GEOCODE_URL,
    OPEN_METEO_TIMEOUT_S,
    OPEN_METEO_UNIT_FIELDS,
    OPEN_METEO_UNIT_PARAMS,
    WEATHER_FORECAST_DAYS,
    WEATHER_READING_TTL_S,
    WMO_WEATHER_CODES,
)
from core.preferences import Preferences

logger = logging.getLogger(LOGGER_TOOLS)


def _conditions(code) -> str:
    return WMO_WEATHER_CODES.get(code, f"weather code {code}")


class WeatherReading(BaseModel):
    """One fetched observation, with everything needed to judge its age and
    whether it still means what its labels say."""

    location: str
    current: dict
    units: dict
    forecast: list[dict]
    observed_at: Optional[str] = None
    # Where the reading was taken. Carried on the reading rather than looked up
    # again by a caller, so a displayed coordinate can never belong to a
    # different place than the numbers beside it. Deliberately NOT in
    # as_result(): the model has no use for coordinates, and would read them out.
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    # The unit params this reading was FETCHED with. Compared against current
    # preferences to decide whether it may still be served.
    units_used: dict
    fetched_at: datetime
    # Wall clock answers "how old is this?" for a human; the monotonic stamp
    # answers "has the TTL elapsed?" without a clock change lying about it.
    fetched_monotonic: float

    def age_seconds(self) -> float:
        return max(0.0, time.monotonic() - self.fetched_monotonic)

    def as_result(self) -> dict:
        """The reading as the tool returns it — an absent field omits its key
        (a wind_speed of 0 is calm; a missing wind_speed is unknown)."""
        result = {
            "location": self.location,
            "current": self.current,
            "units": self.units,
            "forecast": self.forecast,
        }
        if self.observed_at is not None:
            result["observed_at"] = self.observed_at
        return result


class WeatherService:
    """Fetches, holds, and re-fetches weather. Raises on failure — turning a
    failure into `{"error": ...}` is the Tool's job, not this layer's."""

    def __init__(
        self,
        default_location: str,
        preferences: Preferences,
        ttl_s: float = WEATHER_READING_TTL_S,
    ) -> None:
        self._default = default_location
        self._preferences = preferences
        self._ttl_s = ttl_s
        # Geocoding results are stable — cache per place name for the session.
        self._coords: dict[str, dict] = {}
        # Last reading per place. Keyed by place because a question about Denver
        # must not be answered with the reading held for Buffalo.
        self._readings: dict[str, WeatherReading] = {}
        # Fetches in flight, per place. Until the dashboard poll existed nothing
        # could call refresh() concurrently; now a scheduled poll and a tool call
        # routinely land together, and without this they would both fetch and the
        # loser's result would overwrite the winner's for no gain.
        self._inflight: dict[str, asyncio.Task] = {}

    def place_for(self, city: str = "") -> str:
        """The place a call resolves to — shared so `held` and `reading` agree
        on which entry a bare "what's the weather?" refers to."""
        return (city or self._default).strip()

    def held(self, city: str = "") -> Optional[WeatherReading]:
        """The last reading for a place regardless of age, or None.

        Exists for the failure path: a caller that could not fetch still needs
        to be able to say what it last knew AND how old that is.
        """
        return self._readings.get(self.place_for(city).lower())

    async def reading(self, city: str = "") -> WeatherReading:
        """The current reading, fetching only if the held one cannot serve."""
        held = self.held(city)
        if held is not None and self._is_fresh(held):
            logger.info(
                "weather: serving held reading for %s (age %.0fs, ttl %.0fs)",
                held.location, held.age_seconds(), self._ttl_s,
            )
            return held
        return await self.refresh(city)

    async def refresh(self, city: str = "") -> WeatherReading:
        """Fetch now, ignoring the TTL, and hold the result.

        The bypass is not a debug hook: a unit change and a manual refresh both
        need a reading that is new rather than merely recent.

        Concurrent callers for the same place JOIN the fetch already running
        instead of starting a second one. The join is shielded, so a caller that
        gives up (a socket closing mid-poll) cancels its own wait and not the
        fetch every other caller is still waiting on.
        """
        place = self.place_for(city)
        if not place:
            raise LookupError("no city given and no default location configured")
        key = place.lower()
        task = self._inflight.get(key)
        if task is None:
            task = asyncio.ensure_future(self._refresh(place))
            self._inflight[key] = task
            task.add_done_callback(lambda done, k=key: self._inflight.pop(k, None))
        else:
            logger.info("weather: joining the fetch already in flight for %s", place)
        return await asyncio.shield(task)

    async def _refresh(self, place: str) -> WeatherReading:
        location = await self._geocode(place)
        if location is None:
            raise LookupError(f"no location found for {place!r}")
        unit_params = self._unit_params()
        reading = self._parse(location, await self._fetch(location, unit_params), unit_params)
        self._readings[place.lower()] = reading
        logger.info("weather: fetched %s with units %s", reading.location, reading.units)
        return reading

    def _is_fresh(self, reading: WeatherReading) -> bool:
        # A unit change ages a reading out instantly, however recent it is: the
        # numbers in it are in the OLD unit, and serving them under the new
        # preference would mislabel them.
        if reading.units_used != self._unit_params():
            logger.info("weather: held reading was fetched with different units — refetching")
            return False
        return reading.age_seconds() < self._ttl_s

    def _unit_params(self) -> dict:
        """Preferences as Open-Meteo query params. Read per call, so a
        preference written a moment ago is already in effect."""
        units = self._preferences.units()
        return {
            OPEN_METEO_UNIT_PARAMS[dimension]: value
            for dimension, value in units.items()
            if dimension in OPEN_METEO_UNIT_PARAMS
        }

    async def _fetch(self, location: dict, unit_params: dict) -> dict:
        async with httpx.AsyncClient(timeout=OPEN_METEO_TIMEOUT_S) as client:
            resp = await client.get(
                OPEN_METEO_FORECAST_URL,
                params={
                    "latitude": location["latitude"],
                    "longitude": location["longitude"],
                    "current": ",".join(OPEN_METEO_CURRENT_FIELDS),
                    "daily": ",".join(OPEN_METEO_DAILY_FIELDS),
                    "forecast_days": WEATHER_FORECAST_DAYS,
                    "timezone": "auto",
                    **unit_params,
                },
            )
            resp.raise_for_status()
        return resp.json()

    def _parse(self, location: dict, data: dict, unit_params: dict) -> WeatherReading:
        current = data.get("current") or {}
        daily = data.get("daily") or {}
        codes = daily.get("weather_code") or []

        forecast = []
        for i, date in enumerate(daily.get("time") or []):
            day = {"date": date}
            day.update(_mapped(daily, OPEN_METEO_DAILY_KEYS, index=i))
            if i < len(codes):
                day["conditions"] = _conditions(codes[i])
            forecast.append(day)

        measurements = _mapped(current, OPEN_METEO_CURRENT_KEYS)
        if current.get("weather_code") is not None:
            measurements["conditions"] = _conditions(current["weather_code"])

        return WeatherReading(
            location=location["label"],
            current=measurements,
            units=_units_from_response(data),
            forecast=forecast,
            observed_at=_observed_at(data),
            latitude=location.get("latitude"),
            longitude=location.get("longitude"),
            units_used=unit_params,
            fetched_at=datetime.now(timezone.utc),
            fetched_monotonic=time.monotonic(),
        )

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


def _mapped(source: dict, keys: dict, index: Optional[int] = None) -> dict:
    """Rename response fields to neutral keys, omitting what isn't there."""
    out = {}
    for neutral, field in keys.items():
        value = source.get(field)
        if index is not None:
            value = value[index] if isinstance(value, list) and index < len(value) else None
        if value is not None:
            out[neutral] = value
    return out


def _units_from_response(data: dict) -> dict:
    """The unit per dimension AS THE API DECLARED IT.

    This is the whole point of the rework: `current_units` is the only statement
    of unit that is guaranteed to match the number it sits beside. What we
    requested is an intention; this is the answer.
    """
    current_units = data.get("current_units") or {}
    daily_units = data.get("daily_units") or {}
    units = {}
    for dimension, field in OPEN_METEO_UNIT_FIELDS.items():
        label = current_units.get(field)
        if label is None:
            # A payload with only a daily block still reports temperature.
            label = daily_units.get(OPEN_METEO_DAILY_UNIT_FIELDS.get(dimension, ""))
        if label is not None:
            units[dimension] = label
    return units


def _observed_at(data: dict) -> Optional[str]:
    """The observation's timestamp in UTC.

    The response is in the location's own zone (`timezone=auto`), which is right
    for display and wrong for comparing ages, so it is normalised here once.
    """
    stamp = (data.get("current") or {}).get("time")
    if not stamp:
        return None
    try:
        local = datetime.fromisoformat(stamp)
    except ValueError:
        return None
    offset = data.get("utc_offset_seconds") or 0
    return (local - timedelta(seconds=offset)).strftime("%Y-%m-%dT%H:%M:%SZ")
