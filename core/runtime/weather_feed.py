"""WeatherFeed — the ambient weather push, shared by every open dashboard.

Telemetry and memory each poll per connection, which is right for them: both are
cheap and local. This one leaves the machine, so N dashboards must not become N
pollers of a third-party API. One loop runs for as long as at least one
connection is subscribed and stops with the last one — nothing is fetched on
behalf of a browser that is not open.

The connect push and the interval push are deliberately different calls.
Subscribing serves the HELD reading when there is a fresh one (a reload must not
cost a request), while the interval is the poll itself and always refetches.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Awaitable, Callable

from core.constants import LOGGER_ROOT, WEATHER_POLL_INTERVAL_S
from core.runtime.weather_view import weather_event
from core.tools.weather_service import WeatherService

logger = logging.getLogger(f"{LOGGER_ROOT}.weather")

Sender = Callable[[dict], Awaitable[None]]


class WeatherFeed:
    def __init__(self, service: WeatherService,
                 interval_s: float = WEATHER_POLL_INTERVAL_S) -> None:
        self._service = service
        self._interval_s = interval_s
        self._subscribers: set[Sender] = set()
        self._task: asyncio.Task | None = None

    async def subscribe(self, send: Sender) -> None:
        """Add a connection and push what is known to it now."""
        self._subscribers.add(send)
        if self._task is None or self._task.done():
            logger.info("weather poll started — %d dashboard(s) connected",
                        len(self._subscribers))
            self._task = asyncio.create_task(self._poll())
        try:
            await send(await weather_event(self._service))
        except Exception as exc:
            # No reading and no network on the very first connect: the tile stays
            # at "—", which is the truth. Never a placeholder.
            logger.warning("weather: nothing to send on connect (%s)", exc)

    def unsubscribe(self, send: Sender) -> None:
        self._subscribers.discard(send)
        if self._subscribers or self._task is None:
            return
        logger.info("weather poll stopped — no dashboards connected")
        self._task.cancel()
        self._task = None

    async def _poll(self) -> None:
        while True:
            await asyncio.sleep(self._interval_s)
            if not self._subscribers:
                return
            try:
                event = await weather_event(self._service, refresh=True)
            except Exception as exc:
                # Nothing held and the fetch failed — hold the tile where it is
                # and try again next tick rather than pushing a half-frame.
                logger.warning("weather poll failed, nothing to push: %s", exc)
                continue
            if not event.get("is_current"):
                logger.info("weather poll: refresh failed, pushing the held "
                            "reading as stale (%.1f min old)", event["age_minutes"])
            await self.broadcast(event)

    async def broadcast(self, event: dict) -> None:
        """Push one frame to every dashboard. Used by the poll, and by a unit
        change — which is a new reading arriving off the interval's clock."""
        # Each send goes through its own connection's lock; one dead socket must
        # not stop the frame reaching the others.
        for send in list(self._subscribers):
            try:
                await send(event)
            except Exception as exc:
                logger.info("weather: dropping a subscriber that would not take "
                            "the frame (%s)", exc)
                self._subscribers.discard(send)
