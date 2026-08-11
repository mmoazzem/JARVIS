"""profile_event() — data/profile.json as the MEMORY panel needs to see it.

Read-only and derived. The profile is written by digest -> merge and NOTHING
here writes it back: this module is a view, so a panel change can never reach
memory storage. Structured data out and no presentation, the same seam
telemetry.sample() sits behind — the panel decides how to draw N categories,
this decides only what the numbers are.

Categories are whatever the profile actually contains (today: two), never a
padded five. A count the profile cannot support (recall latency) is None, not
a plausible number.
"""
from __future__ import annotations

from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Optional

from core.constants import (
    LOG_DATE_FORMAT,
    MEMORY_EVENT_TYPE,
    MEMORY_RECENT_MAX,
    MEMORY_TIME_FORMAT,
    PROFILE_PATH,
)
from core.memory.base_digest import FactRecord
from core.memory.merge import load_profile
from core.memory.timestamps import parse_ts, sort_key

# One read serves every open connection: N tabs polling the same unchanged
# file cost one stat each, not N parses of the whole profile.
_cache: dict[Path, tuple[Optional[tuple[int, int]], dict]] = {}


def profile_stamp(path: Path = PROFILE_PATH) -> Optional[tuple[int, int]]:
    """A cheap identity for the file's current content, or None when absent.

    mtime alone can miss a same-nanosecond replace, so size rides along; the
    profile is written by os.replace, which always yields a fresh inode
    timestamp anyway. Callers poll THIS, and only build an event when it moves.
    """
    try:
        stat = path.stat()
    except OSError:
        return None
    return (stat.st_mtime_ns, stat.st_size)


def profile_event(path: Path = PROFILE_PATH) -> dict:
    """The MEMORY frame for the current profile.

    A missing or malformed profile is an EMPTY memory, not an error: the panel
    has a real empty state and a torn file is exactly the case it exists for.
    load_profile logs the one warning, and the stamp cache means it logs it
    once per change rather than once per poll.
    """
    stamp = profile_stamp(path)
    cached = _cache.get(path)
    if cached is not None and cached[0] == stamp:
        return cached[1]
    profile = load_profile(path)
    event = _build(profile.facts if profile else [])
    _cache[path] = (stamp, event)
    return event


def _build(facts: list[FactRecord]) -> dict:
    today = datetime.now().strftime(LOG_DATE_FORMAT)
    counts = Counter(fact.category for fact in facts)
    newest = sorted(facts, key=_sort_key, reverse=True)[:MEMORY_RECENT_MAX]
    return {
        "type": MEMORY_EVENT_TYPE,
        # What the profile RETAINS — every stored fact, both sides of a
        # conflict included. The category counts partition this same list, so
        # the bars always add up to the number above them.
        "facts_total": len(facts),
        "added_today": sum(1 for fact in facts if _local_day(fact.turn_ts) == today),
        # Nothing measures recall latency: the profile stores no timing, and
        # there is no retrieval step to time yet (Layer 3 renders the whole
        # working view into the prompt). The panel renders "—" for None.
        "recall_p95_ms": None,
        # Descending, tie-broken by name, so a live update never reshuffles
        # bars that did not change.
        "categories": [
            {"key": key, "count": count}
            for key, count in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
        ],
        "recent": [{"ts": _clock(fact.turn_ts), "text": fact.fact} for fact in newest],
    }


def _when(ts: str) -> Optional[datetime]:
    """A fact's timestamp as an aware datetime, or None when it isn't one.

    The parse itself moved to core.memory.timestamps when the SESSIONS panel
    turned out to need exactly the same defence against exactly the same
    values — one parser, two readers.
    """
    return parse_ts(ts)


def _sort_key(fact: FactRecord) -> tuple[bool, datetime]:
    return sort_key(fact.turn_ts)


def _local_day(ts: str) -> Optional[str]:
    when = _when(ts)
    return when.astimezone().strftime(LOG_DATE_FORMAT) if when else None


def _clock(ts: str) -> Optional[str]:
    """HH:MM local, or None when the stored ts isn't a timestamp."""
    when = _when(ts)
    return when.astimezone().strftime(MEMORY_TIME_FORMAT) if when else None
