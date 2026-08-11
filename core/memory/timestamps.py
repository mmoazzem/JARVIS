"""parse_ts() — the one place a stored timestamp is turned into a datetime.

Stored timestamps are inconsistent by history: some naive, some Z-suffixed, some
not timestamps at all (`"third_turn"`, from an early extractor), some absent.
Mixing naive and aware values in one sort RAISES, and a raw string sort ranks
"third_turn" above every real ISO stamp — so both the profile view and the
sessions view need exactly this parse, and neither may invent its own.

Naive values are read as UTC: every writer of these was working in UTC, and the
alternative is refusing to order records that are perfectly orderable.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

# Unparseable stamps sort HERE — last in a newest-first list, never first.
UNDATED = datetime.min.replace(tzinfo=timezone.utc)


def parse_ts(value) -> Optional[datetime]:
    """An aware datetime, or None when the value is not a timestamp at all."""
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def sort_key(value) -> tuple[bool, datetime]:
    """Newest-first ordering that puts unparseable stamps last.

    The bool leads so that "is a real timestamp" outranks the timestamp itself:
    reversed, True sorts ahead of False and the undated tail keeps its place.
    """
    when = parse_ts(value)
    return (when is not None, when or UNDATED)


def as_utc_z(value) -> Optional[str]:
    """The stamp as ISO 8601 with a Z suffix, whatever it looked like on disk."""
    when = parse_ts(value)
    if when is None:
        return None
    return when.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
