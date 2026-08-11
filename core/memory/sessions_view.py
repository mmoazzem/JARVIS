"""sessions_event() — the event log's conversations as the SESSIONS panel sees them.

Read-only over the JSONL, plus one sidecar it owns outright. The panel's × means
DISMISS: it writes an id into logs/sessions_index.json and nothing else. The
event log is append-only ground truth, Layer 2 has already digested part of it,
and rewriting it on a UI click would strand derived facts in profile.json with no
provenance — so this module has no code path that can remove a record.

Session identity is REAL, not inferred. Every record written since slice 8
carries the `session_id` of the connection (or CLI run) that produced it; a gap
heuristic over timestamps would have invented sessions rather than reported them.
Records older than the field have no id and are grouped one session per day —
never backfilled.
"""
from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional

from core.constants import (
    EVENT_LOG_DATE_PATTERN,
    EVENT_LOG_GLOB,
    EVENTS_LOG_DIR,
    LEGACY_SESSION_PREFIX,
    LEGACY_SESSION_TITLE,
    LOGGER_MEMORY,
    SESSION_TITLE_MAX,
    SESSION_UNTITLED,
    SESSIONS_DISMISSED_KEY,
    SESSIONS_EVENT_TYPE,
    SESSIONS_HIDDEN_AT_KEY,
    SESSIONS_INDEX_PATH,
)
from core.memory.digest import atomic_write_text
from core.memory.timestamps import as_utc_z, sort_key

logger = logging.getLogger(LOGGER_MEMORY)

_DAY_FROM_NAME = re.compile(EVENT_LOG_DATE_PATTERN)


# --- the dismissed sidecar --------------------------------------------------

def load_dismissed(path: Path = SESSIONS_INDEX_PATH) -> dict:
    """Dismissed id -> hidden_at. A missing or torn sidecar hides NOTHING.

    Failing open is deliberate: the worst case is a session reappearing in the
    panel, and the alternative — failing closed — would hide real conversations
    because of a malformed file.
    """
    try:
        stored = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("sessions index unreadable (%s) — nothing is hidden", exc)
        return {}
    dismissed = stored.get(SESSIONS_DISMISSED_KEY) if isinstance(stored, dict) else None
    return dismissed if isinstance(dismissed, dict) else {}


def dismiss(session_id: str, path: Path = SESSIONS_INDEX_PATH) -> dict:
    """Hide one session from the panel. Writes an id and a time — no records.

    `hidden_at` is stored although nothing reads it yet: an ARCHIVED view is
    planned, and writing the timestamp now makes that a pure frontend addition
    rather than a migration.
    """
    dismissed = load_dismissed(path)
    dismissed[session_id] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(path, json.dumps({SESSIONS_DISMISSED_KEY: dismissed}, indent=2))
    logger.info("session dismissed from the panel (records untouched): %s", session_id)
    return dismissed


# --- reading the log --------------------------------------------------------

class _Session:
    def __init__(self, session_id: str, title: Optional[str]) -> None:
        self.id = session_id
        self.fixed_title = title      # legacy sessions are titled by their date
        self.first_user: Optional[str] = None
        self.turns = 0
        self.started_at: Optional[str] = None

    def add(self, record: dict) -> None:
        ts = record.get("ts")
        # Earliest wins, and an unparseable stamp never displaces a real one.
        if self.started_at is None or sort_key(ts) < sort_key(self.started_at):
            self.started_at = ts
        if record.get("role") != "exchange":
            return
        self.turns += 1
        if self.first_user is None:
            user = (record.get("user") or "").strip()
            if user:
                self.first_user = user

    def as_dict(self) -> dict:
        return {
            "id": self.id,
            "title": self.fixed_title or _title(self.first_user),
            "turns": self.turns,
            # Always Z-suffixed on the way out, whatever it looked like on disk.
            # None when the record's stamp was not a timestamp — the panel
            # renders UNKNOWN rather than guessing.
            "started_at": as_utc_z(self.started_at),
            "live": False,
        }


def _title(first_user: Optional[str]) -> str:
    """The session's first user message, cut on a word boundary."""
    text = " ".join((first_user or "").split())
    if not text:
        return SESSION_UNTITLED
    if len(text) <= SESSION_TITLE_MAX:
        return text
    cut = text[:SESSION_TITLE_MAX]
    # Only break on a space that leaves something worth reading; a single very
    # long word is truncated rather than reduced to nothing.
    space = cut.rfind(" ")
    if space > SESSION_TITLE_MAX // 2:
        cut = cut[:space]
    return cut.rstrip(" ,.;:") + "…"


def _day_of(path: Path) -> str:
    match = _DAY_FROM_NAME.search(path.name)
    return match.group(1) if match else path.stem


def _collect(log_dir: Path) -> dict[str, _Session]:
    sessions: dict[str, _Session] = {}
    try:
        files = sorted(log_dir.glob(EVENT_LOG_GLOB))
    except OSError:
        return sessions
    for path in files:
        day = _day_of(path)
        legacy_id = f"{LEGACY_SESSION_PREFIX}{day}"
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except OSError as exc:
            logger.warning("sessions: could not read %s (%s)", path.name, exc)
            continue
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                # One torn line must not cost the whole day. The log is
                # append-only, so a partial last line is the expected shape of
                # a process killed mid-write.
                continue
            if not isinstance(record, dict):
                continue
            session_id = record.get("session_id")
            if isinstance(session_id, str) and session_id:
                session = sessions.get(session_id)
                if session is None:
                    session = sessions[session_id] = _Session(session_id, None)
            else:
                session = sessions.get(legacy_id)
                if session is None:
                    session = sessions[legacy_id] = _Session(
                        legacy_id, LEGACY_SESSION_TITLE.format(date=day))
            session.add(record)
    return sessions


def sessions_event(live_id: Optional[str] = None,
                   log_dir: Path = EVENTS_LOG_DIR,
                   index_path: Path = SESSIONS_INDEX_PATH,
                   open_ids: Iterable[str] = ()) -> dict:
    """The SESSIONS frame. `live_id` is the ASKING connection's session — only
    that one is `live`, so another open tab appears in the list without
    claiming to be the viewer's own conversation.

    `open_ids` are the sessions currently connected. They are passed in rather
    than read from disk because a conversation that has not been spoken to yet
    has written nothing: without this, a second tab would be invisible to the
    first until someone typed in it.
    """
    dismissed = load_dismissed(index_path)
    sessions = _collect(log_dir)

    for open_id in ([live_id] if live_id else []) + list(open_ids):
        if open_id and open_id not in sessions:
            sessions[open_id] = _Session(open_id, None)

    rows = []
    for session_id, session in sessions.items():
        if session_id in dismissed and session_id != live_id:
            continue
        row = session.as_dict()
        row["live"] = session_id == live_id
        rows.append(row)

    # Newest first, the live session always at the top: it is the one being
    # written to, so its position must not depend on a timestamp race.
    rows.sort(key=lambda row: (row["live"], sort_key(row["started_at"])), reverse=True)
    return {"type": SESSIONS_EVENT_TYPE, "sessions": rows}
