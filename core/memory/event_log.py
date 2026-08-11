"""
Event log — raw interaction capture for the future memory layer.

Append-only JSONL, one file per calendar day. CAPTURE ONLY: no digest, no
summarization, no intelligence — that arrives in a later pass and will read
these files.

It subscribes to the same structured event stream as the CLI and the speech
pipeline: tokens are ASSEMBLED into one record per turn (never one line per
token), and notable occurrences (error / recovery / speech interruption) are
noted on the turn record — or written standalone when they land between turns.

Assembly state belongs to the TURN, not to the log. It used to live on the log
as a single buffer, so two turns running at once shared it: the second
begin_turn dropped the first turn's record, both streams appended into the one
buffer, and one wrongly-paired exchange reached disk (a real BRAVO question
recorded with an ALPHA answer). Anything downstream reads that as a fact about
the user and cannot tell it is false, so the buffer is per-turn now.

A failed write must NEVER break a turn: every disk touch is wrapped; failures
become a log warning and the conversation continues.
"""
from __future__ import annotations

import asyncio
import json
import logging
import weakref
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from core.constants import (
    EVENT_LOG_FILE_FORMAT,
    EVENT_SPEECH_INTERRUPTED,
    EVENTS_LOG_DIR,
    LOGGER_MEMORY,
)

logger = logging.getLogger(LOGGER_MEMORY)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class Turn:
    """One turn's own buffer — the handle begin_turn() hands back.

    Whoever owns the turn owns this; two turns in flight are two objects, so
    neither can see or overwrite the other's half-assembled record.
    """

    def __init__(self, user_text: str, session_id: Optional[str] = None) -> None:
        self.session_id = session_id
        # The record IS the on-disk shape — one line, written verbatim at
        # end_turn. Nothing about the format changed when the buffer moved here.
        self.record: dict = {
            "ts": _now_iso(),
            "role": "exchange",  # one record = one user/assistant exchange
            "user": user_text,
            "assistant": "",
            "events": [],
        }
        # Which conversation this turn belongs to. Nothing on disk carried this
        # before, so a session could only ever be GUESSED from timestamp gaps —
        # which invents sessions rather than reporting them. Records written
        # before this field exists are grouped per day as legacy sessions;
        # they are never rewritten to add it.
        if session_id is not None:
            self.record["session_id"] = session_id


class EventLog:
    def __init__(self, enabled: bool = True, log_dir: Path = EVENTS_LOG_DIR) -> None:
        self._enabled = enabled
        self._dir = log_dir
        self._write_lock = asyncio.Lock()
        # Open turns, held WEAKLY — this is the set of records not yet on disk,
        # which is what tells a late arrival (see feed_speech) whether its turn
        # can still be amended. Weak, so a turn abandoned mid-generation is
        # collected with its caller instead of stranding a buffer here.
        self._open: "weakref.WeakSet[Turn]" = weakref.WeakSet()

    def _path_for_today(self) -> Path:
        # Computed per write so a session that crosses midnight rolls files.
        return self._dir / datetime.now().strftime(EVENT_LOG_FILE_FORMAT)

    # --- turn assembly (subscriber side, all non-blocking) ----------------------

    def begin_turn(self, user_text: str, session_id: Optional[str] = None) -> Optional[Turn]:
        """Open a turn and return its handle — pass it to feed() and end_turn().

        None when capture is disabled, which both of those accept, so callers
        never branch on it.

        `session_id` names the conversation the turn belongs to. It is the
        CALLER's, because only the caller knows what a session is on its
        surface: one WebSocket connection, or one CLI process run.
        """
        if not self._enabled:
            return None
        turn = Turn(user_text, session_id)
        self._open.add(turn)
        return turn

    def feed(self, turn: Optional[Turn], event: dict) -> None:
        """Consume one orchestrator event; assemble, never write here."""
        if turn is None:
            return
        kind = event.get("type")
        if kind == "token":
            turn.record["assistant"] += event.get("content", "")
        elif kind == "error":
            turn.record["events"].append({"type": "error", "message": event.get("message", "")})
        elif kind == "delegation":
            turn.record["events"].append({"type": "delegation", "tool": event.get("tool", "")})
        elif kind == "recovery":
            # Explicit agent event: the zero-content recovery path ran (gotcha #2).
            turn.record["events"].append({"type": "recovery_attempted"})

    def feed_speech(self, turn: Optional[Turn], event: dict) -> None:
        """Consume a speech event for a NAMED turn. Interruptions are the notable
        ones to persist.

        The handle used to be inferred: attach to the open turn when there was
        exactly one, otherwise write standalone. That held only while speech
        existed solely in the CLI, which has one turn and one listener. With
        browser speech and several connections, "exactly one" stops being true
        and every interruption would silently become an orphan record. The
        speech pipeline now carries the handle from the turn that produced the
        audio, so this attaches to the right turn or to none.

        A turn that has already been written is no longer open — its audio is
        still playing, but its record is on disk and cannot be amended, so a
        late interruption is persisted standalone rather than lost.
        """
        if not self._enabled:
            return
        if event.get("type") != EVENT_SPEECH_INTERRUPTED:
            return
        if turn is not None and turn in self._open:
            turn.record["events"].append({"type": EVENT_SPEECH_INTERRUPTED})
        else:
            record = {"ts": _now_iso(), "role": "event", "type": EVENT_SPEECH_INTERRUPTED}
            # A standalone record still belongs to the session that produced it,
            # when the caller named one — the turn it could not be attached to
            # is still the turn that was speaking.
            if turn is not None and turn.session_id is not None:
                record["session_id"] = turn.session_id
            asyncio.get_running_loop().create_task(self._append(record))

    async def end_turn(self, turn: Optional[Turn]) -> None:
        """Write this turn's record. Failures warn and are swallowed.

        A turn that never reaches here — the client vanished mid-generation —
        writes nothing at all: a half-assembled exchange must not land on disk
        looking complete.
        """
        if turn is None or turn not in self._open:
            return  # disabled, or already written: never write a turn twice
        self._open.discard(turn)
        await self._append(turn.record)

    # --- disk ---------------------------------------------------------------

    async def _append(self, record: dict) -> None:
        try:
            line = json.dumps(record, ensure_ascii=False)
            async with self._write_lock:
                await asyncio.to_thread(self._write_line, line)
        except Exception as exc:
            # The log is a bystander: a locked/full/missing disk never breaks a turn.
            logger.warning("event log write failed (turn continues): %s", exc)

    def _write_line(self, line: str) -> None:
        self._dir.mkdir(parents=True, exist_ok=True)
        with open(self._path_for_today(), "a", encoding="utf-8") as f:
            f.write(line + "\n")
