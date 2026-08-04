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

    def __init__(self, user_text: str) -> None:
        # The record IS the on-disk shape — one line, written verbatim at
        # end_turn. Nothing about the format changed when the buffer moved here.
        self.record: dict = {
            "ts": _now_iso(),
            "role": "exchange",  # one record = one user/assistant exchange
            "user": user_text,
            "assistant": "",
            "events": [],
        }


class EventLog:
    def __init__(self, enabled: bool = True, log_dir: Path = EVENTS_LOG_DIR) -> None:
        self._enabled = enabled
        self._dir = log_dir
        self._write_lock = asyncio.Lock()
        # Open turns, held WEAKLY. The log tracks them for one reason only: the
        # speech side-channel is wired once at startup and has no handle of its
        # own (see feed_speech). Weak, so a turn abandoned mid-generation is
        # collected with its caller instead of stranding a buffer here.
        self._open: "weakref.WeakSet[Turn]" = weakref.WeakSet()

    def _path_for_today(self) -> Path:
        # Computed per write so a session that crosses midnight rolls files.
        return self._dir / datetime.now().strftime(EVENT_LOG_FILE_FORMAT)

    # --- turn assembly (subscriber side, all non-blocking) ----------------------

    def begin_turn(self, user_text: str) -> Optional[Turn]:
        """Open a turn and return its handle — pass it to feed() and end_turn().

        None when capture is disabled, which both of those accept, so callers
        never branch on it.
        """
        if not self._enabled:
            return None
        turn = Turn(user_text)
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

    def feed_speech(self, event: dict) -> None:
        """Consume a speech event. Interruptions are the notable ones to persist.

        This is the one subscriber with no handle: the speech pipeline is wired
        once at startup, long before any turn exists. It attaches to the open
        turn only when there is EXACTLY one — with several in flight there is no
        way to know which one was interrupted, and guessing is precisely what
        corrupted this log before, so it is written standalone instead.
        """
        if not self._enabled:
            return
        if event.get("type") != EVENT_SPEECH_INTERRUPTED:
            return
        open_turns = list(self._open)
        if len(open_turns) == 1:
            open_turns[0].record["events"].append({"type": EVENT_SPEECH_INTERRUPTED})
        else:
            # Between turns (residual speech after the record was written):
            # persist it standalone so the interruption is never lost.
            record = {"ts": _now_iso(), "role": "event", "type": EVENT_SPEECH_INTERRUPTED}
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
