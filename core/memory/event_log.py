"""
Event log — raw interaction capture for the future memory layer.

Append-only JSONL, one file per calendar day. CAPTURE ONLY: no digest, no
summarization, no intelligence — that arrives in a later pass and will read
these files.

It subscribes to the same structured event stream as the CLI and the speech
pipeline: tokens are ASSEMBLED into one record per turn (never one line per
token), and notable occurrences (error / recovery / speech interruption) are
noted on the turn record — or written standalone when they land between turns.

A failed write must NEVER break a turn: every disk touch is wrapped; failures
become a log warning and the conversation continues.
"""
from __future__ import annotations

import asyncio
import json
import logging
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


class EventLog:
    def __init__(self, enabled: bool = True, log_dir: Path = EVENTS_LOG_DIR) -> None:
        self._enabled = enabled
        self._dir = log_dir
        self._write_lock = asyncio.Lock()
        self._turn: Optional[dict] = None
        self._thinking_count = 0

    def _path_for_today(self) -> Path:
        # Computed per write so a session that crosses midnight rolls files.
        return self._dir / datetime.now().strftime(EVENT_LOG_FILE_FORMAT)

    # --- turn assembly (subscriber side, all non-blocking) ----------------------

    def begin_turn(self, user_text: str) -> None:
        if not self._enabled:
            return
        self._turn = {
            "ts": _now_iso(),
            "role": "exchange",  # one record = one user/assistant exchange
            "user": user_text,
            "assistant": "",
            "events": [],
        }
        self._thinking_count = 0

    def feed(self, event: dict) -> None:
        """Consume one orchestrator event; assemble, never write here."""
        if not self._enabled or self._turn is None:
            return
        kind = event.get("type")
        if kind == "token":
            self._turn["assistant"] += event.get("content", "")
        elif kind == "error":
            self._turn["events"].append({"type": "error", "message": event.get("message", "")})
        elif kind == "thinking":
            # The agent yields "thinking" once per model call, so a SECOND one in
            # the same turn means the zero-content recovery path ran (gotcha #2).
            self._thinking_count += 1
            if self._thinking_count == 2:
                self._turn["events"].append({"type": "recovery_attempted"})

    def feed_speech(self, event: dict) -> None:
        """Consume a speech event. Interruptions are the notable ones to persist."""
        if not self._enabled:
            return
        if event.get("type") != EVENT_SPEECH_INTERRUPTED:
            return
        if self._turn is not None:
            self._turn["events"].append({"type": EVENT_SPEECH_INTERRUPTED})
        else:
            # Between turns (residual speech after the record was written):
            # persist it standalone so the interruption is never lost.
            record = {"ts": _now_iso(), "role": "event", "type": EVENT_SPEECH_INTERRUPTED}
            asyncio.get_running_loop().create_task(self._append(record))

    async def end_turn(self) -> None:
        """Write the assembled turn record. Failures warn and are swallowed."""
        if not self._enabled or self._turn is None:
            return
        record, self._turn = self._turn, None
        await self._append(record)

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
