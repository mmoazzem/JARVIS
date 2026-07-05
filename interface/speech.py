"""
Speech pipeline — an event-stream subscriber beside the CLI renderer.

Consumes the SAME structured events the CLI prints (it never touches agent
logic): `token` events accumulate into a sentence buffer; each completed sentence
is synthesized and played while later tokens are still streaming, so speech
starts before generation finishes.

It emits its own structured events (`speaking_started`, `speech_interrupted`,
`speech_done`) through the `on_event` callback — the same envelope idea as
respond(); today the event log records them, a future frontend will render them.

Interruption contract (the future frontend's spec):
  * interrupt() stops the CURRENT clip (flush, mid-word) AND discards every queued
    sentence — nothing pending ever plays.
  * The turn's text is untouched — only audio yields.
  * A sentence mid-synthesis when the interrupt lands is dropped after synthesis.
"""
from __future__ import annotations

import asyncio
import logging
import re
import threading
from typing import Callable, Optional

from core.constants import (
    EVENT_SPEAKING_STARTED,
    EVENT_SPEECH_DONE,
    EVENT_SPEECH_INTERRUPTED,
    LOGGER_SPEECH,
    SENTENCE_END_PATTERN,
    SPEECH_STRIP_CHARS_PATTERN,
)
from interface.audio import PulsePlayer
from models.tts.base import AudioClip, BaseTTS

logger = logging.getLogger(LOGGER_SPEECH)

_SENTENCE_END = re.compile(SENTENCE_END_PATTERN)
_STRIP_CHARS = re.compile(SPEECH_STRIP_CHARS_PATTERN)

_TURN_END = object()  # queue sentinel: the turn's final sentence has been enqueued


def split_sentences(buffer: str) -> tuple[list[str], str]:
    """Split off every COMPLETE sentence; return (sentences, unfinished remainder)."""
    sentences = []
    while True:
        match = _SENTENCE_END.search(buffer)
        if match is None:
            return sentences, buffer
        sentence = buffer[: match.end()].strip()
        if sentence:
            sentences.append(sentence)
        buffer = buffer[match.end():]


def _clean_for_speech(text: str) -> str:
    """Drop markdown markup characters so they are never read aloud."""
    return _STRIP_CHARS.sub("", text).strip()


class SpeechController:
    def __init__(
        self,
        tts: BaseTTS,
        player: PulsePlayer,
        on_event: Optional[Callable[[dict], None]] = None,
        preroll_ms: int = 0,
    ) -> None:
        self._tts = tts
        self._player = player
        # Cold-sink pre-roll (see JarvisConfig.tts_preroll_ms): the sink's start-up
        # swallow eats this silence instead of the first word. 0 = no-op.
        self._preroll_ms = preroll_ms
        self._on_event = on_event or (lambda event: None)
        self._queue: asyncio.Queue = asyncio.Queue()
        self._task: asyncio.Task | None = None
        self._buffer = ""
        self._muted = False
        self._speaking = False
        # Owned by the clip currently in player.play(); interrupt() sets it.
        self._current_stop: threading.Event | None = None
        # Bumped by interrupt(): a sentence that was mid-synthesis when the
        # interrupt landed is stale and must drop even if the NEXT turn has
        # already unmuted by the time its synthesis finishes.
        self._epoch = 0

    # --- lifecycle ----------------------------------------------------------

    def ensure_started(self) -> None:
        """Start the speaker loop once (must be called from the event loop)."""
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._speaker_loop())

    async def aclose(self) -> None:
        self.interrupt()
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

    # --- event-stream subscriber ---------------------------------------------

    def begin_turn(self) -> None:
        """A new turn is starting: unmute and drop any stale partial sentence."""
        self._buffer = ""
        self._muted = False

    def feed(self, event: dict) -> None:
        """Consume one orchestrator event. Non-blocking; never raises."""
        if self._muted:
            return
        kind = event.get("type")
        if kind == "token":
            self._buffer += event.get("content", "")
            sentences, self._buffer = split_sentences(self._buffer)
            for sentence in sentences:
                self._enqueue(sentence)
        elif kind in ("done", "error"):
            # Whatever remains is the turn's last (unterminated) sentence.
            self._enqueue(self._buffer)
            self._buffer = ""
            self._queue.put_nowait(_TURN_END)

    def _with_preroll(self, clip: AudioClip) -> AudioClip:
        """Pad the first clip of a speaking burst with leading silence so the
        sink's cold-start swallow consumes silence, not the opening word."""
        if self._preroll_ms <= 0:
            return clip
        frame_bytes = clip.sample_width * clip.channels
        pad = b"\x00" * (int(clip.sample_rate * self._preroll_ms / 1000) * frame_bytes)
        return clip.model_copy(update={"pcm": pad + clip.pcm})

    def _enqueue(self, text: str) -> None:
        cleaned = _clean_for_speech(text)
        if cleaned:
            self._queue.put_nowait(cleaned)

    # --- interruption ---------------------------------------------------------

    def interrupt(self) -> None:
        """Stop the current clip AND everything queued. Safe to call anytime."""
        self._muted = True  # in-flight synthesis and later feeds are dropped
        self._epoch += 1
        self._buffer = ""
        had_queued = False
        while True:
            try:
                had_queued = self._queue.get_nowait() is not _TURN_END or had_queued
            except asyncio.QueueEmpty:
                break
        stop = self._current_stop
        if stop is not None:
            stop.set()
        if self._speaking or had_queued:
            self._speaking = False
            logger.info("speech interrupted — current clip flushed, queue discarded")
            self._on_event({"type": EVENT_SPEECH_INTERRUPTED})

    # --- direct speech (goodbye etc.) ------------------------------------------

    async def say(self, text: str) -> None:
        """Synthesize and play one utterance to completion, outside any turn."""
        cleaned = _clean_for_speech(text)
        if not cleaned:
            return
        clip = await self._tts.synthesize(cleaned)
        # A one-off utterance always opens a cold sink — pad it too.
        await asyncio.to_thread(self._player.play, self._with_preroll(clip), threading.Event())

    # --- speaker loop -----------------------------------------------------------

    async def _speaker_loop(self) -> None:
        while True:
            item = await self._queue.get()
            if item is _TURN_END:
                if self._speaking:
                    self._speaking = False
                    self._on_event({"type": EVENT_SPEECH_DONE})
                continue
            try:
                epoch = self._epoch
                clip = await self._tts.synthesize(item)
                if self._muted or epoch != self._epoch:
                    continue  # interrupted while this sentence was synthesizing
                if not self._speaking:
                    self._speaking = True
                    self._on_event({"type": EVENT_SPEAKING_STARTED})
                    clip = self._with_preroll(clip)  # first clip of the burst
                logger.info("speaking (%.1fs audio): %.60r", clip.duration_s, item)
                stop = threading.Event()
                self._current_stop = stop
                completed = await asyncio.to_thread(self._player.play, clip, stop)
                self._current_stop = None
                if not completed:
                    self._speaking = False  # clip was flushed mid-play
            except Exception as exc:
                # Speech must never take a turn down — text already rendered.
                self._current_stop = None
                logger.warning("speech failed, sentence dropped: %s", exc)
