"""
Speech pipeline — an event-stream subscriber beside the CLI renderer.

Consumes the SAME structured events the CLI prints (it never touches agent
logic): `token` events accumulate into a sentence buffer; each completed sentence
is synthesized and played while later tokens are still streaming, so speech
starts before generation finishes.

It emits its own structured events (`speaking_started`, `speech_level`,
`speech_interrupted`, `speech_done`) through the `on_event` callback — the same
envelope idea as respond(); the event log records the notable ones and the
frontend renders them.

Every emitted event carries the HANDLE of the turn whose audio it belongs to.
The handle travels on the queue, not on the controller, because audio outlives
the text that produced it: by the time a sentence is interrupted the next turn
may already have begun, and the controller's "current turn" would name the
wrong one. Whoever wires on_event decides what a handle is — the CLI passes the
event log's Turn, the server passes its own speech session.

`speaking_started` means AUDIBLE, not enqueued: it is raised by the player when
the first sample with energy in it reaches the sink, so a frontend that shows a
speaking state is never showing it over silence.

Interruption contract (the frontend's spec):
  * interrupt() stops the CURRENT clip (flush, mid-word) AND discards every queued
    sentence — nothing pending ever plays.
  * The turn's text is untouched — only audio yields.
  * A sentence mid-synthesis when the interrupt lands is dropped after synthesis.
"""
from __future__ import annotations

import asyncio
import logging
import threading
from typing import Any, Callable, Optional

from core.constants import (
    EVENT_SPEAKING_STARTED,
    EVENT_SPEECH_DONE,
    EVENT_SPEECH_INTERRUPTED,
    EVENT_SPEECH_LEVEL,
    LOGGER_SPEECH,
)
from interface.audio import PulsePlayer
from interface.speech_text import (
    clean_for_speech,
    normalize_for_speech,
    split_sentences,
)
from models.tts import create_tts
from models.tts.base import AudioClip, BaseTTS

logger = logging.getLogger(LOGGER_SPEECH)

_TURN_END = object()  # queue sentinel: the turn's final sentence has been enqueued


def _for_synthesis(text: str) -> str:
    """The one place written text becomes spoken text.

    Strip markup, then normalize symbols and abbreviations. This is the ONLY
    string that differs from the transcript — the event log and the frontend
    both keep what the model actually wrote, symbols and all.
    """
    return normalize_for_speech(clean_for_speech(text))


async def build_controller(
    config, on_event: Optional[Callable[[Any, dict], None]] = None
) -> "SpeechController":
    """Assemble the speech pipeline from config — the one place it is built.

    Raises AudioUnavailableError / ValueError on a machine that cannot play
    audio; every caller (CLI, server) decides for itself what that means, which
    is always "voice stays off and the app runs on", never a crash.
    """
    player = PulsePlayer()
    await asyncio.to_thread(player._lib)  # fail fast, where it can still be handled
    return SpeechController(
        create_tts(config.tts_engine, config.tts_voice),
        player,
        on_event=on_event,
        preroll_ms=config.tts_preroll_ms,
    )


class SpeechController:
    def __init__(
        self,
        tts: BaseTTS,
        player: PulsePlayer,
        on_event: Optional[Callable[[Any, dict], None]] = None,
        preroll_ms: int = 0,
    ) -> None:
        self._tts = tts
        self._player = player
        # Cold-sink pre-roll (see JarvisConfig.tts_preroll_ms): the sink's start-up
        # swallow eats this silence instead of the first word. 0 = no-op.
        self._preroll_ms = preroll_ms
        self._on_event = on_event or (lambda handle, event: None)
        self._queue: asyncio.Queue = asyncio.Queue()
        self._task: asyncio.Task | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._buffer = ""
        self._muted = False
        self._speaking = False
        # The turn currently being FED (feed/_enqueue stamp it onto queue items)
        # and the turn whose audio is SOUNDING. They are the same turn until
        # speech runs past the end of its own text, which it routinely does.
        self._turn_handle: Any = None
        self._playing_handle: Any = None
        # Owned by the clip currently in player.play(); interrupt() sets it.
        self._current_stop: threading.Event | None = None
        # Bumped by interrupt(): a sentence that was mid-synthesis when the
        # interrupt landed is stale and must drop even if the NEXT turn has
        # already unmuted by the time its synthesis finishes. It also fences off
        # player callbacks still in flight from the abandoned clip.
        self._epoch = 0

    # --- lifecycle ----------------------------------------------------------

    def ensure_started(self) -> None:
        """Start the speaker loop once (must be called from the event loop)."""
        self._loop = asyncio.get_running_loop()
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

    def begin_turn(self, handle: Any = None) -> None:
        """A new turn is starting: unmute, drop any stale partial sentence, and
        adopt this turn's handle for everything fed from here on."""
        self._buffer = ""
        self._muted = False
        self._turn_handle = handle

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
            self._queue.put_nowait((self._turn_handle, _TURN_END))

    def _with_preroll(self, clip: AudioClip) -> AudioClip:
        """Pad the first clip of a speaking burst with leading silence so the
        sink's cold-start swallow consumes silence, not the opening word."""
        if self._preroll_ms <= 0:
            return clip
        frame_bytes = clip.sample_width * clip.channels
        pad = b"\x00" * (int(clip.sample_rate * self._preroll_ms / 1000) * frame_bytes)
        return clip.model_copy(update={"pcm": pad + clip.pcm})

    def _enqueue(self, text: str) -> None:
        spoken = _for_synthesis(text)
        if spoken:
            self._queue.put_nowait((self._turn_handle, spoken))

    # --- interruption ---------------------------------------------------------

    def interrupt(self) -> None:
        """Stop the current clip AND everything queued. Safe to call anytime."""
        self._muted = True  # in-flight synthesis and later feeds are dropped
        self._epoch += 1
        self._buffer = ""
        had_queued = False
        # The interruption belongs to the turn whose audio was cut off — which is
        # the one SOUNDING, or, if nothing had reached the sink yet, the one whose
        # sentences were still waiting in the queue.
        handle = self._playing_handle
        while True:
            try:
                queued_handle, item = self._queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            if item is not _TURN_END:
                had_queued = True
                if handle is None:
                    handle = queued_handle
        stop = self._current_stop
        if stop is not None:
            stop.set()
        if self._speaking or had_queued:
            self._speaking = False
            self._playing_handle = None
            logger.info("speech interrupted — current clip flushed, queue discarded")
            self._on_event(handle, {"type": EVENT_SPEECH_INTERRUPTED})

    # --- direct speech (goodbye etc.) ------------------------------------------

    async def say(self, text: str) -> None:
        """Synthesize and play one utterance to completion, outside any turn."""
        spoken = _for_synthesis(text)
        if not spoken:
            return
        clip = await self._tts.synthesize(spoken)
        # A one-off utterance always opens a cold sink — pad it too.
        await asyncio.to_thread(self._player.play, self._with_preroll(clip), threading.Event())

    # --- speaker loop -----------------------------------------------------------

    # --- player callbacks (arrive on the PLAYER's thread) ------------------------

    def _from_player(self, handle: Any, epoch: int, event: dict) -> None:
        """Hand one player callback back to the event loop.

        Called off-thread, so it does no work here beyond the hop. The epoch
        fences off readings from a clip that was already interrupted: the write
        thread can be one step behind the flush.
        """
        loop = self._loop
        if loop is None or epoch != self._epoch:
            return
        loop.call_soon_threadsafe(self._on_player_event, handle, epoch, event)

    def _on_player_event(self, handle: Any, epoch: int, event: dict) -> None:
        if self._muted or epoch != self._epoch:
            return
        if event["type"] == EVENT_SPEAKING_STARTED:
            if self._speaking:
                return  # already sounding — one start per speaking burst
            self._speaking = True
        elif not self._speaking:
            return  # a level from before the burst opened has nothing to meter
        self._on_event(handle, event)

    # --- speaker loop -----------------------------------------------------------

    async def _speaker_loop(self) -> None:
        while True:
            handle, item = await self._queue.get()
            if item is _TURN_END:
                # ALWAYS closes the turn's speech, even when nothing sounded: a
                # subscriber that opened a bracket on this turn has no other way
                # to learn that it will never be spoken.
                self._speaking = False
                self._playing_handle = None
                self._on_event(handle, {"type": EVENT_SPEECH_DONE})
                continue
            try:
                epoch = self._epoch
                clip = await self._tts.synthesize(item)
                if self._muted or epoch != self._epoch:
                    continue  # interrupted while this sentence was synthesizing
                if not self._speaking:
                    clip = self._with_preroll(clip)  # first clip of the burst
                logger.info("speaking (%.1fs audio): %.60r", clip.duration_s, item)
                stop = threading.Event()
                self._current_stop = stop
                self._playing_handle = handle
                completed = await asyncio.to_thread(
                    self._player.play,
                    clip,
                    stop,
                    lambda h=handle, e=epoch: self._from_player(
                        h, e, {"type": EVENT_SPEAKING_STARTED}
                    ),
                    lambda rms, h=handle, e=epoch: self._from_player(
                        h, e, {"type": EVENT_SPEECH_LEVEL, "rms": rms}
                    ),
                )
                self._current_stop = None
                if not completed:
                    self._speaking = False  # clip was flushed mid-play
            except Exception as exc:
                # Speech must never take a turn down — text already rendered.
                self._current_stop = None
                logger.warning("speech failed, sentence dropped: %s", exc)
