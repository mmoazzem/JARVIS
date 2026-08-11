"""
Server-side speech coordination — one audio device, N connections.

server.py cannot use SpeechController the way the CLI does. The CLI is one
person at one terminal: one turn, one speaker, nothing to arbitrate. The server
has N sockets that can finish turns in the same second and ONE sound card.

THE RULE: one turn speaks at a time, first claim wins, and a turn that starts
while another is speaking is answered SILENTLY. It is not queued and it does not
interrupt. Queued audio would play minutes late, reading an answer aloud to
whoever is sitting there for a question they stopped waiting on; interrupting
would mutilate an answer someone is already half-way through hearing. Silence is
the only outcome that is never wrong for the wrong person — and the text answer,
which is what was actually asked for, arrives identically either way.

The claim is taken at turn START, not when the first sentence is ready: a turn
that will speak owns the device for its whole length, so the device can never be
handed over in the gap between two sentences of the same answer.

Nothing is arbitrated across PROCESSES. The CLI and the server open separate
PulseAudio streams and the sink mixes them, so running both means two voices at
once. A cross-process lock (a lock file on the sink) is a different decision and
is not taken here — run one of the two.

Playback stays on the HOST: these sessions REPORT speech to the browser, which
never receives audio.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any, Awaitable, Callable, Optional

from core.constants import (
    EVENT_SPEAKING_STARTED,
    EVENT_SPEECH_DONE,
    EVENT_SPEECH_INTERRUPTED,
    EVENT_SPEECH_LEVEL,
    LOGGER_SPEECH,
    WS_AUDIO_LEVEL,
    WS_SPEECH_END,
    WS_SPEECH_PENDING,
    WS_SPEECH_START,
)
from core.memory.event_log import EventLog, Turn
from interface.audio import AudioUnavailableError
from interface.speech import SpeechController, build_controller

logger = logging.getLogger(LOGGER_SPEECH)

Sender = Callable[[dict], Awaitable[None]]


class SpeechSession:
    """One turn's claim on the audio device, and its reporting channel.

    It is the handle the speech pipeline stamps onto that turn's audio, so every
    event that comes back names the session it belongs to — including the ones
    that arrive after the turn's text is long finished.
    """

    def __init__(self, director: "SpeechDirector", turn: Optional[Turn], send: Sender) -> None:
        self.turn = turn
        self._director = director
        self._send = send
        self._closed = False
        # Sends are fired from the pipeline's synchronous callback, so they are
        # tasks; hold them or the loop may collect one mid-flight.
        self._sends: set[asyncio.Task] = set()

    # --- the turn's event stream ------------------------------------------------

    def feed(self, event: dict) -> None:
        """Pass one orchestrator event to the speech pipeline."""
        self._director.feed(event)

    # --- the pipeline's event stream --------------------------------------------

    def on_speech(self, event: dict) -> None:
        """Translate one pipeline event into the wire vocabulary."""
        kind = event.get("type")
        if kind == EVENT_SPEAKING_STARTED:
            self.emit({"type": WS_SPEECH_START})
        elif kind == EVENT_SPEECH_LEVEL:
            self.emit({"type": WS_AUDIO_LEVEL, "rms": event.get("rms", 0.0)})
        elif kind in (EVENT_SPEECH_DONE, EVENT_SPEECH_INTERRUPTED):
            self.close()

    def close(self) -> None:
        """Close the bracket and release the device. Idempotent — a turn that is
        interrupted and then reaches its end must not end twice."""
        if self._closed:
            return
        self._closed = True
        self.emit({"type": WS_SPEECH_END})
        self._director.release(self)

    # --- reporting ----------------------------------------------------------------

    def emit(self, event: dict) -> None:
        task = asyncio.create_task(self._send_quietly(event))
        self._sends.add(task)
        task.add_done_callback(self._sends.discard)

    async def _send_quietly(self, event: dict) -> None:
        try:
            await self._send(event)
        except Exception as exc:
            # A socket that went while audio was playing is ordinary: the sound
            # lives on the host and finishes (or is stopped) either way.
            logger.debug("speech frame not delivered (%s): %s", event.get("type"), exc)


class SpeechDirector:
    """Owns the process's one speech pipeline and hands it out a turn at a time."""

    def __init__(
        self,
        config,
        event_log: EventLog,
        build: Callable[..., Awaitable[SpeechController]] = build_controller,
    ) -> None:
        self._config = config
        self._event_log = event_log
        self._build = build
        self._controller: Optional[SpeechController] = None
        self._unavailable = False
        self._active: Optional[SpeechSession] = None

    async def open(self, turn: Optional[Turn], send: Sender) -> Optional[SpeechSession]:
        """Claim the device for one turn. None means this turn stays silent.

        None is not a failure the client needs to hear about: no session means no
        speech_pending, so the client already knows to rest at `done`.
        """
        if not self._config.tts_enabled or self._unavailable:
            return None
        if self._active is not None:
            logger.info("audio device busy — this turn is answered silently")
            return None
        controller = await self._ensure_controller()
        if controller is None:
            return None
        session = SpeechSession(self, turn, send)
        self._active = session
        controller.begin_turn(session)
        controller.ensure_started()
        await send({"type": WS_SPEECH_PENDING})
        return session

    def feed(self, event: dict) -> None:
        if self._controller is not None:
            self._controller.feed(event)

    def release(self, session: SpeechSession) -> None:
        if self._active is session:
            self._active = None

    def abort(self, session: Optional[SpeechSession]) -> None:
        """The listener has gone: stop the sound and close the bracket.

        The device is released here rather than waiting for the pipeline's own
        interruption event, because a turn abandoned before it ever spoke
        produces no event at all — and the next connection would wait forever
        for a device nobody is using.
        """
        if session is None or session is not self._active:
            return
        if self._controller is not None:
            self._controller.interrupt()
        session.close()

    # --- the pipeline ---------------------------------------------------------

    async def _ensure_controller(self) -> Optional[SpeechController]:
        if self._controller is not None:
            return self._controller
        try:
            self._controller = await self._build(self._config, self._on_speech)
        except (AudioUnavailableError, ValueError) as exc:
            # Reported once and then never retried: a machine with no sound card
            # will not grow one, and a warning per turn would bury the log.
            self._unavailable = True
            logger.warning("voice unavailable — turns stay silent: %s", exc)
            return None
        return self._controller

    def _on_speech(self, handle: Any, event: dict) -> None:
        """One pipeline event, routed by the handle that produced the audio."""
        session = handle if isinstance(handle, SpeechSession) else None
        # The event log is core-owned and must not depend on a socket being up:
        # it is fed first, and from the turn's own handle.
        self._event_log.feed_speech(session.turn if session else None, event)
        if session is not None:
            session.on_speech(event)
