"""Headless WebSocket entry point — the frontend walking skeleton.

Boot is main.py's ladder MINUS interactivity: config.yaml must already exist
(run the CLI once so the wizard commits it) and a missing model is an error
here, never a prompt. The browser is a client, not the launcher — this process
runs and is testable with no frontend attached.

Wire contract (one message in, one event stream out):
  client -> {"text": "<user message>"}            — a turn to answer
         -> {"op": "set_preference", "key": "units.temperature",
             "value": "celsius"}                  — an operation, never chat
         -> {"op": "dismiss_session", "id": "<session_id>"} — HIDES a session
            from the panel. Removes nothing: the JSONL is append-only ground
            truth and Layer 2 has already digested part of it.
  server -> one JSON object per orchestrator event, verbatim
            (thinking / token / delegation / recovery / error / done)
         -> plus `telemetry` frames pushed ~1 Hz OUTSIDE any turn, a `memory`
            frame on connect and whenever profile.json changes, and a `weather`
            frame on connect and on the shared 10-minute poll, and a `sessions`
            frame on every connect, disconnect and completed turn — the events
            this file originates rather than relays.

Each connection mints a `session_id` (uuid4) and every event-log record it
writes carries it, so the SESSIONS panel reports real conversations instead of
inferring them from timestamp gaps. Only the ASKING connection's session is
`live` in its own frame.
         -> plus the speech bracket (speech_pending / speech_start /
            audio_level / speech_end) for a turn that claimed the audio device,
            and a `preference` ack for an applied settings write.

A message carrying `op` is ROUTED ON THAT KEY and never reaches the model: an
operation the server does not know is an error frame, not something to answer
conversationally.

Speech PLAYS ON THIS HOST — browser and server are the same machine, so the
sound comes out of the speakers here and the browser is only told about it. No
audio crosses the socket. Which turns may speak, and what happens when two want
to at once, is interface/speech_session.py's decision.

Each connection gets its own conversation: a reload starts fresh, and two open
tabs cannot see each other's history. There is no resume — closing the socket
discards that past.

The events are the SAME structured stream the CLI renders — the frontend is a
second consumer of respond(), not a fork of response logic. Turn capture
(memory Layer 1) and dev logging both live in core, so a browser turn logs
exactly like a CLI turn; this entry point only has to initialize the logging
config, as every entry point must.
"""
import asyncio
import atexit
import contextlib
import json
import logging
import sys
import uuid

from websockets.asyncio.server import serve
from websockets.exceptions import ConnectionClosed

from core.constants import (
    CONFIG_PATH,
    LOGGER_ROOT,
    MEMORY_POLL_INTERVAL_S,
    PREFERENCE_KEY_SEPARATOR,
    PREFERENCE_UNITS,
    STAGE_DAEMON_FAILED,
    STAGE_MODEL_MISSING,
    STAGE_MODEL_READY,
    STAGE_NOT_INSTALLED,
    TELEMETRY_INTERVAL_S,
    WS_HOST,
    WS_OP_DISMISS_SESSION,
    WS_OP_ID_FIELD,
    WS_OP_KEY,
    WS_OP_KEY_FIELD,
    WS_OP_SET_PREFERENCE,
    WS_OP_VALUE_FIELD,
    WS_PORT,
    WS_PREFERENCE_EVENT_TYPE,
    WS_TEXT_KEY,
)
from core.credentials import load_credentials
from core.preferences import PreferenceError
from core.memory.profile_view import profile_event, profile_stamp
from core.memory.sessions_view import dismiss as dismiss_session, sessions_event
from core.orchestrator.orchestrator import Orchestrator
from core.runtime.ollama_manager import ensure_ollama_ready, stop_owned_daemon
from core.runtime.telemetry import sample as sample_telemetry
from core.runtime.weather_feed import WeatherFeed
from core.runtime.weather_view import weather_event
from interface.speech_session import SpeechDirector
from setup import config as cfg
from setup.logging_setup import setup_logging

log = logging.getLogger(f"{LOGGER_ROOT}.web")


async def _headless_boot(config) -> bool:
    """The daemon+model ladder with no prompts: every unmet precondition is a
    logged failure telling the user to finish setup in the CLI first."""
    async for ev in ensure_ollama_ready(config.primary_model, config.ollama_base_url):
        if ev.stage == STAGE_MODEL_READY:
            return True
        if ev.stage in (STAGE_NOT_INSTALLED, STAGE_DAEMON_FAILED, STAGE_MODEL_MISSING):
            log.error("headless boot failed at %s: %s", ev.stage, ev.detail)
            print(f"Not ready ({ev.stage}): {ev.detail}")
            print("Run the CLI (python main.py) once to complete setup, then retry.")
            return False
    return False


async def _push_telemetry(websocket, send) -> None:
    """Ambient meter feed: pushes host load on its own clock, with no turn
    involved. Cancelled by the handler when the socket goes; a source that
    cannot answer omits its key (telemetry.sample never raises), so this loop
    keeps running even with the GPU probe dead."""
    failed = False
    while True:
        await asyncio.sleep(TELEMETRY_INTERVAL_S)
        try:
            event = (await sample_telemetry()).as_event()
        except Exception:  # belt and braces: sampling is meant to never raise
            if not failed:
                failed = True
                log.exception("telemetry sampling failed — meters will hold")
            continue
        failed = False
        await send(event)


async def _push_memory(send) -> None:
    """Ambient MEMORY feed: the profile on connect, and again when it changes.

    Shares the telemetry pusher's machinery — the same lock-guarded send, the
    same cancel-on-close — but not its clock: telemetry is a sample of a value
    that is always moving, while the profile only moves when a digest rewrites
    it. So this is edge-triggered off a stat, and the file is READ only when
    the stat says it moved (profile_event caches, so N tabs cost one read).
    """
    stamp = profile_stamp()
    await send(profile_event())
    while True:
        await asyncio.sleep(MEMORY_POLL_INTERVAL_S)
        current = profile_stamp()
        if current == stamp:
            continue
        stamp = current
        # A profile that vanished or went malformed is an EMPTY memory here,
        # never an exception: profile_event() answers for both, so the socket
        # survives a torn file mid-merge.
        await send(profile_event())


class _SessionHub:
    """Who is connected right now, and what each of them should be told.

    The frame is built PER RECIPIENT, not once and fanned out: `live` names the
    asking connection's own session, so one shared payload would tell every tab
    that someone else's conversation was theirs.
    """

    def __init__(self) -> None:
        self._connections: dict[str, object] = {}

    def add(self, session_id: str, send) -> None:
        self._connections[session_id] = send

    def remove(self, session_id: str) -> None:
        self._connections.pop(session_id, None)

    async def frame_for(self, session_id: str) -> dict:
        # Reading every day-file is disk work; off the event loop it goes.
        return await asyncio.to_thread(
            sessions_event, session_id, open_ids=list(self._connections))

    async def broadcast(self) -> None:
        """Tell every open dashboard — a connect, a disconnect or a finished
        turn changes what all of them should be showing, not just one."""
        for session_id, send in list(self._connections.items()):
            try:
                await send(await self.frame_for(session_id))
            except Exception as exc:
                log.info("sessions: dropping a connection that would not take "
                         "the frame (%s)", exc)
                self._connections.pop(session_id, None)


async def _dismiss_session(payload: dict, session_id: str, hub: _SessionHub, send) -> None:
    """Hide one stored session from the panel. Removes NO records.

    The live session is refused here rather than trusted to the UI: the button
    is omitted for it, but the op is a message anyone can send, and hiding the
    conversation being written to would make the panel disagree with itself.
    """
    target = payload.get(WS_OP_ID_FIELD)
    if not isinstance(target, str) or not target:
        await send({"type": "error", "message": "dismiss_session needs an id"})
        return
    if target == session_id:
        await send({"type": "error", "message": "refusing to dismiss the live session"})
        return
    await asyncio.to_thread(dismiss_session, target)
    await hub.broadcast()


async def _handle_op(orchestrator: Orchestrator, weather: WeatherFeed,
                     hub: _SessionHub, payload: dict, session_id: str, send) -> None:
    """Apply one client operation. Never raises — a refusal is an error frame.

    Validation lives in the preference store, not here: the server must not be
    an arbitrary write into that file, and the store is the only place that
    knows which keys and values exist.
    """
    op = payload.get(WS_OP_KEY)
    if op == WS_OP_DISMISS_SESSION:
        await _dismiss_session(payload, session_id, hub, send)
        return
    if op != WS_OP_SET_PREFERENCE:
        await send({"type": "error", "message": f"unknown op: {op!r}"})
        return
    key = payload.get(WS_OP_KEY_FIELD)
    value = payload.get(WS_OP_VALUE_FIELD)
    try:
        preferences = orchestrator.preferences.set(key, value)
    except PreferenceError as exc:
        log.info("rejected preference write %r=%r: %s", key, value, exc)
        await send({"type": "error", "message": str(exc)})
        return
    log.info("preference %s set to %r", key, value)
    await send({"type": WS_PREFERENCE_EVENT_TYPE, "preferences": preferences})

    if str(key).startswith(f"{PREFERENCE_UNITS}{PREFERENCE_KEY_SEPARATOR}"):
        # The stored preference is already true; this only makes the HELD
        # reading agree with it. A failed refresh must not un-apply the write —
        # the next tool call refetches anyway, since a reading fetched under
        # different units can never be served.
        #
        # Every dashboard is told, not just the one that asked: the unit is a
        # property of the reading, so a second tab showing the old unit would be
        # showing a number that is no longer labelled correctly.
        try:
            await weather.broadcast(await weather_event(orchestrator.weather, refresh=True))
        except Exception as exc:
            log.warning("preference applied, but the weather refresh failed: %s", exc)


def _turn_handler(orchestrator: Orchestrator, speech: SpeechDirector,
                  weather: WeatherFeed, hub: _SessionHub):
    async def handle(websocket) -> None:
        peer = websocket.remote_address
        log.info("client connected: %s", peer)

        # Conversation state is per CONNECTION, by decision: a new socket starts
        # fresh and two browsers can never read each other's past. Only the
        # history is per-connection — the model, tools and warmup stay in the
        # process-wide Orchestrator, so a refresh costs a list, not a model load.
        # It is a local, so closing the socket releases it.
        conversation = orchestrator.new_conversation()

        # A session IS a connection here, and it is minted rather than inferred:
        # every record this connection writes carries this id, so the SESSIONS
        # panel reports conversations instead of guessing them from time gaps.
        session_id = str(uuid.uuid4())

        # Two producers now share one socket (the turn stream and the telemetry
        # clock), so every send goes through one lock — a meter frame must never
        # land in the middle of a token stream.
        send_lock = asyncio.Lock()

        async def send(event: dict) -> None:
            async with send_lock:
                await websocket.send(json.dumps(event, ensure_ascii=False))

        telemetry = asyncio.create_task(_push_telemetry(websocket, send))
        memory = asyncio.create_task(_push_memory(send))
        # Weather is the one ambient feed that leaves the machine, so it is
        # shared: this connection joins the poll rather than starting its own.
        await weather.subscribe(send)
        # Joining changes what EVERY panel should show, not just this one's.
        hub.add(session_id, send)
        await hub.broadcast()
        # The turn's speech OUTLIVES the turn's text, so this is held at
        # connection scope: it is what a closing socket has to stop.
        session = None
        try:
            async for message in websocket:
                try:
                    payload = json.loads(message)
                    # An op is routed on its key BEFORE any chat handling, so a
                    # settings write can never be mistaken for something to answer.
                    is_op = WS_OP_KEY in payload
                    user_text = "" if is_op else (payload.get(WS_TEXT_KEY) or "").strip()
                except (json.JSONDecodeError, AttributeError, TypeError):
                    await send({"type": "error", "message": "expected {\"text\": ...}"})
                    continue
                if is_op:
                    await _handle_op(orchestrator, weather, hub, payload, session_id, send)
                    continue
                if not user_text:
                    continue
                log.info("turn from %s: %r", peer, user_text[:120])
                # The turn handle is opened here, not inside respond(), because
                # speech needs to name the turn its audio belongs to.
                turn = orchestrator.event_log.begin_turn(user_text, session_id)
                session = await speech.open(turn, send)
                # Stream the core events verbatim — the browser decides rendering,
                # exactly as the CLI does. No response logic lives here. Speech is
                # fed first: it is non-blocking, and every event it gets earlier
                # is time the listener is not waiting on silence.
                async for event in orchestrator.respond(user_text, conversation, turn=turn):
                    if session is not None:
                        session.feed(event)
                    await send(event)
                # The turn just landed on disk, so this session's turn count
                # moved — every panel is told rather than left to poll for it.
                await hub.broadcast()
        except ConnectionClosed:
            # A tab closing without a close frame is ordinary, not a crash —
            # one line, no traceback. Anything else still propagates.
            log.info("client closed the connection: %s", peer)
        finally:
            speech.abort(session)  # no listener left — stop the sound, free the device
            weather.unsubscribe(send)
            hub.remove(session_id)
            await hub.broadcast()
            for task in (telemetry, memory):
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError, ConnectionClosed):
                    await task
            log.info("client disconnected: %s", peer)

    return handle


async def _amain() -> None:
    # New entry point, same rule as main.py: logging config is initialized at
    # startup or this process's logs go nowhere.
    setup_logging()
    load_credentials()

    if not CONFIG_PATH.exists():
        print("No config.yaml — run the CLI (python main.py) once to complete setup.")
        sys.exit(1)
    config = cfg.load()

    if not await _headless_boot(config):
        sys.exit(1)

    orchestrator = Orchestrator(config)
    warmup = await orchestrator.warmup()
    log.info("warmup: %s", warmup.model_dump())

    # One device, one pipeline, process-wide — every connection speaks through
    # this or not at all. Built lazily on the first turn that wants it, so a
    # text-only run never loads a voice.
    speech = SpeechDirector(config, orchestrator.event_log)

    # One poll for the whole process, running only while a dashboard is open.
    weather = WeatherFeed(orchestrator.weather)
    # Which conversations are open right now — only the connections know.
    hub = _SessionHub()

    async with serve(_turn_handler(orchestrator, speech, weather, hub), WS_HOST, WS_PORT):
        print(f"Jarvis WebSocket server on ws://{WS_HOST}:{WS_PORT} "
              f"(model {config.primary_model}). Ctrl-C stops it.")
        log.info("serving on %s:%s", WS_HOST, WS_PORT)
        await asyncio.get_running_loop().create_future()  # run until cancelled


def main() -> None:
    # Ownership follows creation, same as the CLI: stop the daemon iff we started it.
    atexit.register(stop_owned_daemon)
    try:
        asyncio.run(_amain())
    except KeyboardInterrupt:
        print("\nServer stopped.")


if __name__ == "__main__":
    main()
