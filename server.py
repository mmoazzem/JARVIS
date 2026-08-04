"""Headless WebSocket entry point — the frontend walking skeleton.

Boot is main.py's ladder MINUS interactivity: config.yaml must already exist
(run the CLI once so the wizard commits it) and a missing model is an error
here, never a prompt. The browser is a client, not the launcher — this process
runs and is testable with no frontend attached.

Wire contract (one message in, one event stream out):
  client -> {"text": "<user message>"}
  server -> one JSON object per orchestrator event, verbatim
            (thinking / token / delegation / recovery / error / done)
         -> plus `telemetry` frames pushed ~1 Hz OUTSIDE any turn, and a
            `memory` frame on connect and whenever profile.json changes —
            the two events this file originates rather than relays.

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

from websockets.asyncio.server import serve
from websockets.exceptions import ConnectionClosed

from core.constants import (
    CONFIG_PATH,
    LOGGER_ROOT,
    MEMORY_POLL_INTERVAL_S,
    STAGE_DAEMON_FAILED,
    STAGE_MODEL_MISSING,
    STAGE_MODEL_READY,
    STAGE_NOT_INSTALLED,
    TELEMETRY_INTERVAL_S,
    WS_HOST,
    WS_PORT,
    WS_TEXT_KEY,
)
from core.credentials import load_credentials
from core.memory.profile_view import profile_event, profile_stamp
from core.orchestrator.orchestrator import Orchestrator
from core.runtime.ollama_manager import ensure_ollama_ready, stop_owned_daemon
from core.runtime.telemetry import sample as sample_telemetry
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


def _turn_handler(orchestrator: Orchestrator):
    async def handle(websocket) -> None:
        peer = websocket.remote_address
        log.info("client connected: %s", peer)

        # Conversation state is per CONNECTION, by decision: a new socket starts
        # fresh and two browsers can never read each other's past. Only the
        # history is per-connection — the model, tools and warmup stay in the
        # process-wide Orchestrator, so a refresh costs a list, not a model load.
        # It is a local, so closing the socket releases it.
        conversation = orchestrator.new_conversation()

        # Two producers now share one socket (the turn stream and the telemetry
        # clock), so every send goes through one lock — a meter frame must never
        # land in the middle of a token stream.
        send_lock = asyncio.Lock()

        async def send(event: dict) -> None:
            async with send_lock:
                await websocket.send(json.dumps(event, ensure_ascii=False))

        telemetry = asyncio.create_task(_push_telemetry(websocket, send))
        memory = asyncio.create_task(_push_memory(send))
        try:
            async for message in websocket:
                try:
                    user_text = (json.loads(message).get(WS_TEXT_KEY) or "").strip()
                except (json.JSONDecodeError, AttributeError):
                    await send({"type": "error", "message": "expected {\"text\": ...}"})
                    continue
                if not user_text:
                    continue
                log.info("turn from %s: %r", peer, user_text[:120])
                # Stream the core events verbatim — the browser decides rendering,
                # exactly as the CLI does. No response logic lives here.
                async for event in orchestrator.respond(user_text, conversation):
                    await send(event)
        except ConnectionClosed:
            # A tab closing without a close frame is ordinary, not a crash —
            # one line, no traceback. Anything else still propagates.
            log.info("client closed the connection: %s", peer)
        finally:
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

    async with serve(_turn_handler(orchestrator), WS_HOST, WS_PORT):
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
