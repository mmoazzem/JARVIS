"""Headless WebSocket entry point — the frontend walking skeleton.

Boot is main.py's ladder MINUS interactivity: config.yaml must already exist
(run the CLI once so the wizard commits it) and a missing model is an error
here, never a prompt. The browser is a client, not the launcher — this process
runs and is testable with no frontend attached.

Wire contract (one message in, one event stream out):
  client -> {"text": "<user message>"}
  server -> one JSON object per orchestrator event, verbatim
            (thinking / token / delegation / recovery / error / done)

The events are the SAME structured stream the CLI renders — the frontend is a
second consumer of respond(), not a fork of response logic. Turn capture
(memory Layer 1) and dev logging both live in core, so a browser turn logs
exactly like a CLI turn; this entry point only has to initialize the logging
config, as every entry point must.
"""
import asyncio
import atexit
import json
import logging
import sys

from websockets.asyncio.server import serve

from core.constants import (
    CONFIG_PATH,
    LOGGER_ROOT,
    STAGE_DAEMON_FAILED,
    STAGE_MODEL_MISSING,
    STAGE_MODEL_READY,
    STAGE_NOT_INSTALLED,
    WS_HOST,
    WS_PORT,
    WS_TEXT_KEY,
)
from core.credentials import load_credentials
from core.orchestrator.orchestrator import Orchestrator
from core.runtime.ollama_manager import ensure_ollama_ready, stop_owned_daemon
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


def _turn_handler(orchestrator: Orchestrator):
    async def handle(websocket) -> None:
        peer = websocket.remote_address
        log.info("client connected: %s", peer)
        try:
            async for message in websocket:
                try:
                    user_text = (json.loads(message).get(WS_TEXT_KEY) or "").strip()
                except (json.JSONDecodeError, AttributeError):
                    await websocket.send(json.dumps(
                        {"type": "error", "message": "expected {\"text\": ...}"}
                    ))
                    continue
                if not user_text:
                    continue
                log.info("turn from %s: %r", peer, user_text[:120])
                # Stream the core events verbatim — the browser decides rendering,
                # exactly as the CLI does. No response logic lives here.
                async for event in orchestrator.respond(user_text):
                    await websocket.send(json.dumps(event, ensure_ascii=False))
        finally:
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
