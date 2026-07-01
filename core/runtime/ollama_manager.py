"""
Ollama runtime manager — DETECT, REMEDIATE, and REPORT; never INSTALL.

The bootstrap bash script installs the Ollama *binary*; this module only manages its
runtime state: is it installed, is the daemon up (start it if not — and remember that
WE started it), is the required model pulled (stream a pull on request).

Two M4 principles shape this file:
  * Presentation-free (FIX C). Nothing here prints. Boot progress is REPORTED as a
    stream of structured `BootEvent`s, exactly like the orchestrator's respond()
    event stream — the interface (CLI now, frontend later) decides the wording.
  * Ownership follows creation (FIX B). We stop the daemon IF AND ONLY IF we started
    it; a daemon that was already running belongs to the system and is left alone.
    `stop_owned_daemon` is sync + idempotent so it is safe from atexit / a signal path.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import signal
import subprocess
import time
from dataclasses import dataclass
from typing import AsyncIterator, Optional

import httpx
from pydantic import BaseModel

from core.constants import (
    LOGGER_ROOT,
    OLLAMA_BINARY,
    OLLAMA_DAEMON_FAILED_DETAIL,
    OLLAMA_DEFAULT_BASE_URL,
    OLLAMA_ENDPOINT_PULL,
    OLLAMA_ENDPOINT_TAGS,
    OLLAMA_NOT_INSTALLED_DETAIL,
    OLLAMA_POLL_INTERVAL_S,
    OLLAMA_PROBE_TIMEOUT_S,
    OLLAMA_SERVE_ARGS,
    OLLAMA_START_TIMEOUT_S,
    OLLAMA_SYSTEMCTL_START_ARGS,
    OLLAMA_SYSTEMCTL_STOP_ARGS,
    STAGE_DAEMON_FAILED,
    STAGE_DAEMON_READY,
    STAGE_MODEL_MISSING,
    STAGE_MODEL_READY,
    STAGE_NOT_INSTALLED,
    STAGE_PULL_FAILED,
    STAGE_PULLING_MODEL,
    STAGE_STARTING_DAEMON,
)

logger = logging.getLogger(LOGGER_ROOT)


class BootEvent(BaseModel):
    """One boot state transition, reported as data for the interface to render.

    `stage` is the discriminator (a STAGE_* constant); the other fields carry only
    what a given stage needs. Mirrors the respond() event dicts in spirit.
    """

    stage: str
    detail: str = ""
    model: Optional[str] = None
    progress: Optional[float] = None  # percent 0..100, for STAGE_PULLING_MODEL
    elapsed_s: Optional[float] = None  # for STAGE_WARMUP_READY


@dataclass
class _DaemonHandle:
    """How WE started the daemon, so we can stop exactly that later."""

    method: str  # "systemctl" | "serve"
    pid: Optional[int] = None  # the `ollama serve` pid (serve method only)


# Set ONLY when this process starts the daemon. Module-level so a single teardown
# (atexit) can honour ownership regardless of which exit path fires.
_owned_daemon: Optional[_DaemonHandle] = None


async def _probe_tags(base_url: str) -> Optional[list[str]]:
    """Return pulled model names if the daemon answers, else None (= not running)."""
    try:
        async with httpx.AsyncClient(timeout=OLLAMA_PROBE_TIMEOUT_S) as client:
            resp = await client.get(f"{base_url.rstrip('/')}{OLLAMA_ENDPOINT_TAGS}")
        if resp.status_code == 200:
            return [m["name"] for m in resp.json().get("models", [])]
    except Exception:
        # Connection refused / timeout — the daemon isn't up yet. Not an error here.
        pass
    return None


async def _start_daemon() -> Optional[_DaemonHandle]:
    """Start the Ollama daemon: try systemctl, then fall back to `ollama serve`.

    Returns a handle describing HOW it was started (so teardown can stop that exact
    thing), or None if no start could be attempted. Readiness is confirmed by polling.
    """
    # Preferred: a managed service, if this box runs one (it survives our process).
    try:
        proc = await asyncio.create_subprocess_exec(
            *OLLAMA_SYSTEMCTL_START_ARGS,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        rc = await proc.wait()
        if rc == 0:
            logger.info("started ollama via systemctl")
            return _DaemonHandle(method="systemctl")
        logger.info("systemctl start ollama exited %s — falling back to serve", rc)
    except Exception as exc:
        logger.info("systemctl unavailable (%s) — falling back to serve", exc)

    # Fallback: spawn `ollama serve` detached so it outlives this boot sequence; keep
    # its pid so we can stop it if WE own it.
    try:
        proc = await asyncio.create_subprocess_exec(
            *OLLAMA_SERVE_ARGS,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
            start_new_session=True,
        )
        logger.info("spawned `ollama serve` (pid %s)", proc.pid)
        return _DaemonHandle(method="serve", pid=proc.pid)
    except Exception as exc:
        logger.error("failed to spawn `ollama serve`: %s", exc)
        return None


async def _wait_until_running(base_url: str) -> Optional[list[str]]:
    """Poll /api/tags for up to OLLAMA_START_TIMEOUT_S. Return models or None."""
    deadline = time.monotonic() + OLLAMA_START_TIMEOUT_S
    while time.monotonic() < deadline:
        models = await _probe_tags(base_url)
        if models is not None:
            return models
        await asyncio.sleep(OLLAMA_POLL_INTERVAL_S)
    return None


def stop_owned_daemon() -> bool:
    """Stop the daemon IFF Jarvis started it (FIX B). No-op otherwise.

    Sync and idempotent on purpose: it runs from atexit on every exit path (normal
    /exit, sys.exit after a failed precondition, Ctrl-C) without needing a live event
    loop. Returns True if it stopped something.
    """
    global _owned_daemon
    handle = _owned_daemon
    if handle is None:
        return False  # we didn't start it → the system owns it → leave it running
    _owned_daemon = None  # idempotent: only the first call acts
    try:
        if handle.method == "systemctl":
            subprocess.run(
                OLLAMA_SYSTEMCTL_STOP_ARGS,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
        elif handle.pid is not None:
            os.kill(handle.pid, signal.SIGTERM)
        logger.info("stopped the ollama daemon we started (%s)", handle)
        return True
    except ProcessLookupError:
        return False  # already gone
    except Exception as exc:
        logger.warning("failed to stop owned daemon (%s): %s", handle, exc)
        return False


async def ensure_ollama_ready(
    required_model: str, base_url: str = OLLAMA_DEFAULT_BASE_URL
) -> AsyncIterator[BootEvent]:
    """Walk the gated daemon ladder, yielding a BootEvent per transition.

    The final event's stage is terminal for the boot gate:
      * not_installed / daemon_failed -> not ready (caller messages + exits).
      * model_missing                 -> ready pending a pull (caller offers one).
      * model_ready                   -> ready.

    Model presence is checked against the LIVE daemon AFTER it is confirmed up — never
    from a pre-start scan, whose list is stale/unknown if the daemon was down (FIX A).
    Never raises for an expected failure mode.
    """
    global _owned_daemon

    # Rung 1: the binary must exist. We never install it ourselves.
    if shutil.which(OLLAMA_BINARY) is None:
        yield BootEvent(stage=STAGE_NOT_INSTALLED, detail=OLLAMA_NOT_INSTALLED_DETAIL)
        return

    # Rung 2: is the daemon answering? If not, start it (and remember we own it), poll.
    models = await _probe_tags(base_url)
    if models is None:
        yield BootEvent(stage=STAGE_STARTING_DAEMON)
        handle = await _start_daemon()
        if handle is not None:
            _owned_daemon = handle
        models = await _wait_until_running(base_url)
        if models is None:
            yield BootEvent(stage=STAGE_DAEMON_FAILED, detail=OLLAMA_DAEMON_FAILED_DETAIL)
            return
        yield BootEvent(stage=STAGE_DAEMON_READY)
    # (If it was already up, started_it stays False — the system owns it, no line.)

    # Rung 3: authoritative LIVE presence check against the running daemon.
    if required_model in models:
        yield BootEvent(
            stage=STAGE_MODEL_READY,
            model=required_model,
            detail=f"Model '{required_model}' present.",
        )
    else:
        yield BootEvent(
            stage=STAGE_MODEL_MISSING,
            model=required_model,
            detail=f"Model '{required_model}' is not pulled yet",
        )


async def pull_model(
    name: str, base_url: str = OLLAMA_DEFAULT_BASE_URL
) -> AsyncIterator[BootEvent]:
    """Stream a model download as BootEvents: pulling_model (+progress) then either
    model_ready or pull_failed.

    Pulling can take many minutes, so NO read timeout is imposed (only a short connect
    timeout). Never raises for an expected failure (network, bad model name).
    """
    url = f"{base_url.rstrip('/')}{OLLAMA_ENDPOINT_PULL}"
    timeout = httpx.Timeout(
        connect=OLLAMA_PROBE_TIMEOUT_S, read=None, write=None, pool=None
    )
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            async with client.stream(
                "POST", url, json={"model": name, "stream": True}
            ) as resp:
                if resp.status_code != 200:
                    body = (await resp.aread()).decode().strip()
                    yield BootEvent(
                        stage=STAGE_PULL_FAILED,
                        model=name,
                        detail=f"pull returned {resp.status_code}: {body}",
                    )
                    return
                async for line in resp.aiter_lines():
                    if not line.strip():
                        continue
                    event = json.loads(line)
                    # Ollama signals a hard failure inline rather than via status.
                    if event.get("error"):
                        yield BootEvent(
                            stage=STAGE_PULL_FAILED, model=name, detail=event["error"]
                        )
                        return
                    total = event.get("total")
                    completed = event.get("completed")
                    pct = completed / total * 100 if (total and completed) else None
                    yield BootEvent(
                        stage=STAGE_PULLING_MODEL,
                        model=name,
                        detail=event.get("status", ""),
                        progress=pct,
                    )
        logger.info("pull complete [%s]", name)
        yield BootEvent(stage=STAGE_MODEL_READY, model=name, detail=f"{name} pulled")
    except Exception as exc:
        logger.error("pull failed [%s]: %s", name, exc)
        yield BootEvent(stage=STAGE_PULL_FAILED, model=name, detail=str(exc))
