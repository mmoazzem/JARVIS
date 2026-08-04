"""
Host telemetry — CPU / RAM / GPU / VRAM as structured data.

Same defensive contract as setup/system_scan.py: a probe that cannot answer
yields an ABSENT field, never an exception. Absence is the whole point — a
meter reading 0% is indistinguishable from a dead sensor, so "unknown" is
expressed by omitting the key and letting the consumer decide what to show.

Sampling is COALESCED: every caller inside TELEMETRY_CACHE_S shares one
reading. Two browser tabs must not mean two nvidia-smi spawns per second, and
psutil's CPU percentage is a delta since the previous call — independent
callers would hand each other near-zero intervals and read garbage.
"""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Optional

import psutil
from pydantic import BaseModel

from core.constants import (
    LOGGER_ROOT,
    MB_PER_GB,
    NVIDIA_SMI_TELEMETRY_ARGS,
    NVIDIA_SMI_TIMEOUT_S,
    TELEMETRY_CACHE_S,
    TELEMETRY_EVENT_TYPE,
)

logger = logging.getLogger(f"{LOGGER_ROOT}.telemetry")


class Telemetry(BaseModel):
    """One instant of host load. Every field optional: None means "could not
    measure", which is NOT the same as zero and must not be sent as one."""

    cpu: Optional[int] = None
    ram: Optional[int] = None
    gpu: Optional[int] = None
    vram_used_gb: Optional[float] = None
    vram_total_gb: Optional[float] = None

    def as_event(self) -> dict:
        """The wire shape: an event like any other, unmeasured keys dropped."""
        return {"type": TELEMETRY_EVENT_TYPE, **self.model_dump(exclude_none=True)}


# A source that breaks would otherwise log once per second forever; warn on the
# first failure of each kind, then stay quiet until it recovers.
_warned: set[str] = set()


def _warn_once(key: str, message: str) -> None:
    if key in _warned:
        logger.debug("%s (still failing)", message)
        return
    _warned.add(key)
    logger.warning(message)


def _recovered(key: str) -> None:
    if key in _warned:
        _warned.discard(key)
        logger.info("telemetry source recovered: %s", key)


def _sample_host() -> tuple[Optional[int], Optional[int]]:
    """CPU% and RAM% from psutil. Both are non-blocking /proc reads (interval=None
    returns the delta since the previous call), so they stay on the event loop."""
    try:
        cpu = round(psutil.cpu_percent(interval=None))
        ram = round(psutil.virtual_memory().percent)
    except Exception as exc:  # a sensor going away is data, not a crash
        _warn_once("psutil", f"telemetry: psutil unavailable: {exc}")
        return None, None
    _recovered("psutil")
    return cpu, ram


async def _sample_gpu() -> tuple[Optional[int], Optional[float], Optional[float]]:
    """GPU utilisation and VRAM from nvidia-smi, spawned as a subprocess so the
    read never blocks the loop. Returns (util%, used GB, total GB)."""
    try:
        proc = await asyncio.create_subprocess_exec(
            *NVIDIA_SMI_TELEMETRY_ARGS,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(
            proc.communicate(), timeout=NVIDIA_SMI_TIMEOUT_S
        )
    except asyncio.CancelledError:
        raise
    except Exception as exc:  # binary missing, spawn failure, timeout
        _warn_once("nvidia-smi", f"telemetry: nvidia-smi probe failed: {exc}")
        return None, None, None

    if proc.returncode != 0:
        _warn_once(
            "nvidia-smi",
            f"telemetry: nvidia-smi exited {proc.returncode}: {stderr.decode().strip()}",
        )
        return None, None, None

    rows = stdout.decode().strip().splitlines()
    if not rows:
        _warn_once("nvidia-smi", "telemetry: nvidia-smi returned no GPU rows")
        return None, None, None

    # First GPU only: "12, 1816, 16303" (util %, used MiB, total MiB)
    parts = [p.strip() for p in rows[0].split(",")]
    try:
        util = int(parts[0])
        used_gb = round(int(parts[1]) / MB_PER_GB, 1)
        total_gb = round(int(parts[2]) / MB_PER_GB, 1)
    except (IndexError, ValueError) as exc:
        _warn_once("nvidia-smi", f"telemetry: could not parse {rows[0]!r}: {exc}")
        return None, None, None

    _recovered("nvidia-smi")
    return util, used_gb, total_gb


_sample_lock = asyncio.Lock()
_cached: Optional[Telemetry] = None
_cached_at = 0.0

# psutil's first cpu_percent() call has no previous call to measure against and
# returns a meaningless 0.0 — prime it at import so the first real sample is real.
psutil.cpu_percent(interval=None)


async def sample() -> Telemetry:
    """Current host load. Never raises; never fabricates a missing reading."""
    global _cached, _cached_at
    async with _sample_lock:
        now = time.monotonic()
        if _cached is not None and now - _cached_at < TELEMETRY_CACHE_S:
            return _cached
        cpu, ram = _sample_host()
        gpu, vram_used, vram_total = await _sample_gpu()
        _cached = Telemetry(
            cpu=cpu, ram=ram, gpu=gpu,
            vram_used_gb=vram_used, vram_total_gb=vram_total,
        )
        _cached_at = time.monotonic()
        return _cached
