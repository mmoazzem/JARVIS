"""
Pure system scan — answers "what is this machine?" as structured data.

Every probe here is DEFENSIVE: a missing tool (no nvidia-smi), a non-zero exit, or
an unreachable daemon becomes a None/empty field plus a noted reason — never an
exception out of this module. The wizard/boot layer decides how to present any of
it; this file only reports. (See M4 architecture rules in the milestone brief.)
"""
from __future__ import annotations

import asyncio
import logging
import platform
import shutil
from typing import Optional

import httpx
import psutil
from pydantic import BaseModel

from core.constants import (
    BYTES_PER_GB,
    LOGGER_ROOT,
    MB_PER_GB,
    NVIDIA_SMI_BINARY,
    NVIDIA_SMI_QUERY_ARGS,
    NVIDIA_SMI_TIMEOUT_S,
    OLLAMA_BINARY,
    OLLAMA_DEFAULT_BASE_URL,
    OLLAMA_ENDPOINT_TAGS,
    OLLAMA_PROBE_TIMEOUT_S,
)

logger = logging.getLogger(LOGGER_ROOT)


class SystemReport(BaseModel):
    """A snapshot of the host, as data. Optional fields are None when a probe could
    not answer; `notes` carries the human-readable reason for each gap."""

    os_system: str
    os_release: str
    platform_str: str
    python_version: str
    cpu_cores: Optional[int] = None
    ram_total_gb: Optional[float] = None
    gpu_name: Optional[str] = None
    gpu_vram_gb: Optional[float] = None
    ollama_installed: bool = False
    ollama_running: bool = False
    # None = we could NOT query (daemon down) — distinct from [] = queried, none found.
    # This list is informational (the wizard rundown) only; the boot gate re-checks
    # presence against the live daemon after starting it (FIX A).
    pulled_models: Optional[list[str]] = None
    notes: list[str] = []


async def _probe_gpu(notes: list[str]) -> tuple[Optional[str], Optional[float]]:
    """Read GPU name + total VRAM from nvidia-smi. No GPU / no driver → (None, None)."""
    if shutil.which(NVIDIA_SMI_BINARY) is None:
        notes.append("nvidia-smi not found — no NVIDIA GPU detected")
        return None, None
    try:
        proc = await asyncio.create_subprocess_exec(
            *NVIDIA_SMI_QUERY_ARGS,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(
            proc.communicate(), timeout=NVIDIA_SMI_TIMEOUT_S
        )
    except Exception as exc:  # timeout, spawn failure — degrade to "unknown", no raise
        notes.append(f"nvidia-smi probe failed: {exc}")
        return None, None

    if proc.returncode != 0:
        notes.append(f"nvidia-smi exited {proc.returncode}: {stderr.decode().strip()}")
        return None, None

    # First line is the first GPU: "NVIDIA GeForce RTX 5080, 16303"
    first = stdout.decode().strip().splitlines()
    if not first:
        notes.append("nvidia-smi returned no GPU rows")
        return None, None
    parts = [p.strip() for p in first[0].split(",")]
    name = parts[0] if parts else None
    vram_gb = None
    if len(parts) >= 2:
        try:
            vram_gb = round(int(parts[1]) / MB_PER_GB, 1)
        except ValueError:
            notes.append(f"could not parse VRAM from nvidia-smi: {parts[1]!r}")
    return name, vram_gb


async def _probe_ollama(
    base_url: str, notes: list[str]
) -> tuple[bool, bool, Optional[list[str]]]:
    """Return (installed, running, pulled_models).

    Installed is decided by the binary on PATH; running + the model list come from a
    single /api/tags call so one probe answers both. When the daemon can't be reached
    the model list is None (UNKNOWN — we couldn't ask), never [] (which would falsely
    claim "queried, none pulled" — the stale-scan bug, FIX A).
    """
    installed = shutil.which(OLLAMA_BINARY) is not None
    if not installed:
        notes.append("ollama binary not on PATH")

    running = False
    models: Optional[list[str]] = None
    try:
        async with httpx.AsyncClient(timeout=OLLAMA_PROBE_TIMEOUT_S) as client:
            resp = await client.get(f"{base_url.rstrip('/')}{OLLAMA_ENDPOINT_TAGS}")
        if resp.status_code == 200:
            running = True
            models = [m["name"] for m in resp.json().get("models", [])]
        else:
            notes.append(f"ollama /api/tags returned {resp.status_code}")
    except Exception as exc:  # daemon down or unreachable — data, not an error
        notes.append(f"ollama not reachable at {base_url}: {exc}")

    return installed, running, models


async def scan_system(base_url: str = OLLAMA_DEFAULT_BASE_URL) -> SystemReport:
    """Probe the host and return a SystemReport. Never raises for expected gaps."""
    notes: list[str] = []

    cpu_cores = psutil.cpu_count(logical=True)
    ram_total_gb = round(psutil.virtual_memory().total / BYTES_PER_GB, 1)

    # GPU and Ollama probes both do I/O — run them concurrently.
    (gpu_name, gpu_vram_gb), (installed, running, models) = await asyncio.gather(
        _probe_gpu(notes),
        _probe_ollama(base_url, notes),
    )

    report = SystemReport(
        os_system=platform.system(),
        os_release=platform.release(),
        platform_str=platform.platform(),
        python_version=platform.python_version(),
        cpu_cores=cpu_cores,
        ram_total_gb=ram_total_gb,
        gpu_name=gpu_name,
        gpu_vram_gb=gpu_vram_gb,
        ollama_installed=installed,
        ollama_running=running,
        pulled_models=models,
        notes=notes,
    )
    logger.info("system scan: %s", report.model_dump())
    return report
