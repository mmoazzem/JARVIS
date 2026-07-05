"""Jarvis entry point — the real boot ceremony (Milestone 4).

Boot is a hard, gated pipeline — each stage is a precondition for the next, and
config.yaml is written at EXACTLY ONE point:
  config present?  no  -> wizard builds a config IN MEMORY (scan + recommend)
                   yes -> load the existing config
  then (both)      -> the shared daemon+model ladder (start Ollama if needed, check
                      the model against the LIVE daemon, offer a pull if missing)
  first run only   -> COMMIT: write config.yaml (now that setup truly succeeded)
  then             -> warm the model, start the chat loop.

Boot state is reported as structured BootEvents; a BootRenderer decides the wording
(the same events will drive a future frontend). Daemon ownership follows creation:
if we start Ollama, we stop it on every exit path via a single atexit teardown; if it
was already running, we leave it alone.
"""
import asyncio
import atexit
import logging
import sys

from core.constants import (
    BANNER_TEXT,
    CONFIG_PATH,
    LOGGER_ROOT,
    STAGE_DAEMON_FAILED,
    STAGE_MODEL_MISSING,
    STAGE_MODEL_READY,
    STAGE_NOT_INSTALLED,
    STAGE_PULL_DECLINED,
    STAGE_PULL_FAILED,
    STAGE_WARMING,
    STAGE_WARMUP_FAILED,
    STAGE_WARMUP_READY,
)
from core.runtime.ollama_manager import (
    BootEvent,
    ensure_ollama_ready,
    pull_model,
    stop_owned_daemon,
)
from setup import config as cfg
from setup.config import JarvisConfig
from setup.logging_setup import setup_logging
from setup.wizard import run_wizard
from core.orchestrator.orchestrator import Orchestrator
from interface.cli import BootRenderer, confirm_pull, run_chat

log = logging.getLogger(LOGGER_ROOT)


async def _drive_boot(config: JarvisConfig, render: BootRenderer) -> bool:
    """Run the gated daemon+model ladder, rendering each transition; return readiness.

    On a live model-miss it OFFERS the pull (never a silent auto-pull) and streams the
    download. Writes NO config — the commit is the caller's, only after this is True.
    """
    async for ev in ensure_ollama_ready(config.primary_model, config.ollama_base_url):
        render(ev)
        if ev.stage in (STAGE_NOT_INSTALLED, STAGE_DAEMON_FAILED):
            return False
        if ev.stage == STAGE_MODEL_READY:
            return True
        if ev.stage == STAGE_MODEL_MISSING:
            if not confirm_pull(config.primary_model):
                render(BootEvent(stage=STAGE_PULL_DECLINED, model=config.primary_model))
                return False
            async for pev in pull_model(config.primary_model, config.ollama_base_url):
                render(pev)
                if pev.stage == STAGE_MODEL_READY:
                    return True
                if pev.stage == STAGE_PULL_FAILED:
                    return False
            return False
    return False


async def _amain() -> None:
    setup_logging()
    render = BootRenderer()

    first_run = not CONFIG_PATH.exists()
    # First run builds the config in memory (no write); a later run loads the one that
    # a previous successful setup committed.
    config = await run_wizard() if first_run else cfg.load()

    if not await _drive_boot(config, render):
        # Every unmet precondition already rendered its next step. Exit non-zero; the
        # atexit teardown stops the daemon iff we started it.
        sys.exit(1)

    if first_run:
        # COMMIT POINT: Ollama is running AND the model is present — only now is it true
        # that "config.yaml on disk == setup completed", so only now do we write it.
        cfg.save(config)
        print(f"\n  Wrote config.yaml — mode={config.mode}, primary={config.primary_model}\n")

    orchestrator = Orchestrator(config)

    # Warm the model so the first question isn't a cold load. Sourced from warmup's
    # structured result and rendered as boot events (FIX C), not a bare print.
    if config.mode == "local":
        render(BootEvent(stage=STAGE_WARMING, model=config.primary_model))
        warmup = await orchestrator.warmup()
        if warmup.success:
            render(BootEvent(
                stage=STAGE_WARMUP_READY, model=warmup.model_id, elapsed_s=warmup.elapsed_s
            ))
        else:
            render(BootEvent(stage=STAGE_WARMUP_FAILED, detail=warmup.error or ""))
        log.info("warmup: %s", warmup.model_dump())

    print(BANNER_TEXT)
    await run_chat(orchestrator, config)
    log.info("Chat session ended.")


def main() -> None:
    # Ownership follows creation (FIX B): if we start the daemon, we stop it on ANY
    # exit path. atexit fires for normal return, sys.exit, and our caught Ctrl-C alike;
    # stop_owned_daemon is a no-op when the daemon was already running at boot.
    atexit.register(stop_owned_daemon)
    try:
        asyncio.run(_amain())
    except KeyboardInterrupt:
        # Ctrl-C during boot/pull (the chat loop handles its own Ctrl-C). No config is
        # written until setup fully succeeds, so nothing is left behind; the daemon is
        # cleaned up by the atexit teardown.
        print("\nSetup cancelled.")
        sys.exit(130)


if __name__ == "__main__":
    main()
