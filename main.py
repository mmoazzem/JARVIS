"""Jarvis entry point — boot config + logging, then run the chat loop.

Milestone 3: assumes Ollama is up and the model is pulled. A full boot ceremony
(wizard, health gating) arrives in Milestone 4; warmup here is opportunistic.
"""
import asyncio
import logging

from core.constants import BANNER_TEXT, CONFIG_PATH, LOGGER_ROOT
from setup import config as cfg
from setup.logging_setup import setup_logging
from core.orchestrator.orchestrator import Orchestrator
from interface.cli import run_chat


async def _amain() -> None:
    setup_logging()
    log = logging.getLogger(LOGGER_ROOT)

    if not CONFIG_PATH.exists():
        log.error("config.yaml not found — run the setup wizard (Milestone 4) first")
        return

    config = cfg.load()
    orchestrator = Orchestrator(config)

    # Opportunistic warmup so the first turn isn't a cold load. Full boot is M4.
    warmup = await orchestrator.warmup()
    log.info("warmup: %s", warmup.model_dump())

    print(BANNER_TEXT)
    await run_chat(orchestrator)
    log.info("Chat session ended.")


def main() -> None:
    asyncio.run(_amain())


if __name__ == "__main__":
    main()
