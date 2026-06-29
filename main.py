"""Jarvis entry point — Milestone 1 boot sequence (config + logging only)."""
import logging

from core.constants import CONFIG_PATH, LOGGER_ROOT
from setup.logging_setup import setup_logging


def main() -> None:
    setup_logging()
    log = logging.getLogger(LOGGER_ROOT)

    if not CONFIG_PATH.exists():
        log.info("config.yaml not found — setup wizard required before first run")
    else:
        log.info("Config found: %s", CONFIG_PATH)

    log.info("Boot complete.")


if __name__ == "__main__":
    main()
