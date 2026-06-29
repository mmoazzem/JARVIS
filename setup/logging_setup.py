"""
Logging configuration: daily append-mode file, quiet console.

File handler captures INFO+ (including httpx). Console is WARNING+ only so the
chat surface stays clean. A session-boundary marker is written at every startup
so consecutive sessions are visually distinct inside the same log file.
"""
import logging
from datetime import datetime

from core.constants import (
    LOG_CONSOLE_FORMAT,
    LOG_DATE_FORMAT,
    LOG_FILE_FORMAT,
    LOG_FILE_NAME_FORMAT,
    LOG_SESSION_BOUNDARY,
    LOG_TIME_FORMAT,
    LOGGER_ROOT,
    LOGS_DIR,
)


def setup_logging() -> None:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOGS_DIR / datetime.now().strftime(LOG_FILE_NAME_FORMAT)

    root = logging.getLogger()
    if root.handlers:
        # Already configured (e.g., during testing) — don't add duplicate handlers.
        return
    root.setLevel(logging.DEBUG)

    file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(
        logging.Formatter(LOG_FILE_FORMAT, datefmt=LOG_TIME_FORMAT)
    )

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.WARNING)
    console_handler.setFormatter(logging.Formatter(LOG_CONSOLE_FORMAT))

    root.addHandler(file_handler)
    root.addHandler(console_handler)

    # Ensure httpx request logs reach the file (they log at INFO by default).
    logging.getLogger("httpx").setLevel(logging.INFO)

    logging.getLogger(LOGGER_ROOT).info(LOG_SESSION_BOUNDARY)
