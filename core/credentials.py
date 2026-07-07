"""
Credential seam — API keys for keyed backends live in a gitignored .env.

load_credentials() is called once at boot; get_key() reads the process
environment, so keys exported in the shell work identically. A missing .env or
unset key returns None — keyless backends (DuckDuckGo, Wikipedia, Open-Meteo)
run with nothing configured, and a future keyed backend (Tavily, cloud models)
reports "no key configured" as a structured error instead of crashing.

Variable names are documented in the committed .env.example; real values are
never committed anywhere.
"""
from __future__ import annotations

import os

from dotenv import load_dotenv

from core.constants import ENV_PATH


def load_credentials() -> None:
    """Load .env into the process environment. No-op when the file is absent;
    values already exported in the shell win over .env lines."""
    load_dotenv(ENV_PATH)


def get_key(name: str) -> str | None:
    """Return the named key, or None when unset. An empty value (a template
    line like `TAVILY_API_KEY=`) counts as unset."""
    return os.getenv(name) or None
