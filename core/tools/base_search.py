"""
BaseSearch — the swappable web-search seam (same pattern as BaseTTS for voices).

A backend turns a query into structured hits; the Tool wrapper
(web_search_tool.py) owns the never-raise contract, so implementations are free
to raise on failure and the wrapper degrades it to `{"error": ...}` data.
DuckDuckGo is the keyless default; keyed backends (Tavily, Brave, Google) are
future implementations behind this same interface — their keys come from the
credential seam (core/credentials.py) when they land.
"""
from __future__ import annotations

from abc import ABC, abstractmethod


class BaseSearch(ABC):
    @abstractmethod
    async def search(self, query: str, max_results: int) -> list[dict]:
        """Return up to max_results hits: {"title", "url", "snippet"} each."""
