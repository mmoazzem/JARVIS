"""
DuckDuckGo search backend — the keyless default implementation of BaseSearch.

Uses the `ddgs` library: free, no API key (fits the no-secrets rule), but it
scrapes public engines, so rate limits and layout changes surface as
exceptions — the Tool wrapper degrades those to a structured error and Jarvis
answers honestly. When a keyed backend (Tavily/Brave) is wanted, it swaps in
behind BaseSearch via config; nothing else changes.
"""
from __future__ import annotations

import asyncio

from ddgs import DDGS

from core.constants import SEARCH_TIMEOUT_S
from core.tools.base_search import BaseSearch


class DuckDuckGoSearch(BaseSearch):
    async def search(self, query: str, max_results: int) -> list[dict]:
        # ddgs is synchronous — run it off the event loop.
        hits = await asyncio.to_thread(
            lambda: DDGS(timeout=SEARCH_TIMEOUT_S).text(query, max_results=max_results)
        )
        # Map the ddgs wire fields (title/href/body) onto the BaseSearch contract.
        return [
            {
                "title": hit.get("title", ""),
                "url": hit.get("href", ""),
                "snippet": hit.get("body", ""),
            }
            for hit in hits
        ]
