"""
Web search tool — the Tool the model calls; the backend behind it is swappable.

The BaseSearch backend is chosen by config (`search_backend`); DuckDuckGo is
the keyless default. Fetch-and-read (Tools Pass 3): the top `fetch_count`
result pages are fetched and reduced to real text, because snippets alone made
the model mis-synthesize precise data (schedules, venues, numbers). Multiple
fetched sources let it cross-reference. A failed fetch skips that source; if
every fetch fails the tool falls back to snippets — search never hard-fails.
Any backend failure (rate limit, scrape breakage, network) returns
`{"error": ...}` — never raises — exactly like the weather-unreachable path.
"""
from __future__ import annotations

import asyncio
import logging

from core.constants import LOGGER_TOOLS, SEARCH_MAX_RESULTS
from core.tools.base import Tool
from core.tools.base_search import BaseSearch
from core.tools.page_fetcher import PageFetcher

logger = logging.getLogger(LOGGER_TOOLS)


class WebSearchTool(Tool):
    name = "web_search"
    description = (
        "Search the web and read the top result pages. Use for current events, "
        "news, sports results, prices, schedules, or anything recent or "
        "time-sensitive that you may not know."
    )
    parameters = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "The search query."},
        },
        "required": ["query"],
    }
    status = "searching the web"

    def __init__(self, backend: BaseSearch, fetcher: PageFetcher, fetch_count: int) -> None:
        self._backend = backend
        self._fetcher = fetcher
        self._fetch_count = fetch_count

    async def run(self, query: str = "") -> dict:
        query = query.strip()
        if not query:
            return {"error": "no search query given"}
        try:
            results = await self._backend.search(query, SEARCH_MAX_RESULTS)
        except Exception as exc:
            logger.warning("web search failed for %r: %s", query, exc)
            return {"error": f"search failed: {exc}"}
        if not results:
            return {"error": f"no results found for {query!r}"}

        # fetch_count 0 is the escape hatch: snippet-only, the old fast path.
        if self._fetch_count > 0:
            top = [hit["url"] for hit in results[: self._fetch_count] if hit.get("url")]
            pages = await asyncio.gather(*(self._fetcher.fetch(url) for url in top))
            sources = [page for page in pages if "error" not in page]
            if sources:
                # Full page text supersedes those hits' snippets; keep the
                # remaining hits as snippets so nothing found is lost.
                return {
                    "query": query,
                    "sources": sources,
                    "other_results": results[self._fetch_count :],
                }
            if top:
                logger.warning("all %d page fetches failed for %r", len(top), query)
                return {
                    "query": query,
                    "results": results,
                    "note": (
                        "none of the top result pages could be read; these are "
                        "search snippets only — treat details as unverified"
                    ),
                }
        return {"query": query, "results": results}
