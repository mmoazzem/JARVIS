"""
Web search tool — the Tool the model calls; the backend behind it is swappable.

The BaseSearch backend is chosen by config (`search_backend`); DuckDuckGo is
the keyless default. Search-only this pass: titles/urls/snippets, no page
fetching. Any backend failure (rate limit, scrape breakage, network) returns
`{"error": ...}` — never raises — exactly like the weather-unreachable path.
"""
from __future__ import annotations

import logging

from core.constants import LOGGER_TOOLS, SEARCH_MAX_RESULTS
from core.tools.base import Tool
from core.tools.base_search import BaseSearch

logger = logging.getLogger(LOGGER_TOOLS)


class WebSearchTool(Tool):
    name = "web_search"
    description = (
        "Search the web. Use for current events, news, sports results, prices, "
        "or anything recent or time-sensitive that you may not know."
    )
    parameters = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "The search query."},
        },
        "required": ["query"],
    }
    status = "searching the web"

    def __init__(self, backend: BaseSearch) -> None:
        self._backend = backend

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
        return {"query": query, "results": results}
