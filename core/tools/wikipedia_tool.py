"""
Wikipedia tool — keyless factual lookups via the Wikipedia REST API.

Two steps: search resolves a fuzzy topic to the best article key, then the
summary endpoint returns the lead extract as structured data. Disambiguation
pages come back with a note so the model can ask which sense was meant;
not-found and network failure return `{"error": ...}` — never raises.
"""
from __future__ import annotations

import logging

import httpx

from core.constants import (
    HTTP_USER_AGENT,
    LOGGER_TOOLS,
    WIKIPEDIA_SEARCH_URL,
    WIKIPEDIA_SUMMARY_URL,
    WIKIPEDIA_TIMEOUT_S,
)
from core.tools.base import Tool

logger = logging.getLogger(LOGGER_TOOLS)


class WikipediaTool(Tool):
    name = "wikipedia"
    description = (
        "Look up a topic on Wikipedia for factual background: people, places, "
        "concepts, history, science. Returns the article summary. Not for "
        "current events or news — use web_search for those."
    )
    parameters = {
        "type": "object",
        "properties": {
            "topic": {
                "type": "string",
                "description": "The topic to look up, e.g. 'Alan Turing'.",
            },
        },
        "required": ["topic"],
    }
    status = "checking Wikipedia"

    async def run(self, topic: str = "") -> dict:
        topic = topic.strip()
        if not topic:
            return {"error": "no topic given"}
        try:
            async with httpx.AsyncClient(
                timeout=WIKIPEDIA_TIMEOUT_S,
                headers={"User-Agent": HTTP_USER_AGENT},
                follow_redirects=True,
            ) as client:
                resp = await client.get(
                    WIKIPEDIA_SEARCH_URL, params={"q": topic, "limit": 1}
                )
                resp.raise_for_status()
                pages = resp.json().get("pages") or []
                if not pages:
                    return {"error": f"no Wikipedia article found for {topic!r}"}
                resp = await client.get(
                    WIKIPEDIA_SUMMARY_URL.format(key=pages[0]["key"])
                )
                resp.raise_for_status()
                summary = resp.json()
        except Exception as exc:
            logger.warning("wikipedia lookup failed for %r: %s", topic, exc)
            return {"error": f"Wikipedia unreachable: {exc}"}

        result = {
            "title": summary.get("title") or pages[0].get("title", topic),
            "summary": summary.get("extract", ""),
            "url": summary.get("content_urls", {}).get("desktop", {}).get("page", ""),
        }
        if summary.get("type") == "disambiguation":
            # Ambiguous topic — flag it as data so the model can ask which
            # sense was meant or retry with a more specific topic.
            result["note"] = (
                f"{topic!r} is ambiguous (a disambiguation page); "
                "retry with a more specific topic or ask the user"
            )
        return result
