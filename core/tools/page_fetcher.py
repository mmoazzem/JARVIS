"""
Page fetcher — download a URL and reduce it to clean main text.

The shared fetch step behind web_search (auto-fetching top results) and
fetch_url (explicit URL). Owns the never-raise contract for the whole
fetch-extract pipeline: network errors, HTTP errors, and unreadable pages
(JS-rendered, non-HTML) all come back as `{"error": ...}` data. Extracted
text is capped at max_chars per page so a handful of fetched sources cannot
blow the context budget; truncation is marked so the model knows text is
missing rather than the page ending there.
"""
from __future__ import annotations

import asyncio
import logging

import httpx

from core.constants import FETCH_TRUNCATION_MARKER, HTTP_USER_AGENT, LOGGER_TOOLS
from core.tools.base_extractor import BaseExtractor

logger = logging.getLogger(LOGGER_TOOLS)


class PageFetcher:
    def __init__(self, extractor: BaseExtractor, max_chars: int, timeout: float) -> None:
        self._extractor = extractor
        self._max_chars = max_chars
        self._timeout = timeout

    async def fetch(self, url: str) -> dict:
        """Fetch one page: {"url", "title", "text"} on success, {"error": ...} on
        any failure. Never raises."""
        try:
            async with httpx.AsyncClient(
                timeout=self._timeout,
                headers={"User-Agent": HTTP_USER_AGENT},
                follow_redirects=True,
            ) as client:
                resp = await client.get(url)
                resp.raise_for_status()
                html = resp.text
        except Exception as exc:
            logger.warning("fetch failed for %s: %s", url, exc)
            return {"error": f"couldn't fetch {url}: {exc}"}

        # Extraction is synchronous CPU work — keep it off the event loop.
        extracted = await asyncio.to_thread(self._extractor.extract, html)
        if not extracted or not extracted.get("text"):
            # Static-HTML ceiling (JS-rendered or non-article page) — a clean
            # miss the model can say out loud, not a crash. Routine skip-on-
            # failure case, so INFO (log file only), not console WARNING.
            logger.info("no extractable text at %s", url)
            return {"error": f"couldn't read {url} (no extractable text)"}

        text = extracted["text"]
        if len(text) > self._max_chars:
            text = text[: self._max_chars] + FETCH_TRUNCATION_MARKER
        logger.info("fetched %s (%d chars)", url, len(text))
        return {"url": url, "title": extracted.get("title") or "", "text": text}
