"""
Fetch-URL tool — read one specific page the user (or model) points at.

The standalone half of fetch-and-read: web_search auto-fetches its top hits,
this handles "summarize this page: <url>" and pasted links. All failure modes
arrive from the fetcher as structured errors, so this wrapper never raises.
"""
from __future__ import annotations

from core.tools.base import Tool
from core.tools.page_fetcher import PageFetcher


class FetchUrlTool(Tool):
    name = "fetch_url"
    description = (
        "Fetch and read a specific web page. Use when the user gives a URL or "
        "asks about a particular page or article. Returns the page's main text."
    )
    parameters = {
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "description": "The full URL to read, e.g. 'https://example.com/article'.",
            },
        },
        "required": ["url"],
    }
    status = "reading the page"

    def __init__(self, fetcher: PageFetcher) -> None:
        self._fetcher = fetcher

    async def run(self, url: str = "") -> dict:
        url = url.strip()
        if not url:
            return {"error": "no URL given"}
        return await self._fetcher.fetch(url)
