"""
BaseExtractor — the swappable page-text-extraction seam (same pattern as
BaseSearch/BaseTTS).

An extractor turns raw HTML into the page's main content, stripped of nav,
ads, and boilerplate. Extraction is synchronous and CPU-bound; the fetcher
(page_fetcher.py) runs it off the event loop and owns the never-raise
contract, so implementations are free to raise or return None on failure.
"""
from __future__ import annotations

from abc import ABC, abstractmethod


class BaseExtractor(ABC):
    @abstractmethod
    def extract(self, html: str) -> dict | None:
        """Main content of a page: {"text", "title", "date"} (title/date may be
        None). Returns None when no readable text is found — e.g. a page that
        renders entirely via JavaScript."""
