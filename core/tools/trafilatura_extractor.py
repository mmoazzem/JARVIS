"""
Trafilatura extractor — the default BaseExtractor implementation.

Trafilatura does best-in-class main-content extraction as a single in-repo
library, keeping Jarvis self-contained (no external reader service). Its
static-HTML limit is accepted: pages that render entirely via JavaScript
extract as None, which the fetcher reports as a clean "couldn't read that
page" — browser-driven rendering is a future tier, not this seam.
"""
from __future__ import annotations

import logging

import trafilatura

from core.tools.base_extractor import BaseExtractor

# trafilatura logs per-page extraction misses ("discarding data") at WARNING,
# which would leak onto the console chat surface (console shows WARNING+). The
# fetcher already reports every miss with the URL, so the third-party logger
# adds only noise.
logging.getLogger("trafilatura").setLevel(logging.ERROR)


class TrafilaturaExtractor(BaseExtractor):
    def extract(self, html: str) -> dict | None:
        text = trafilatura.extract(html, include_comments=False)
        if not text:
            return None
        metadata = trafilatura.extract_metadata(html)
        return {
            "text": text,
            "title": getattr(metadata, "title", None),
            "date": getattr(metadata, "date", None),
        }
