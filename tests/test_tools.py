"""Tool seam invariants (core/tools/) — mocked network, no Ollama.

Pinned behaviors:
  * The registry exposes tools in the tools-API wire format and dispatches by
    name; call() NEVER raises — unknown tools and crashing implementations come
    back as {"error": ...} data.
  * Tools return structured DATA, never prose.
  * The weather tool survives network failure and geocode misses with a
    structured error, never an exception (a broken tool must not break a turn).
"""
from datetime import datetime

import httpx

from core.constants import FETCH_TRUNCATION_MARKER
from core.tools.base import Tool
from core.tools.base_extractor import BaseExtractor
from core.tools.base_search import BaseSearch
from core.tools.duckduckgo_search import DuckDuckGoSearch
from core.tools.fetch_url_tool import FetchUrlTool
from core.tools.page_fetcher import PageFetcher
from core.tools.registry import ToolRegistry
from core.tools.time_tool import TimeTool
from core.tools.trafilatura_extractor import TrafilaturaExtractor
from core.tools.weather_tool import WeatherTool
from core.tools.web_search_tool import WebSearchTool
from core.tools.wikipedia_tool import WikipediaTool


class BoomTool(Tool):
    name = "boom"
    description = "always raises"
    parameters = {"type": "object", "properties": {}, "required": []}
    status = "exploding"

    async def run(self, **args) -> dict:
        raise RuntimeError("kaboom")


# --- registry -------------------------------------------------------------


def test_schemas_are_in_tools_api_wire_format():
    registry = ToolRegistry()
    registry.register(TimeTool())

    schemas = registry.schemas()

    assert schemas == [{
        "type": "function",
        "function": {
            "name": "get_time",
            "description": TimeTool.description,
            "parameters": TimeTool.parameters,
        },
    }]


async def test_unknown_tool_returns_error_data_not_raise():
    registry = ToolRegistry()

    result = await registry.call("made_up_tool", {})

    assert "error" in result and "made_up_tool" in result["error"]


async def test_crashing_tool_returns_error_data_not_raise():
    registry = ToolRegistry()
    registry.register(BoomTool())

    result = await registry.call("boom", {})

    assert "error" in result and "kaboom" in result["error"]


async def test_bad_model_args_return_error_data_not_raise():
    registry = ToolRegistry()
    registry.register(TimeTool())

    # get_time takes no args; a hallucinated arg must not blow up the turn.
    result = await registry.call("get_time", {"city": "Denver"})

    assert "error" in result


def test_status_for_unknown_tool_has_a_fallback():
    registry = ToolRegistry()
    assert "made_up_tool" in registry.status_for("made_up_tool")


# --- time tool --------------------------------------------------------------


async def test_time_tool_returns_the_current_local_time():
    result = await TimeTool().run()

    reported = datetime.strptime(result["datetime"], "%Y-%m-%d %H:%M")
    assert abs((datetime.now() - reported).total_seconds()) < 120
    assert result["weekday"] == reported.strftime("%A")


# --- weather tool ----------------------------------------------------------


class _FakeResponse:
    def __init__(self, payload=None, text=""):
        self._payload = payload
        self.text = text

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


GEOCODE_PAYLOAD = {"results": [{
    "name": "Buffalo", "admin1": "New York", "country": "United States",
    "latitude": 42.9, "longitude": -78.9,
}]}
FORECAST_PAYLOAD = {
    "current": {
        "temperature_2m": 71.4, "apparent_temperature": 73.0,
        "relative_humidity_2m": 60, "precipitation": 0.0,
        "weather_code": 2, "wind_speed_10m": 8.1,
    },
    "daily": {
        "time": ["2026-07-05", "2026-07-06", "2026-07-07"],
        "weather_code": [2, 61, 0],
        "temperature_2m_max": [78.0, 70.2, 75.5],
        "temperature_2m_min": [60.1, 58.3, 59.0],
        "precipitation_probability_max": [10, 80, 5],
    },
}


def _route_get(monkeypatch, handler):
    async def fake_get(self, url, params=None, **kw):
        return handler(url, params or {})
    monkeypatch.setattr(httpx.AsyncClient, "get", fake_get)


async def test_weather_returns_structured_data_with_units_in_field_names(monkeypatch):
    def handler(url, params):
        if "geocoding" in url:
            return _FakeResponse(GEOCODE_PAYLOAD)
        return _FakeResponse(FORECAST_PAYLOAD)
    _route_get(monkeypatch, handler)

    result = await WeatherTool("Buffalo, NY").run()

    assert result["location"] == "Buffalo, New York, United States"
    assert result["current"]["temperature_f"] == 71.4
    assert result["current"]["conditions"] == "partly cloudy"
    assert len(result["forecast"]) == 3
    assert result["forecast"][1]["precip_chance_pct"] == 80


async def test_weather_city_override_geocodes_that_city_not_default(monkeypatch):
    seen = []

    def handler(url, params):
        if "geocoding" in url:
            seen.append(params["name"])
            return _FakeResponse(GEOCODE_PAYLOAD)
        return _FakeResponse(FORECAST_PAYLOAD)
    _route_get(monkeypatch, handler)

    await WeatherTool("Buffalo, NY").run(city="Denver")

    assert seen == ["Denver"]


async def test_weather_city_state_string_falls_back_to_city_name(monkeypatch):
    # Open-Meteo's geocoder returns nothing for "Buffalo, NY" — the tool must
    # retry with "Buffalo" instead of failing the default location.
    seen = []

    def handler(url, params):
        if "geocoding" in url:
            seen.append(params["name"])
            payload = {"results": []} if "," in params["name"] else GEOCODE_PAYLOAD
            return _FakeResponse(payload)
        return _FakeResponse(FORECAST_PAYLOAD)
    _route_get(monkeypatch, handler)

    result = await WeatherTool("Buffalo, NY").run()

    assert seen == ["Buffalo, NY", "Buffalo"]
    assert result["location"] == "Buffalo, New York, United States"


async def test_weather_network_failure_returns_structured_error(monkeypatch):
    async def dead(self, url, **kw):
        raise httpx.ConnectError("no route to host")
    monkeypatch.setattr(httpx.AsyncClient, "get", dead)

    result = await WeatherTool("Buffalo, NY").run()

    assert "error" in result and "unreachable" in result["error"]


async def test_weather_geocode_miss_returns_structured_error(monkeypatch):
    _route_get(monkeypatch, lambda url, params: _FakeResponse({"results": []}))

    result = await WeatherTool("Buffalo, NY").run(city="Xyzzyville")

    assert "error" in result and "Xyzzyville" in result["error"]


# --- web search tool ---------------------------------------------------------


class FakeSearch(BaseSearch):
    """Scripted BaseSearch: returns fixed hits or raises, records calls."""

    def __init__(self, results=None, exc: Exception | None = None):
        self._results = results or []
        self._exc = exc
        self.seen: list[tuple[str, int]] = []

    async def search(self, query: str, max_results: int) -> list[dict]:
        self.seen.append((query, max_results))
        if self._exc is not None:
            raise self._exc
        return self._results


class FakeFetcher:
    """Scripted PageFetcher stand-in: per-URL results, records fetched URLs."""

    def __init__(self, pages: dict[str, dict] | None = None):
        self._pages = pages or {}
        self.seen: list[str] = []

    async def fetch(self, url: str) -> dict:
        self.seen.append(url)
        return self._pages.get(url, {"error": f"couldn't fetch {url}: dead"})


HITS = [
    {"title": "Result A", "url": "https://a.example", "snippet": "alpha"},
    {"title": "Result B", "url": "https://b.example", "snippet": "beta"},
    {"title": "Result C", "url": "https://c.example", "snippet": "gamma"},
]
PAGE_A = {"url": "https://a.example", "title": "Page A", "text": "full text A"}
PAGE_B = {"url": "https://b.example", "title": "Page B", "text": "full text B"}


async def test_web_search_fetch_count_zero_is_snippet_only():
    # The escape hatch: search_fetch_count 0 restores the old fast behavior.
    backend = FakeSearch(results=HITS)
    fetcher = FakeFetcher()

    result = await WebSearchTool(backend, fetcher, 0).run(query="nba finals 2026")

    assert result == {"query": "nba finals 2026", "results": HITS}
    assert backend.seen and backend.seen[0][0] == "nba finals 2026"
    assert fetcher.seen == []


async def test_web_search_fetches_top_pages_and_keeps_the_rest_as_snippets():
    fetcher = FakeFetcher({"https://a.example": PAGE_A, "https://b.example": PAGE_B})

    result = await WebSearchTool(FakeSearch(results=HITS), fetcher, 2).run(query="q")

    assert result == {
        "query": "q",
        "sources": [PAGE_A, PAGE_B],
        "other_results": [HITS[2]],
    }
    assert fetcher.seen == ["https://a.example", "https://b.example"]


async def test_web_search_skips_failed_fetches_and_answers_from_the_rest():
    # One dead page must not sink the search — partial success is fine.
    fetcher = FakeFetcher({"https://b.example": PAGE_B})

    result = await WebSearchTool(FakeSearch(results=HITS), fetcher, 2).run(query="q")

    assert result["sources"] == [PAGE_B]
    assert "error" not in result


async def test_web_search_all_fetches_failed_falls_back_to_snippets():
    fetcher = FakeFetcher()  # every fetch errors

    result = await WebSearchTool(FakeSearch(results=HITS), fetcher, 2).run(query="q")

    assert "error" not in result
    assert result["results"] == HITS
    assert "snippets" in result["note"]


async def test_web_search_backend_failure_returns_structured_error():
    # DDG scraping is fragile — a rate limit must degrade, never raise.
    backend = FakeSearch(exc=RuntimeError("202 Ratelimit"))

    result = await WebSearchTool(backend, FakeFetcher(), 3).run(query="anything")

    assert "error" in result and "Ratelimit" in result["error"]


async def test_web_search_no_results_returns_error_data():
    result = await WebSearchTool(FakeSearch(results=[]), FakeFetcher(), 3).run(query="xqzzk")

    assert "error" in result and "xqzzk" in result["error"]


async def test_web_search_empty_query_returns_error_data():
    result = await WebSearchTool(FakeSearch(results=HITS), FakeFetcher(), 3).run(query="  ")

    assert "error" in result


async def test_ddg_backend_maps_wire_fields_to_the_search_contract(monkeypatch):
    class FakeDDGS:
        def __init__(self, **kwargs):
            pass

        def text(self, query, max_results=None):
            assert (query, max_results) == ("q", 3)
            return [{"title": "T", "href": "https://u.example", "body": "S"}]

    monkeypatch.setattr("core.tools.duckduckgo_search.DDGS", FakeDDGS)

    results = await DuckDuckGoSearch().search("q", 3)

    assert results == [{"title": "T", "url": "https://u.example", "snippet": "S"}]


# --- wikipedia tool ----------------------------------------------------------


WIKI_SEARCH_PAYLOAD = {"pages": [{"id": 1208, "key": "Alan_Turing", "title": "Alan Turing"}]}
WIKI_SUMMARY_PAYLOAD = {
    "title": "Alan Turing",
    "type": "standard",
    "extract": "Alan Mathison Turing was an English mathematician and computer scientist.",
    "content_urls": {"desktop": {"page": "https://en.wikipedia.org/wiki/Alan_Turing"}},
}


def _wiki_handler(summary_payload, search_payload=WIKI_SEARCH_PAYLOAD):
    def handler(url, params):
        if "search" in url:
            return _FakeResponse(search_payload)
        return _FakeResponse(summary_payload)
    return handler


async def test_wikipedia_returns_title_summary_and_url(monkeypatch):
    _route_get(monkeypatch, _wiki_handler(WIKI_SUMMARY_PAYLOAD))

    result = await WikipediaTool().run(topic="alan turing")

    assert result == {
        "title": "Alan Turing",
        "summary": WIKI_SUMMARY_PAYLOAD["extract"],
        "url": "https://en.wikipedia.org/wiki/Alan_Turing",
    }


async def test_wikipedia_not_found_returns_error_data(monkeypatch):
    _route_get(monkeypatch, _wiki_handler(WIKI_SUMMARY_PAYLOAD, search_payload={"pages": []}))

    result = await WikipediaTool().run(topic="xqzzk gibberish")

    assert "error" in result and "xqzzk gibberish" in result["error"]


async def test_wikipedia_disambiguation_is_flagged_not_an_error(monkeypatch):
    ambiguous = {**WIKI_SUMMARY_PAYLOAD, "type": "disambiguation", "title": "Mercury"}
    _route_get(monkeypatch, _wiki_handler(ambiguous))

    result = await WikipediaTool().run(topic="Mercury")

    assert "error" not in result
    assert "ambiguous" in result["note"]


async def test_wikipedia_network_failure_returns_structured_error(monkeypatch):
    async def dead(self, url, **kw):
        raise httpx.ConnectError("no route to host")
    monkeypatch.setattr(httpx.AsyncClient, "get", dead)

    result = await WikipediaTool().run(topic="Alan Turing")

    assert "error" in result and "unreachable" in result["error"]


# --- extractor ----------------------------------------------------------------


ARTICLE_HTML = """<html><head><title>Site — Page</title></head><body>
<nav>Home | About | Contact</nav>
<article><h1>Real Headline</h1>
<p>First paragraph of the actual article body with enough words to matter.</p>
<p>Second paragraph continues the story in considerable further detail.</p></article>
<footer>copyright 2026</footer></body></html>"""


def test_trafilatura_extractor_returns_main_text_and_title():
    result = TrafilaturaExtractor().extract(ARTICLE_HTML)

    assert "First paragraph of the actual article body" in result["text"]
    assert "Home | About" not in result["text"]  # boilerplate stripped
    assert result["title"] == "Real Headline"


def test_trafilatura_extractor_returns_none_when_nothing_readable():
    # The JS-rendered-page shape: markup with no readable content.
    html = "<html><body><script>renderApp()</script></body></html>"

    assert TrafilaturaExtractor().extract(html) is None


# --- page fetcher --------------------------------------------------------------


class FakeExtractor(BaseExtractor):
    def __init__(self, result):
        self._result = result

    def extract(self, html: str) -> dict | None:
        return self._result


async def test_fetcher_returns_url_title_and_text(monkeypatch):
    _route_get(monkeypatch, lambda url, params: _FakeResponse(text="<html>…</html>"))
    extractor = FakeExtractor({"text": "clean body", "title": "T", "date": None})

    result = await PageFetcher(extractor, max_chars=100, timeout=5.0).fetch("https://x.example")

    assert result == {"url": "https://x.example", "title": "T", "text": "clean body"}


async def test_fetcher_caps_page_text_with_a_truncation_marker(monkeypatch):
    _route_get(monkeypatch, lambda url, params: _FakeResponse(text="<html>…</html>"))
    extractor = FakeExtractor({"text": "x" * 500, "title": "T", "date": None})

    result = await PageFetcher(extractor, max_chars=100, timeout=5.0).fetch("https://x.example")

    assert result["text"] == "x" * 100 + FETCH_TRUNCATION_MARKER


async def test_fetcher_network_failure_returns_structured_error(monkeypatch):
    async def dead(self, url, **kw):
        raise httpx.ConnectError("no route to host")
    monkeypatch.setattr(httpx.AsyncClient, "get", dead)
    extractor = FakeExtractor({"text": "never reached"})

    result = await PageFetcher(extractor, max_chars=100, timeout=5.0).fetch("https://x.example")

    assert "error" in result and "x.example" in result["error"]


async def test_fetcher_unreadable_page_returns_structured_error(monkeypatch):
    # JS-rendered pages extract empty — the static-HTML ceiling, reported
    # cleanly, never a crash.
    _route_get(monkeypatch, lambda url, params: _FakeResponse(text="<html></html>"))

    result = await PageFetcher(FakeExtractor(None), max_chars=100, timeout=5.0).fetch(
        "https://x.example"
    )

    assert "error" in result and "no extractable text" in result["error"]


# --- fetch_url tool -------------------------------------------------------------


async def test_fetch_url_returns_the_fetched_page():
    fetcher = FakeFetcher({"https://a.example": PAGE_A})

    result = await FetchUrlTool(fetcher).run(url="https://a.example")

    assert result == PAGE_A


async def test_fetch_url_empty_url_returns_error_data():
    result = await FetchUrlTool(FakeFetcher()).run(url="  ")

    assert "error" in result


async def test_fetch_url_fetch_failure_returns_error_data():
    result = await FetchUrlTool(FakeFetcher()).run(url="https://dead.example")

    assert "error" in result and "dead.example" in result["error"]
