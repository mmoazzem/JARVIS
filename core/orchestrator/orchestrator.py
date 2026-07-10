"""
Orchestrator — thin wiring that assembles the first Agent and exposes it.

It constructs the model from config, loads the persona LIVE from identity.yaml, and
hands both to the Agent. No conversational logic lives here; that is the Agent's.
"""
from __future__ import annotations

from datetime import datetime
from typing import AsyncIterator, Optional, Tuple

from setup.config import JarvisConfig, load_identity
from models.ollama_model import OllamaModel
from models.base import WarmupResult
from core.orchestrator.agent import Agent
from core.constants import (
    EVENT_LOG_FILE_FORMAT,
    EVENTS_LOG_DIR,
    LOG_DATE_FORMAT,
    SEARCH_BACKEND_DUCKDUCKGO,
)
from core.memory.base_digest import DayDigest
from core.memory.digest import digest, digest_all
from core.memory.llm_digest import LLMDigest
from core.memory.merge import (
    Profile,
    load_day_digests,
    load_profile,
    merge,
    save_profile,
    working_view,
)
from core.orchestrator.personality import render_profile
from core.tools.duckduckgo_search import DuckDuckGoSearch
from core.tools.fetch_url_tool import FetchUrlTool
from core.tools.page_fetcher import PageFetcher
from core.tools.registry import ToolRegistry
from core.tools.time_tool import TimeTool
from core.tools.trafilatura_extractor import TrafilaturaExtractor
from core.tools.weather_tool import WeatherTool
from core.tools.web_search_tool import WebSearchTool
from core.tools.wikipedia_tool import WikipediaTool

# Search backends behind BaseSearch — keyed ones (Tavily/Brave) join this map
# when their implementations land; picking one is config, not code.
_SEARCH_BACKENDS = {SEARCH_BACKEND_DUCKDUCKGO: DuckDuckGoSearch}


class Orchestrator:
    def __init__(self, config: JarvisConfig) -> None:
        self._config = config
        self._model = OllamaModel(
            config.primary_model,
            config.ollama_base_url,
            max_tokens=config.max_tokens,
            temperature=config.temperature,
            keep_alive=config.ollama_keep_alive,
            timeout=config.ollama_request_timeout,
        )
        # Tools Pass 1: assembling the registry is the ONLY place tools are
        # listed — adding one later is one register() line here.
        tools = None
        if config.tools_enabled:
            search_cls = _SEARCH_BACKENDS.get(config.search_backend)
            if search_cls is None:
                raise ValueError(
                    f"unknown search_backend {config.search_backend!r} "
                    f"— valid: {sorted(_SEARCH_BACKENDS)}"
                )
            # One fetcher shared by search (auto-fetch top hits) and fetch_url.
            fetcher = PageFetcher(
                TrafilaturaExtractor(),
                max_chars=config.fetch_max_chars,
                timeout=config.fetch_timeout,
            )
            tools = ToolRegistry()
            tools.register(TimeTool())
            tools.register(WeatherTool(config.default_location))
            tools.register(WebSearchTool(search_cls(), fetcher, config.search_fetch_count))
            tools.register(WikipediaTool())
            tools.register(FetchUrlTool(fetcher))

        # Persona loaded live at boot — never copied into config (CLAUDE.md).
        self._agent = Agent(self._model, config, load_identity(), tools=tools)
        # Layer-3 memory enters the prompt at boot; /merge refreshes it live.
        stored = load_profile()
        if stored is not None:
            self._agent.profile = render_profile(working_view(stored))

    def respond(self, user_text: str) -> AsyncIterator[dict]:
        """Yield the Agent's structured turn events for one user message."""
        return self._agent.respond(user_text)

    def _digest_extractor(self) -> LLMDigest:
        """Extraction reuses the resident primary model unless config names a
        different digest_model — two 14B models cannot co-reside in 16 GB
        (CLAUDE.md), so a distinct choice rides Ollama's unload/load swap."""
        model = self._model
        if self._config.digest_model and self._config.digest_model != self._config.primary_model:
            model = OllamaModel(
                self._config.digest_model,
                self._config.ollama_base_url,
                max_tokens=self._config.max_tokens,
                temperature=self._config.temperature,
                keep_alive=self._config.ollama_keep_alive,
                timeout=self._config.ollama_request_timeout,
            )
        return LLMDigest(
            model,
            timeout=self._config.digest_timeout,
            passes=self._config.digest_passes,
        )

    async def digest_day(self, day: Optional[str] = None, *, force: bool = False) -> DayDigest:
        """On-demand memory-Layer-2 digest of one day's event log (default: today).

        force bypasses the cache — needed after an extractor change, and the
        only way to refresh TODAY's digest while its event log is still growing
        (past days are immutable once closed, so their caches are final).
        """
        date = datetime.strptime(day, LOG_DATE_FORMAT) if day else datetime.now()
        day_file = EVENTS_LOG_DIR / date.strftime(EVENT_LOG_FILE_FORMAT)
        return await digest(day_file, self._digest_extractor(), force=force)

    def digest_all(
        self, *, force: bool = False
    ) -> AsyncIterator[Tuple[str, Optional[DayDigest]]]:
        """Digest every event day-file, skipping days whose digest exists."""
        return digest_all(self._digest_extractor(), force=force)

    def merge_profile(self) -> Profile:
        """Reduce all cached day digests into the profile (memory Layer 2 merge).

        Persists profile.json and refreshes the agent's Layer-3 prompt view in
        place, so new memory is live from the very next turn.
        """
        profile = merge(load_day_digests())
        save_profile(profile)
        self._agent.profile = render_profile(working_view(profile))
        return profile

    async def warmup(self) -> WarmupResult:
        return await self._model.warmup()

    async def health_check(self) -> bool:
        return await self._model.health_check()
