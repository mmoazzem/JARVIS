"""
Orchestrator — thin wiring that assembles the first Agent and exposes it.

It constructs the model from config, loads the persona LIVE from identity.yaml, and
hands both to the Agent. No conversational logic lives here; that is the Agent's.
"""
from __future__ import annotations

from typing import AsyncIterator

from setup.config import JarvisConfig, load_identity
from models.ollama_model import OllamaModel
from models.base import WarmupResult
from core.orchestrator.agent import Agent
from core.constants import SEARCH_BACKEND_DUCKDUCKGO
from core.tools.duckduckgo_search import DuckDuckGoSearch
from core.tools.registry import ToolRegistry
from core.tools.time_tool import TimeTool
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
            tools = ToolRegistry()
            tools.register(TimeTool())
            tools.register(WeatherTool(config.default_location))
            tools.register(WebSearchTool(search_cls()))
            tools.register(WikipediaTool())

        # Persona loaded live at boot — never copied into config (CLAUDE.md).
        self._agent = Agent(self._model, config, load_identity(), tools=tools)

    def respond(self, user_text: str) -> AsyncIterator[dict]:
        """Yield the Agent's structured turn events for one user message."""
        return self._agent.respond(user_text)

    async def warmup(self) -> WarmupResult:
        return await self._model.warmup()

    async def health_check(self) -> bool:
        return await self._model.health_check()
