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
        # Persona loaded live at boot — never copied into config (CLAUDE.md).
        self._agent = Agent(self._model, config, load_identity())

    def respond(self, user_text: str) -> AsyncIterator[dict]:
        """Yield the Agent's structured turn events for one user message."""
        return self._agent.respond(user_text)

    async def warmup(self) -> WarmupResult:
        return await self._model.warmup()

    async def health_check(self) -> bool:
        return await self._model.health_check()
