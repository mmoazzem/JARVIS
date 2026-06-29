"""
Provider-agnostic model interface.

This is the seam that makes local and cloud models interchangeable: the rest of
the app depends only on `BaseModel` and the structured types below, never on
Ollama/OpenAI specifics. Keep this file free of any provider details.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import AsyncIterator, Optional

from pydantic import BaseModel as PydanticBaseModel


class ModelResponse(PydanticBaseModel):
    """A completed (non-streamed) model turn.

    `content` is the answer the user sees. `thinking` is the accumulated
    chain-of-thought, kept SEPARATE from content (Ollama returns it in its own
    field — see CLAUDE.md). An empty `content` with non-empty `thinking` is a
    real, detectable condition: reasoning consumed the whole token budget. Callers
    must be able to see that, so we never fold thinking into content.
    """

    content: str
    model: str
    thinking: str = ""
    usage: Optional[dict] = None


class WarmupResult(PydanticBaseModel):
    """Structured outcome of forcing a model resident in VRAM.

    Returned (not printed) so any interface can decide how to surface it.
    """

    success: bool
    model_id: str
    elapsed_s: float
    error: Optional[str] = None


class BaseModel(ABC):
    """Async interface every model backend implements.

    Implementations own a single `model_id` (a constructor parameter — never a
    hardcoded name) and translate these calls to their provider's wire protocol.
    """

    @abstractmethod
    async def complete(self, messages: list[dict], **opts) -> ModelResponse:
        """Run one full turn and return content + separated thinking."""
        raise NotImplementedError

    @abstractmethod
    def stream(self, messages: list[dict], **opts) -> AsyncIterator[str]:
        """Yield CONTENT tokens only, as they arrive.

        Reasoning is accumulated internally and never yielded to the caller.
        Declared without `async` so the return type is the async iterator itself;
        implementations are `async def` generators.
        """
        raise NotImplementedError

    @abstractmethod
    async def health_check(self) -> bool:
        """Cheap reachability check against the backend. Never raises."""
        raise NotImplementedError

    @abstractmethod
    async def warmup(self) -> WarmupResult:
        """Force the model resident in VRAM; return a structured result."""
        raise NotImplementedError

    @abstractmethod
    async def unload(self) -> None:
        """Evict the model from VRAM (for the future specialist swap)."""
        raise NotImplementedError
