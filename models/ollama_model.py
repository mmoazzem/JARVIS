"""
Ollama-backed model, spoken to over its OpenAI-compatible API.

One class serves every Ollama model — `model_id` is a constructor parameter, so
nothing here is specific to qwen3. All runtime tunables (token budget, temperature,
keep_alive, timeout) are injected from config; this file holds none of them.

The two historic foot-guns live here and are defended in one place:
  * Reasoning arrives in a SEPARATE wire field (`reasoning` / `reasoning_content`),
    NOT inside `<think>` tags. We split content from reasoning at the source and
    never yield reasoning to callers — no `<think>` parsing exists anywhere.
  * Reasoning shares the `max_tokens` budget. On hard prompts it can eat the whole
    budget and leave content empty. We don't hide that: a content-empty turn that
    produced reasoning is logged loudly and returned as a detectable empty-content
    `ModelResponse` (recovery is the orchestrator's job, in M3).
"""
from __future__ import annotations

import logging
import time
from typing import AsyncIterator, Optional

import httpx
from openai import AsyncOpenAI

from core.constants import (
    CONTENT_FIELD,
    LOGGER_MODEL,
    OLLAMA_API_KEY_PLACEHOLDER,
    OLLAMA_OPENAI_SUFFIX,
    REASONING_FIELDS,
    ROLE_USER,
)
from models.base import BaseModel, ModelResponse, WarmupResult

logger = logging.getLogger(LOGGER_MODEL)

# Smallest possible generation: just enough to force a cold model into VRAM.
_WARMUP_PROMPT = "ok"
_WARMUP_MAX_TOKENS = 1

# keep_alive value that tells Ollama to evict the model immediately after the call.
_UNLOAD_KEEP_ALIVE = 0


def _extract_reasoning(delta_or_message) -> str:
    """Pull reasoning out of a streamed delta or a full message, field-name-agnostic.

    The OpenAI client doesn't model Ollama's reasoning field, so it lands in the
    object's extra fields. We check both real attributes and `model_extra` for any
    of the known reasoning field names — this is the ONLY place reasoning is read.
    """
    extra = getattr(delta_or_message, "model_extra", None) or {}
    for field in REASONING_FIELDS:
        value = getattr(delta_or_message, field, None)
        if value is None:
            value = extra.get(field)
        if value:
            return value
    return ""


class OllamaModel(BaseModel):
    def __init__(
        self,
        model_id: str,
        base_url: str,
        *,
        max_tokens: int,
        temperature: float,
        keep_alive: str,
        timeout: float,
    ) -> None:
        self.model_id = model_id
        self._base_url = base_url.rstrip("/")
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._keep_alive = keep_alive
        self._timeout = timeout

        # Ollama's OpenAI-compatible surface lives under /v1; the key is required by
        # the client but ignored by the server.
        self._client = AsyncOpenAI(
            base_url=f"{self._base_url}{OLLAMA_OPENAI_SUFFIX}",
            api_key=OLLAMA_API_KEY_PLACEHOLDER,
            timeout=timeout,
        )

    def _request_kwargs(self, messages: list[dict], **opts) -> dict:
        """Assemble call kwargs, letting per-call opts override config defaults.

        keep_alive isn't part of the OpenAI schema, so it rides in extra_body —
        that's how Ollama learns how long to keep the model resident.
        """
        keep_alive = opts.pop("keep_alive", self._keep_alive)
        return {
            "model": self.model_id,
            "messages": messages,
            "max_tokens": opts.pop("max_tokens", self._max_tokens),
            "temperature": opts.pop("temperature", self._temperature),
            "extra_body": {"keep_alive": keep_alive},
            **opts,
        }

    async def stream(self, messages: list[dict], **opts) -> AsyncIterator[str]:
        """Yield only content tokens; accumulate reasoning separately for logging."""
        kwargs = self._request_kwargs(messages, **opts)
        stream = await self._client.chat.completions.create(stream=True, **kwargs)

        content_chars = 0
        reasoning_chars = 0
        async for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta

            reasoning = _extract_reasoning(delta)
            if reasoning:
                reasoning_chars += len(reasoning)
                # Reasoning is for the log only — never surfaced to the caller.
                logger.debug("reasoning +%d chars", len(reasoning))

            token = getattr(delta, CONTENT_FIELD, None)
            if token:
                content_chars += len(token)
                yield token

        logger.info(
            "stream done [%s]: content=%d chars, reasoning=%d chars",
            self.model_id,
            content_chars,
            reasoning_chars,
        )
        # The budget gotcha, made loud: reasoning ran but no answer came out.
        if content_chars == 0 and reasoning_chars > 0:
            logger.warning(
                "ZERO-CONTENT turn [%s]: reasoning consumed the budget "
                "(%d reasoning chars, 0 content). Raise max_tokens or recover upstream.",
                self.model_id,
                reasoning_chars,
            )

    async def complete(self, messages: list[dict], **opts) -> ModelResponse:
        """Run a full turn, returning content and reasoning as separate fields."""
        kwargs = self._request_kwargs(messages, **opts)
        resp = await self._client.chat.completions.create(stream=False, **kwargs)

        message = resp.choices[0].message
        content = message.content or ""
        thinking = _extract_reasoning(message)
        usage = resp.usage.model_dump() if resp.usage else None

        logger.info(
            "complete done [%s]: content=%d chars, reasoning=%d chars",
            self.model_id,
            len(content),
            len(thinking),
        )
        if not content and thinking:
            logger.warning(
                "ZERO-CONTENT turn [%s]: reasoning consumed the budget "
                "(%d reasoning chars, 0 content). Returning detectable empty content.",
                self.model_id,
                len(thinking),
            )

        return ModelResponse(
            content=content,
            model=self.model_id,
            thinking=thinking,
            usage=usage,
        )

    async def warmup(self) -> WarmupResult:
        """Force the model resident in VRAM with a 1-token request.

        Warmup IS the cold load, so the configured request timeout applies. Returns
        a structured result rather than printing — the interface decides display.
        """
        start = time.monotonic()
        try:
            await self._client.chat.completions.create(
                model=self.model_id,
                messages=[{"role": ROLE_USER, "content": _WARMUP_PROMPT}],
                max_tokens=_WARMUP_MAX_TOKENS,
                extra_body={"keep_alive": self._keep_alive},
            )
            elapsed = time.monotonic() - start
            logger.info("warmup ok [%s] in %.2fs", self.model_id, elapsed)
            return WarmupResult(
                success=True, model_id=self.model_id, elapsed_s=elapsed
            )
        except Exception as exc:  # surfaced structurally, never swallowed silently
            elapsed = time.monotonic() - start
            logger.error("warmup failed [%s]: %s", self.model_id, exc)
            return WarmupResult(
                success=False,
                model_id=self.model_id,
                elapsed_s=elapsed,
                error=str(exc),
            )

    async def unload(self) -> None:
        """Evict the model from VRAM via keep_alive=0 (for the future swap)."""
        try:
            await self._client.chat.completions.create(
                model=self.model_id,
                messages=[{"role": ROLE_USER, "content": _WARMUP_PROMPT}],
                max_tokens=_WARMUP_MAX_TOKENS,
                extra_body={"keep_alive": _UNLOAD_KEEP_ALIVE},
            )
            logger.info("unload requested [%s]", self.model_id)
        except Exception as exc:
            logger.error("unload failed [%s]: %s", self.model_id, exc)

    async def health_check(self) -> bool:
        """Cheap reachability probe against the Ollama server root. Never raises."""
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.get(self._base_url)
            return resp.status_code == 200
        except Exception as exc:
            logger.warning("health_check failed [%s]: %s", self._base_url, exc)
            return False
