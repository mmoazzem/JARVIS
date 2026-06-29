"""
The Agent — the reusable unit (model + role-prompt + tools + run-loop).

The orchestrator is the first Agent instance. `respond()` is an async generator
that yields STRUCTURED EVENTS, not raw strings — that envelope is the seam a future
frontend consumes unchanged while today's CLI just prints `token` events.

Two M2 findings shape this loop:
  * The conversational path is STREAMING-ONLY. User turns drive `model.stream()`,
    never `complete()` — streaming resets the read clock so long reasoning never
    trips the 30s timeout and never triggers a retry-storm.
  * One model call per normal turn. If the model answers directly, that content IS
    the answer — we never discard it to re-synthesize.
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import AsyncIterator

from core.constants import (
    FALLBACK_MESSAGE,
    LOGGER_ORCHESTRATOR,
    RECOVERY_INSTRUCTION,
    ROLE_USER,
)
from core.orchestrator.conversation import Conversation
from core.orchestrator.personality import build_system_prompt
from models.base import BaseModel

logger = logging.getLogger(LOGGER_ORCHESTRATOR)


class Agent:
    def __init__(self, model: BaseModel, config, identity: dict) -> None:
        self._model = model
        self._config = config
        self._identity = identity
        self._conversation = Conversation(config.context_token_budget)

        # Reserved specialist seam: registered so future routing layers on without
        # restructuring the Agent. NEVER invoked in M3 — nothing loads DeepSeek.
        self._tools = {"deep_reason": self._deep_reason}

    def _system_prompt(self) -> str:
        """Build the live system prompt: identity + small runtime state."""
        state = {
            "Current date and time": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "Mode": getattr(self._config, "mode", ""),
        }
        return build_system_prompt(
            self._identity,
            state,
            profile="",  # Layer 3 reserved for future memory — empty today.
            enable_thinking=self._config.enable_thinking,
        )

    async def respond(self, user_text: str) -> AsyncIterator[dict]:
        """Run one turn, yielding structured events. Never yields silent-empty."""
        self._conversation.add_user(user_text)
        messages = self._conversation.to_messages(self._system_prompt())

        # Signal that the model is working/reasoning before the first content token.
        yield {"type": "thinking"}

        content = ""
        async for event in self._stream_content(messages):
            if event["type"] == "token":
                content += event["content"]
            yield event
            if event["type"] == "error":
                return

        # Gotcha #2 recovery: zero content (reasoning likely ate the budget). Try
        # EXACTLY ONCE more, re-asking for a direct answer to leave room for content.
        if not content.strip():
            logger.warning("zero-content turn — attempting one recovery")
            yield {"type": "thinking"}
            recovery = [*messages, {"role": ROLE_USER, "content": RECOVERY_INSTRUCTION}]
            async for event in self._stream_content(recovery):
                if event["type"] == "token":
                    content += event["content"]
                yield event
                if event["type"] == "error":
                    return

        if not content.strip():
            # Still nothing — fail loudly and visibly, never silently empty.
            logger.error("recovery failed — emitting fallback message")
            yield {"type": "token", "content": FALLBACK_MESSAGE}
            yield {"type": "done"}
            return

        self._conversation.add_assistant(content)
        yield {"type": "done"}

    async def _stream_content(self, messages: list[dict]) -> AsyncIterator[dict]:
        """Drive one model.stream() call, wrapping tokens/errors as events."""
        try:
            async for token in self._model.stream(messages):
                yield {"type": "token", "content": token}
        except Exception as exc:  # honest, visible failure — not a silent drop
            logger.exception("model stream failed")
            yield {"type": "error", "message": str(exc)}

    async def _deep_reason(self, *args, **kwargs):
        """RESERVED specialist seam (DeepSeek-R1 reasoning swap).

        Registered but NEVER invoked in M3 — a second 14B model cannot co-reside in
        16 GB (CLAUDE.md), so activating this requires an unload/load swap. Wired
        here only so future multi-agent routing layers on cleanly.
        """
        raise NotImplementedError("deep_reason is a reserved specialist seam; inactive in M3")
