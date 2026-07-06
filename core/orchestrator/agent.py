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

import json
import logging
from datetime import datetime
from typing import AsyncIterator, Optional

from core.constants import (
    FALLBACK_MESSAGE,
    LOGGER_ORCHESTRATOR,
    RECOVERY_INSTRUCTION,
    ROLE_ASSISTANT,
    ROLE_TOOL,
    ROLE_USER,
)
from core.orchestrator.conversation import Conversation
from core.orchestrator.personality import build_system_prompt
from core.tools.registry import ToolRegistry
from models.base import BaseModel

logger = logging.getLogger(LOGGER_ORCHESTRATOR)


class Agent:
    def __init__(
        self,
        model: BaseModel,
        config,
        identity: dict,
        tools: Optional[ToolRegistry] = None,
    ) -> None:
        self._model = model
        self._config = config
        self._identity = identity
        self._conversation = Conversation(config.context_token_budget)
        # The tool seam: the model sees these schemas and decides when to call.
        # Adding a capability = registering a Tool; this loop never changes.
        self._tools = tools

    def _system_prompt(self) -> str:
        """Build the live system prompt: identity + small runtime state."""
        state = {
            # Date only — clock time comes from the get_time tool, so the prompt
            # never competes with (or pre-empts) the tool as a source of truth.
            "Current date": datetime.now().strftime("%Y-%m-%d"),
            "Mode": getattr(self._config, "mode", ""),
        }
        return build_system_prompt(
            self._identity,
            state,
            profile="",  # Layer 3 reserved for future memory — empty today.
            enable_thinking=self._config.enable_thinking,
        )

    async def respond(self, user_text: str) -> AsyncIterator[dict]:
        """Run one turn, yielding structured events. Never yields silent-empty.

        Tool turns are ONE clean loop — model→tool→model: the first call carries
        the tool schemas; if the model calls tools, each run is announced as a
        `delegation` event, results go back as tool messages, and a second call
        (WITHOUT tools, so it cannot chain) streams the final answer.
        """
        self._conversation.add_user(user_text)
        messages = self._conversation.to_messages(self._system_prompt())
        schemas = self._tools.schemas() if self._tools is not None else None

        # Signal that the model is working/reasoning before the first content token.
        yield {"type": "thinking"}

        content = ""
        tool_calls: list[dict] = []
        async for event in self._stream_content(messages, tools=schemas):
            if event["type"] == "token":
                content += event["content"]
                yield event
            elif event["type"] == "tool_call":
                tool_calls.append(event)  # internal — surfaced as `delegation` below
            else:
                yield event
                if event["type"] == "error":
                    return

        if tool_calls:
            messages = [*messages, _assistant_tool_message(content, tool_calls)]
            for call in tool_calls:
                yield {
                    "type": "delegation",
                    "tool": call["name"],
                    "status": self._tools.status_for(call["name"]),
                }
                result = await self._tools.call(call["name"], call["arguments"])
                messages.append({
                    "role": ROLE_TOOL,
                    "tool_call_id": call["id"],
                    "content": json.dumps(result, ensure_ascii=False),
                })
            # Final answer from the tool data — no tools on this call.
            yield {"type": "thinking"}
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
            yield {"type": "recovery"}
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

    async def _stream_content(
        self, messages: list[dict], tools: Optional[list[dict]] = None
    ) -> AsyncIterator[dict]:
        """Drive one model call, wrapping tokens/tool-calls/errors as events."""
        try:
            async for event in self._model.stream_events(messages, tools=tools):
                yield event
        except Exception as exc:  # honest, visible failure — not a silent drop
            logger.exception("model stream failed")
            yield {"type": "error", "message": str(exc)}


def _assistant_tool_message(content: str, tool_calls: list[dict]) -> dict:
    """The assistant turn that requested the tools, in the wire format the model
    expects to see back before the tool-result messages."""
    return {
        "role": ROLE_ASSISTANT,
        "content": content,
        "tool_calls": [
            {
                "id": call["id"],
                "type": "function",
                "function": {
                    "name": call["name"],
                    "arguments": json.dumps(call["arguments"], ensure_ascii=False),
                },
            }
            for call in tool_calls
        ],
    }
