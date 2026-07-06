"""
Tool registry — adding a tool is registering an instance, no core changes.

The registry is what the Agent sees: it exposes every registered tool's schema
in the tools-API format and dispatches calls by name. `call()` NEVER raises —
an unknown tool or a crashing implementation comes back as `{"error": ...}`,
which flows to the model as data so the turn always completes with an honest
answer.
"""
from __future__ import annotations

import logging

from core.constants import LOGGER_TOOLS
from core.tools.base import Tool

logger = logging.getLogger(LOGGER_TOOLS)


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def schemas(self) -> list[dict]:
        """All registered tools in the OpenAI/Ollama `tools` wire format."""
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters,
                },
            }
            for tool in self._tools.values()
        ]

    def status_for(self, name: str) -> str:
        """Human status line for a running tool (fallback covers unknown names)."""
        tool = self._tools.get(name)
        return tool.status if tool is not None else f"using {name}"

    async def call(self, name: str, args: dict) -> dict:
        """Run a tool by name. Always returns a dict; never raises."""
        tool = self._tools.get(name)
        if tool is None:
            # The model hallucinated a tool name — tell it so, as data.
            logger.warning("model called unknown tool %r", name)
            return {"error": f"unknown tool: {name}"}
        try:
            result = await tool.run(**args)
            if "error" in result:
                logger.info("tool %s(%s) returned error: %s", name, args, result["error"])
            else:
                logger.info("tool %s(%s) ok", name, args)
            return result
        except Exception as exc:
            # Includes TypeError from bad model-extracted args — same contract.
            logger.warning("tool %s(%s) failed: %s", name, args, exc)
            return {"error": f"{name} failed: {exc}"}
