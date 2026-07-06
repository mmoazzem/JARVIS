"""
Time tool — pure `datetime`, no network. The cheapest possible tool: it proves
the native tool-calling loop with zero external failure modes, and it is the
single source of truth for clock time (the system prompt carries only the date).
"""
from __future__ import annotations

from datetime import datetime

from core.tools.base import Tool


class TimeTool(Tool):
    name = "get_time"
    description = "Get the current local date and time."
    parameters = {"type": "object", "properties": {}, "required": []}
    status = "checking the time"

    async def run(self) -> dict:
        now = datetime.now().astimezone()
        return {
            "datetime": now.strftime("%Y-%m-%d %H:%M"),
            "weekday": now.strftime("%A"),
            "timezone": str(now.tzinfo),
        }
