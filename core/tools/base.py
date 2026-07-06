"""
Provider-agnostic tool interface — the seam every future tool implements.

A Tool is data-in, data-out: `run(**args)` returns STRUCTURED DATA (a dict),
never prose — the model turns data into words, interfaces decide display, same
discipline as everywhere else. A failing tool returns `{"error": ...}` rather
than raising, so a broken tool can never take a turn down (the registry
enforces this even if an implementation forgets).
"""
from __future__ import annotations

from abc import ABC, abstractmethod


class Tool(ABC):
    """One capability the model may invoke.

    Class attributes describe the tool to the model (name/description/parameters
    become the wire schema) and to the user (`status` renders while it runs,
    e.g. "checking the weather").
    """

    name: str
    description: str
    # JSON Schema for the arguments object, as the tools API expects.
    parameters: dict
    # Short human phrase shown while the tool runs — presentation data, not logic.
    status: str

    @abstractmethod
    async def run(self, **args) -> dict:
        """Execute with model-extracted args; return structured data or {"error": ...}."""
        raise NotImplementedError
