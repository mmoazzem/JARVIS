"""
BaseDigest — the swappable fact-extraction seam (same pattern as BaseModel).

Memory Layer 2: an extractor reads one day's cleaned exchanges and emits typed
FactRecords. The rest of the app depends only on this interface and the schemas
below, never on how extraction happens — the LLM-backed implementation
(llm_digest.py) is the default; a different model or a non-LLM extractor slots
in behind the same seam.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

from pydantic import BaseModel as PydanticBaseModel, field_validator

from core.constants import FACT_SOURCES


class FactRecord(PydanticBaseModel):
    """One extracted claim.

    `subject` is the identity key: two facts are the SAME real-world fact (an
    override or conflict candidate) if and only if they share a subject — all
    merge logic keys on it. `source` carries trust (FACT_SOURCES, ordered
    highest first); `turn_ts` carries provenance. Facts that share a subject
    but disagree are linked by a common `conflict_group` and BOTH kept —
    surfacing contradictions is the point; resolving them is Layer 3's job.
    """

    subject: str
    fact: str
    category: str
    source: str
    turn_ts: str
    conflict_group: Optional[str] = None

    @field_validator("source")
    @classmethod
    def _known_source(cls, value: str) -> str:
        if value not in FACT_SOURCES:
            raise ValueError(f"unknown source {value!r} — valid: {FACT_SOURCES}")
        return value


class DayDigest(PydanticBaseModel):
    """All facts extracted from one day-file; persisted as the digest cache."""

    date: str  # YYYY-MM-DD, derived from the event file's own name
    source_file: str
    extracted_at: str
    # Which model/implementation produced this — cached digests outlive model
    # swaps, so the file must say where its facts came from.
    extractor: str
    facts: list[FactRecord]


class BaseDigest(ABC):
    """Async interface every fact extractor implements."""

    # Provenance label recorded on the DayDigest (the model id for LLM extractors).
    extractor_id: str = ""

    @abstractmethod
    async def extract(self, exchanges: list[dict]) -> list[FactRecord]:
        """Emit fact records from one day of cleaned exchanges.

        Each exchange: {"ts", "user", "assistant", "tools"} — assistant text
        arrives already markdown-stripped, and "tools" lists the tool names the
        turn delegated to (the tool_derived signal).
        """
        raise NotImplementedError
