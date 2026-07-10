"""
digest(day_file) -> DayDigest — memory Layer 2, the extraction entrypoint.

Turns one raw event-log day-file (Layer 1) into typed facts, persisted under
data/digests/ as a CACHE: the expensive LLM pass runs once per day-file, and
merge (stage 2) is then a cheap reduction over the cached JSON files, never a
re-extraction. On-demand only — the /digest command or a direct call, no
scheduler.
"""
from __future__ import annotations

import json
import logging
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import AsyncIterator, Optional

from core.constants import (
    DIGEST_FILE_FORMAT,
    DIGESTS_DIR,
    EVENT_LOG_FILE_FORMAT,
    EVENT_LOG_GLOB,
    EVENTS_LOG_DIR,
    FACT_SOURCE_ASSISTANT,
    FACT_SOURCE_TOOL,
    FACT_SOURCE_USER,
    LOG_DATE_FORMAT,
    LOGGER_MEMORY,
)
from core.memory.base_digest import BaseDigest, DayDigest, FactRecord

logger = logging.getLogger(LOGGER_MEMORY)

# Markdown constructs removed before extraction, applied in order. Assistant
# answers are markdown-heavy (headings, bold, tables); facts must be plain text
# so the extractor sees "Date: July 19, 2026", not pipes and asterisks.
_MD_PATTERNS = (
    (re.compile(r"```.*?```", re.DOTALL), " "),  # fenced code blocks
    (re.compile(r"`([^`]*)`"), r"\1"),  # inline code
    (re.compile(r"!\[([^\]]*)\]\([^)]*\)"), r"\1"),  # images -> alt text
    (re.compile(r"\[([^\]]*)\]\([^)]*\)"), r"\1"),  # links -> anchor text
    (re.compile(r"^#{1,6}\s*", re.MULTILINE), ""),  # heading markers
    (re.compile(r"^\s*[-*+]\s+", re.MULTILINE), ""),  # bullet markers
    (re.compile(r"[*_]{1,3}([^*_]+)[*_]{1,3}"), r"\1"),  # bold / italic
    (re.compile(r"^[\s|:\-]+$", re.MULTILINE), ""),  # table separators / rules
    (re.compile(r"\|"), " "),  # table cell pipes
)


def strip_markdown(text: str) -> str:
    """Reduce a markdown-formatted answer to plain text for extraction."""
    for pattern, repl in _MD_PATTERNS:
        text = pattern.sub(repl, text)
    text = re.sub(r"[ \t]+", " ", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def digest_path(date: str, out_dir: Path = DIGESTS_DIR) -> Path:
    """Where the digest for a YYYY-MM-DD day lives on disk."""
    return out_dir / datetime.strptime(date, LOG_DATE_FORMAT).strftime(DIGEST_FILE_FORMAT)


def atomic_write_text(path: Path, text: str) -> None:
    """Replace path's content all-or-nothing (write sibling, os.replace).

    A plain write_text truncates first: a crash or a concurrent writer mid-way
    leaves a torn JSON file, and a torn digest silently drops its whole day
    from every merge until someone re-digests. Shared with merge's profile
    write — same file, same stakes.
    """
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


async def digest(
    day_file: Path,
    extractor: BaseDigest,
    *,
    force: bool = False,
    out_dir: Path = DIGESTS_DIR,
) -> DayDigest:
    """Extract one day-file into a DayDigest, reusing the on-disk cache unless forced.

    FileNotFoundError propagates when the day-file is absent — the caller named
    the day and decides how to say "nothing happened that day".
    """
    date = datetime.strptime(day_file.name, EVENT_LOG_FILE_FORMAT).strftime(LOG_DATE_FORMAT)
    out_path = digest_path(date, out_dir)
    cached = _load_cache(out_path)
    if cached is not None and not force:
        # A zero-fact cache is NOT trusted: it can only be a genuinely empty
        # day (the write guard below blocks degenerate overwrites), and
        # re-checking one costs little — so it self-heals if events arrived.
        if cached.facts:
            return cached

    exchanges = _read_exchanges(day_file)
    facts = _link_conflicts(_ground_sources(await extractor.extract(exchanges), exchanges))
    result = DayDigest(
        date=date,
        source_file=day_file.name,
        extracted_at=datetime.now(timezone.utc).isoformat(),
        extractor=extractor.extractor_id or type(extractor).__name__,
        facts=facts,
    )
    if cached is not None and len(facts) < len(cached.facts):
        # A weaker extraction must never destroy a better cache: the model is
        # non-deterministic, and losing verified facts is silent data loss —
        # union passes only ever stabilize recall UPWARD, so fewer facts than
        # the cache holds (zero included) marks a degenerate run, --force or
        # not. Deleting the digest file is the deliberate way to accept a
        # smaller one. The rejected output lands beside the cache for
        # inspection.
        rejected = out_path.with_suffix(".rejected.json")
        atomic_write_text(rejected, result.model_dump_json(indent=2))
        logger.warning(
            "digest of %s yielded %d facts, fewer than the %d-fact cache — "
            "keeping the cache (rejected output -> %s; delete %s to accept "
            "a smaller digest)",
            day_file.name,
            len(facts),
            len(cached.facts),
            rejected,
            out_path.name,
        )
        return cached
    out_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(out_path, result.model_dump_json(indent=2))
    logger.info("digested %s: %d facts -> %s", day_file.name, len(facts), out_path)
    return result


async def digest_all(
    extractor: BaseDigest,
    *,
    events_dir: Path = EVENTS_LOG_DIR,
    out_dir: Path = DIGESTS_DIR,
    force: bool = False,
) -> AsyncIterator[tuple[str, Optional[DayDigest]]]:
    """Digest every day-file in events_dir, skipping already-digested days.

    Idempotency is CACHE-FILE EXISTENCE, never a progress cursor: a separate
    "last digested" pointer can drift from what is actually on disk, while a
    digest file either exists or it doesn't. That also makes an aborted bulk
    run resumable — finished days are skipped on the next invocation.

    Yields (date, digest) per day-file, digest None when skipped, so callers
    can report progress during a long multi-day extraction.
    """
    for day_file in sorted(events_dir.glob(EVENT_LOG_GLOB)):
        try:
            date = datetime.strptime(day_file.name, EVENT_LOG_FILE_FORMAT).strftime(
                LOG_DATE_FORMAT
            )
        except ValueError:
            continue  # a stray file that merely resembles a day-file
        if not force and digest_path(date, out_dir).exists():
            yield date, None
            continue
        yield date, await digest(day_file, extractor, force=force, out_dir=out_dir)


def _load_cache(out_path: Path) -> DayDigest | None:
    """The cached digest for a day, or None when absent/unreadable."""
    if not out_path.exists():
        return None
    try:
        return DayDigest.model_validate_json(out_path.read_text(encoding="utf-8"))
    except Exception:
        # A corrupt cache file is a miss, not a crash — extraction rebuilds it.
        logger.warning("unreadable digest cache %s — re-extracting", out_path)
        return None


def _read_exchanges(day_file: Path) -> list[dict]:
    """Load the day's exchange records, cleaned for the extractor.

    A malformed LINE or FIELD (torn write, foreign JSON shape, null/typed-wrong
    value) loses at most that record — never the day. Usable fields of a partly
    broken record are kept; a record with neither user nor assistant text
    carries no facts and is dropped entirely.
    """
    exchanges: list[dict] = []
    for line in day_file.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            # A torn line (crash mid-write) loses that record, never the day.
            # Routine and handled — log-file only, not console-worthy.
            logger.info("skipping unparseable event line in %s", day_file.name)
            continue
        if not isinstance(record, dict):
            logger.info("skipping non-object event line in %s", day_file.name)
            continue
        if record.get("role") != "exchange":
            continue  # standalone events (e.g. speech interruptions) carry no facts
        user = _text(record.get("user"))
        assistant = _text(record.get("assistant"))
        if not user and not assistant:
            continue
        events = record.get("events")
        exchanges.append(
            {
                "ts": _text(record.get("ts")),
                "user": user,
                "assistant": strip_markdown(assistant),
                "tools": [
                    _text(e.get("tool"))
                    for e in (events if isinstance(events, list) else [])
                    if isinstance(e, dict) and e.get("type") == "delegation"
                ],
            }
        )
    return exchanges


def _text(value) -> str:
    """A field's string value, or "" when absent/null/typed wrong."""
    return value if isinstance(value, str) else ""


def _ground_sources(facts: list[FactRecord], exchanges: list[dict]) -> list[FactRecord]:
    """Re-derive assistant-side trust from the event log, not the model.

    Whether a turn delegated to a tool is a FACT of Layer 1, keyed by turn_ts —
    so tool_derived vs assistant_claimed is computed here deterministically.
    Only user_asserted stays the model's call: "did the user assert this" is a
    semantic judgment the log alone cannot make.
    """
    tools_by_ts: dict[str, bool] = {}
    for exchange in exchanges:
        tools_by_ts[exchange["ts"]] = bool(exchange["tools"])
        # The model sometimes truncates tz/microseconds when copying ts.
        tools_by_ts.setdefault(exchange["ts"][:19], bool(exchange["tools"]))
    for fact in facts:
        if fact.source == FACT_SOURCE_USER:
            continue
        had_tools = tools_by_ts.get(fact.turn_ts, tools_by_ts.get(fact.turn_ts[:19]))
        if had_tools is None:
            continue  # mangled ts: keep the (floored) source rather than guess
        fact.source = FACT_SOURCE_TOOL if had_tools else FACT_SOURCE_ASSISTANT
    return facts


def fact_key(fact: str) -> str:
    """Two claims differing only in trailing punctuation/case do not disagree.

    Shared by conflict linking here and by merge's cross-day dedupe — both
    must agree on what "the same value" means.
    """
    return fact.strip().rstrip(".").casefold()


def _link_conflicts(facts: list[FactRecord]) -> list[FactRecord]:
    """Guarantee same-subject disagreements share ONE conflict_group.

    The LLM is asked to link conflicts, but merge DEPENDS on the invariant, so
    groups are rebuilt from scratch here: model-assigned ids are chunk-local
    (the same "c1" from two chunks names two unrelated conflicts) and get
    stamped on lone facts, so subject + disagreeing values is the only
    authority. Conflicts are linked, never resolved — resolution is the
    Layer-3 seam.
    """
    by_subject: dict[str, list[FactRecord]] = {}
    for fact in facts:
        by_subject.setdefault(fact.subject, []).append(fact)
    for subject, group in by_subject.items():
        disagrees = len({fact_key(f.fact) for f in group}) > 1
        for fact in group:
            fact.conflict_group = f"conflict:{subject}" if disagrees else None
    return facts
