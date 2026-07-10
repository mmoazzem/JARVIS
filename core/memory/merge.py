"""
merge(day_digests) -> Profile — memory Layer 2's reduction step.

Noise is dropped by CATEGORY RULE at merge-read, never by a per-fact
"durability" verdict: an LLM asked "worth remembering?" over-drops silently,
while a rule in code is owned, tunable in one constant (EPHEMERAL_CATEGORIES),
and reversible — raw digests keep everything, so changing the list and
re-merging is free, no re-extraction.

Same-subject disagreements stay UNRESOLVED in profile storage (resolution is
the Layer-3 seam); working_view() picks the single value the system prompt
presents, by trust then recency.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from pydantic import BaseModel as PydanticBaseModel

from core.constants import (
    ASSISTANT_SELF_SUBJECT_PREFIX,
    DIGEST_FILE_FORMAT,
    DIGEST_FILE_GLOB,
    DIGESTS_DIR,
    EPHEMERAL_CATEGORIES,
    FACT_SOURCE_USER,
    FACT_SOURCES,
    LOGGER_MEMORY,
    PROFILE_PATH,
)
from core.memory.base_digest import DayDigest, FactRecord
from core.memory.digest import fact_key

logger = logging.getLogger(LOGGER_MEMORY)

_TRUST_RANK = {source: rank for rank, source in enumerate(FACT_SOURCES)}


class Profile(PydanticBaseModel):
    """Durable facts reduced from all day digests; conflicts kept unresolved."""

    merged_at: str
    source_days: list[str]
    facts: list[FactRecord]


def merge(digests: list[DayDigest]) -> Profile:
    """Reduce day digests into the durable profile.

    Per subject: identical values collapse to one record carrying the
    strongest provenance; DISTINCT values are all kept and conflict-linked —
    the override (user beats tool beats assistant, newer beats older at equal
    trust) happens in working_view(), never by deleting a fact here.
    """
    # KNOWN, deferred to RAG: subject near-duplicate drift (FIFA_2026_final_date
    # vs 2026_FIFA_World_Cup_final_date) yields parallel entries here — semantic
    # retrieval absorbs it; exact-string subject keying is not worth patching.
    values: dict[str, dict[str, FactRecord]] = {}
    for day in sorted(digests, key=lambda d: d.date):
        for fact in day.facts:
            if fact.category in EPHEMERAL_CATEGORIES:
                continue  # the drop rule — read-time only, raw digests untouched
            if (
                fact.subject.startswith(ASSISTANT_SELF_SUBJECT_PREFIX)
                and fact.source != FACT_SOURCE_USER
            ):
                # Assistant self-reports ("functioning well") — category drifts
                # across runs, so this rule keys on subject + source instead.
                continue
            subject_values = values.setdefault(fact.subject, {})
            key = fact_key(fact.fact)
            held = subject_values.get(key)
            if held is None or _supersedes(fact, held):
                subject_values[key] = fact
    facts: list[FactRecord] = []
    for subject, subject_values in values.items():
        disagrees = len(subject_values) > 1
        for record in subject_values.values():
            copy = record.model_copy()
            copy.conflict_group = f"conflict:{subject}" if disagrees else None
            facts.append(copy)
    return Profile(
        merged_at=datetime.now(timezone.utc).isoformat(),
        source_days=sorted(d.date for d in digests),
        facts=facts,
    )


def working_view(profile: Profile) -> list[FactRecord]:
    """ONE fact per subject — highest trust, newest — for prompt rendering.

    Storage keeps every side of a conflict; the prompt must not, or the model
    argues with itself mid-answer.
    """
    best: dict[str, FactRecord] = {}
    for fact in profile.facts:
        held = best.get(fact.subject)
        if held is None or _supersedes(fact, held):
            best[fact.subject] = fact
    return [best[subject] for subject in sorted(best)]


def load_day_digests(digest_dir: Path = DIGESTS_DIR) -> list[DayDigest]:
    """Every readable day digest in the cache dir, date-ascending."""
    digests: list[DayDigest] = []
    for path in sorted(digest_dir.glob(DIGEST_FILE_GLOB)):
        try:
            datetime.strptime(path.name, DIGEST_FILE_FORMAT)
        except ValueError:
            continue  # .rejected.json dumps and other strays the glob sweeps in
        try:
            digests.append(DayDigest.model_validate_json(path.read_text(encoding="utf-8")))
        except Exception:
            # One bad cache file must not block merging the rest — but losing
            # a whole day from the profile is worth a console warning.
            logger.warning("skipping unreadable digest %s during merge", path)
    return digests


def save_profile(profile: Profile, path: Path = PROFILE_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(profile.model_dump_json(indent=2), encoding="utf-8")


def load_profile(path: Path = PROFILE_PATH) -> Optional[Profile]:
    """The stored profile, or None when absent/unreadable (re-merge rebuilds it)."""
    if not path.exists():
        return None
    try:
        return Profile.model_validate_json(path.read_text(encoding="utf-8"))
    except Exception:
        logger.warning("unreadable profile %s — ignoring until re-merged", path)
        return None


def _supersedes(challenger: FactRecord, incumbent: FactRecord) -> bool:
    """Higher trust wins; newer wins at equal trust (turn_ts is ISO-sortable)."""
    if challenger.source != incumbent.source:
        return _TRUST_RANK[challenger.source] < _TRUST_RANK[incumbent.source]
    return challenger.turn_ts > incumbent.turn_ts
