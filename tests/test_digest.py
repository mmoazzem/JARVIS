"""Memory Layer-2 digest invariants (core/memory/) — mocked model, no Ollama.

Pinned behaviors:
  * strip_markdown reduces markdown-heavy answers to plain text before the
    LLM pass — facts must not carry pipes, asterisks, or link syntax.
  * FactRecord rejects sources outside the trust vocabulary.
  * LLMDigest runs at DIGEST_TEMPERATURE with thinking suppressed, tolerates
    prose around the JSON array, skips malformed records, and raises (never
    caches emptiness) on zero-content or array-less output.
  * digest() caches per day-file: extraction runs once, force re-extracts,
    torn lines and non-exchange records are skipped, never fatal.
  * Same-subject disagreements ALWAYS share one conflict_group — enforced
    deterministically even when the extractor fails to link them.
"""
import json
from typing import AsyncIterator

import pytest

from core.constants import DIGEST_TEMPERATURE, FACT_CATEGORIES, NO_THINK_DIRECTIVE
from core.memory.base_digest import BaseDigest, FactRecord
from core.memory.digest import _link_conflicts, digest, digest_all, digest_path, strip_markdown
from core.memory.llm_digest import _EXTRACTION_PROMPT, LLMDigest
from models.base import BaseModel, ModelResponse, WarmupResult


def make_fact(**overrides) -> FactRecord:
    base = {
        "subject": "user_home_location",
        "fact": "The user lives in Buffalo, NY.",
        "category": "personal_fact",
        "source": "user_asserted",
        "turn_ts": "2026-07-07T23:03:07+00:00",
        "conflict_group": None,
    }
    return FactRecord(**{**base, **overrides})


class FakeModel(BaseModel):
    """Scripted BaseModel: returns canned completions in order (the last one
    repeats), recording every call — lets tests drive the retry path."""

    model_id = "fake-extractor"

    def __init__(self, *contents: str) -> None:
        self._contents = list(contents)
        self.calls: list[list[dict]] = []
        self.messages: list[dict] = []
        self.opts: dict = {}

    async def complete(self, messages, **opts) -> ModelResponse:
        self.calls.append(messages)
        self.messages = messages
        self.opts = opts
        content = self._contents.pop(0) if len(self._contents) > 1 else self._contents[0]
        return ModelResponse(content=content, model=self.model_id)

    async def stream(self, messages, **opts) -> AsyncIterator[str]:
        yield ""

    async def health_check(self) -> bool:
        return True

    async def warmup(self) -> WarmupResult:
        return WarmupResult(success=True, model_id=self.model_id, elapsed_s=0.0)

    async def unload(self) -> None:
        pass


class FakeExtractor(BaseDigest):
    """Scripted extractor: returns canned facts, records what it was given."""

    extractor_id = "fake"

    def __init__(self, facts=()):
        self._facts = list(facts)
        self.calls: list[list[dict]] = []

    async def extract(self, exchanges):
        self.calls.append(exchanges)
        return [f.model_copy() for f in self._facts]


# --- strip_markdown ---------------------------------------------------------


def test_strip_markdown_flattens_answer_formatting():
    text = (
        "### Key Details:\n"
        "- **Date**: July 19, 2026\n"
        "| Venue | MetLife Stadium |\n"
        "|---|---|\n"
        "See [the schedule](https://example.org/schedule) or `import this`.\n"
        "```python\nprint('hi')\n```\n"
    )

    plain = strip_markdown(text)

    for artifact in ("#", "**", "|", "](", "```"):
        assert artifact not in plain
    for kept in ("Date: July 19, 2026", "Venue", "MetLife Stadium", "the schedule", "import this"):
        assert kept in plain


# --- FactRecord -------------------------------------------------------------


def test_fact_record_rejects_unknown_source():
    with pytest.raises(ValueError):
        make_fact(source="model_guessed")


def test_fact_record_rejects_off_enum_category():
    with pytest.raises(ValueError):
        make_fact(category="sports_result")  # the old free-text vocabulary


def test_every_category_is_spelled_out_in_the_prompt():
    # The prompt is the enum's contract with the model — a category added to
    # the constant but not the prompt would floor every use to the fallback.
    for category in FACT_CATEGORIES:
        assert f'"{category}"' in _EXTRACTION_PROMPT


# --- LLMDigest --------------------------------------------------------------

EXCHANGES = [
    {
        "ts": "2026-07-07T23:03:07+00:00",
        "user": "Who won the match?",
        "assistant": "Argentina won 3-2.",
        "tools": ["web_search"],
    }
]

GOOD_JSON = json.dumps(
    [
        {
            "subject": "ARG_vs_EGT_2026-07-07_result",
            "fact": "Argentina beat Egypt 3-2.",
            "category": "world_fact",
            "source": "tool_derived",
            "turn_ts": "2026-07-07T23:03:07+00:00",
            "conflict_group": None,
        }
    ]
)


async def test_llm_digest_extracts_records_with_pinned_call_shape():
    model = FakeModel(f"Here you go:\n{GOOD_JSON}\nDone.")

    facts = await LLMDigest(model).extract(EXCHANGES)

    assert [f.subject for f in facts] == ["ARG_vs_EGT_2026-07-07_result"]
    assert facts[0].source == "tool_derived"
    # Extraction is mechanical: near-zero temperature, thinking suppressed.
    assert model.opts["temperature"] == DIGEST_TEMPERATURE
    assert NO_THINK_DIRECTIVE in model.messages[0]["content"]
    # The transcript must carry the tool_derived signal and the turn ts.
    assert "tools used: web_search" in model.messages[1]["content"]
    assert "ts=2026-07-07T23:03:07+00:00" in model.messages[1]["content"]
    # The task instruction is re-anchored AFTER the log: on long days the
    # transcript ends with arbitrary content the model must not start answering.
    transcript_end = model.messages[1]["content"].rindex(">>>")
    assert "output ONLY the JSON array" in model.messages[1]["content"][transcript_end:]


async def test_llm_digest_timeout_is_per_call_opt():
    model = FakeModel(GOOD_JSON)
    await LLMDigest(model, timeout=42.0).extract(EXCHANGES)
    assert model.opts["timeout"] == 42.0

    model = FakeModel(GOOD_JSON)
    await LLMDigest(model).extract(EXCHANGES)
    assert "timeout" not in model.opts  # None keeps the model's own default


async def test_llm_digest_skips_malformed_records_keeps_rest():
    bad_and_good = json.dumps([{"subject": "orphan"}, json.loads(GOOD_JSON)[0]])
    model = FakeModel(bad_and_good)

    facts = await LLMDigest(model).extract(EXCHANGES)

    assert len(facts) == 1
    assert facts[0].subject == "ARG_vs_EGT_2026-07-07_result"


async def test_llm_digest_raises_on_output_without_array():
    with pytest.raises(ValueError):
        await LLMDigest(FakeModel("I could not find any facts.")).extract(EXCHANGES)


async def test_llm_digest_raises_on_zero_content():
    # The reasoning-ate-the-budget gotcha must surface, never cache empty.
    with pytest.raises(ValueError):
        await LLMDigest(FakeModel("")).extract(EXCHANGES)


# --- normalization: the model drifts on key names between runs (observed live) --


async def test_key_drift_normalizes_instead_of_skipping():
    drifted = json.dumps(
        [
            {
                "subject": "ARG_vs_EGT_2026-07-07_result",
                "fact": "Argentina beat Egypt 3-2.",
                # no "category" at all, "turn" instead of "turn_ts" — run 2's shape
                "source": "tool_derived",
                "turn": "2026-07-07T23:03:07+00:00",
                "conflict_group": None,
            }
        ]
    )

    facts = await LLMDigest(FakeModel(drifted)).extract(EXCHANGES)

    assert len(facts) == 1
    assert facts[0].turn_ts == "2026-07-07T23:03:07+00:00"
    assert facts[0].category == "world_fact"  # omitted category floors durable


async def test_source_value_drift_floors_to_lowest_trust_never_drops():
    drifted = json.dumps(
        [
            {
                "subject": "a",
                "claim": "Something happened.",  # "claim" -> "fact"
                "type": "world_fact",  # "type" -> "category"
                "source": "Tool-Derived",  # case/hyphen drift
                "ts": "2026-07-07T23:03:07+00:00",  # "ts" -> "turn_ts"
                "conflict": "c9",  # "conflict" -> "conflict_group"
            },
            {
                "subject": "b",
                "fact": "Provenance labels are not trust levels.",
                "category": "other",
                "source": "web_search",  # observed live: tool name in source
                "turn_ts": "2026-07-07T23:03:07+00:00",
            },
            {
                "subject": "c",
                "fact": "Asking is not asserting.",
                "category": "other",
                "source": "User question",  # must NOT inflate to user_asserted
                "turn_ts": "2026-07-07T23:03:07+00:00",
            },
        ]
    )

    facts = await LLMDigest(FakeModel(drifted)).extract(EXCHANGES)

    assert [f.subject for f in facts] == ["a", "b", "c"]
    assert facts[0].fact == "Something happened."
    assert facts[0].category == "world_fact"
    assert facts[0].source == "tool_derived"
    assert facts[0].conflict_group == "c9"
    # Unknown source values floor to the LOWEST trust class instead of killing
    # the record; grounding against the event log upgrades tool-backed ones.
    assert facts[1].source == "assistant_claimed"
    assert facts[2].source == "assistant_claimed"


async def test_off_enum_category_floors_to_durable_fallback_not_dropped():
    drifted = json.dumps(
        [
            {
                "subject": "match_result",
                "fact": "Argentina beat Egypt 3-2.",
                "category": "sports_result",  # the old drifting vocabulary
                "source": "tool_derived",
                "turn_ts": "2026-07-07T23:03:07+00:00",
            }
        ]
    )

    facts = await LLMDigest(FakeModel(drifted)).extract(EXCHANGES)

    assert len(facts) == 1
    assert facts[0].category == "world_fact"


async def test_user_assertion_spellings_map_to_user_asserted():
    drifted = json.dumps(
        [
            {
                "subject": "user_home_location",
                "fact": "The user lives in Buffalo, NY.",
                "category": "personal_fact",
                "source": "User correction",
                "turn_ts": "2026-07-07T23:03:07+00:00",
            }
        ]
    )

    facts = await LLMDigest(FakeModel(drifted)).extract(EXCHANGES)

    assert facts[0].source == "user_asserted"


# --- chunking: long days lose mid-transcript facts in one pass (observed live) --


def make_exchanges(n):
    return [
        {"ts": f"2026-07-07T0{i}:00:00+00:00", "user": f"q{i}", "assistant": f"a{i}", "tools": []}
        for i in range(n)
    ]


async def test_long_days_are_chunked_with_overlap_and_deduped():
    model = FakeModel(GOOD_JSON)

    facts = await LLMDigest(model).extract(make_exchanges(9))

    assert len(model.calls) == 2  # 9 exchanges, window 8, overlap 1
    first, second = (calls[1]["content"] for calls in model.calls)
    assert "q0" in first and "q7" in first and "q8" not in first
    assert "q7" in second and "q8" in second and "q6" not in second
    assert len(facts) == 1  # the overlap's re-extraction deduped away


async def test_failed_chunk_does_not_void_other_chunks():
    # Chunk 1 stays degenerate through its retry; chunk 2 extracts fine.
    model = FakeModel("junk", "still junk", GOOD_JSON)

    facts = await LLMDigest(model).extract(make_exchanges(9))

    assert len(model.calls) == 3
    assert [f.subject for f in facts] == ["ARG_vs_EGT_2026-07-07_result"]


async def test_all_chunks_failing_raises_loudly():
    model = FakeModel("junk", "junk", "junk", "junk")

    with pytest.raises(ValueError):
        await LLMDigest(model).extract(make_exchanges(9))


# --- multi-pass union: a single pass misses facts non-deterministically, and a
# --- run-once cache would freeze the miss forever (observed live) -------------

FACT_BUFFALO = {
    "subject": "user_home_location",
    "fact": "The user lives in Buffalo, NY.",
    "category": "personal_fact",
    "source": "user_asserted",
    "turn_ts": "2026-07-07T23:03:07+00:00",
    "conflict_group": None,
}


async def test_union_catches_facts_missed_in_one_pass():
    # Pass 1 sees only the match result; pass 2 only Buffalo — the union has both.
    model = FakeModel(GOOD_JSON, json.dumps([FACT_BUFFALO]))

    facts = await LLMDigest(model, passes=2).extract(EXCHANGES)

    assert len(model.calls) == 2
    assert {f.subject for f in facts} == {
        "ARG_vs_EGT_2026-07-07_result",
        "user_home_location",
    }


async def test_union_collapses_re_extractions_keeping_strongest_provenance():
    # The same value re-caught by another pass (case/period drift included)
    # is ONE record carrying the highest-trust source seen.
    weaker = {**FACT_BUFFALO, "fact": "the user lives in Buffalo, NY", "source": "assistant_claimed"}
    model = FakeModel(json.dumps([weaker]), json.dumps([FACT_BUFFALO]))

    facts = await LLMDigest(model, passes=2).extract(EXCHANGES)

    assert len(facts) == 1
    assert facts[0].source == "user_asserted"


async def test_union_keeps_same_subject_disagreements_unresolved():
    other_value = {**FACT_BUFFALO, "fact": "The user lives in Rochester, NY."}
    model = FakeModel(json.dumps([FACT_BUFFALO]), json.dumps([other_value]))

    facts = await LLMDigest(model, passes=2).extract(EXCHANGES)

    # Both values survive for digest()'s deterministic conflict linking.
    assert sorted(f.fact for f in facts) == [
        "The user lives in Buffalo, NY.",
        "The user lives in Rochester, NY.",
    ]


async def test_failed_pass_does_not_void_other_passes():
    # Pass 1 stays degenerate through its retry; pass 2 extracts fine.
    model = FakeModel("junk", "still junk", GOOD_JSON)

    facts = await LLMDigest(model, passes=2).extract(EXCHANGES)

    assert len(model.calls) == 3
    assert [f.subject for f in facts] == ["ARG_vs_EGT_2026-07-07_result"]


async def test_all_passes_failing_raises_loudly():
    model = FakeModel("junk")

    with pytest.raises(ValueError):
        await LLMDigest(model, passes=2).extract(EXCHANGES)

    assert len(model.calls) == 4  # 2 passes × (attempt + one retry)


async def test_empty_day_short_circuits_without_model_calls():
    model = FakeModel(GOOD_JSON)

    facts = await LLMDigest(model, passes=3).extract([])

    assert facts == []
    assert model.calls == []  # never ask the model to invent facts from nothing


# --- one-shot re-extract on degenerate output ---------------------------------


async def test_degenerate_first_pass_retries_once_and_recovers():
    model = FakeModel("no array here at all", GOOD_JSON)

    facts = await LLMDigest(model).extract(EXCHANGES)

    assert len(model.calls) == 2
    assert [f.subject for f in facts] == ["ARG_vs_EGT_2026-07-07_result"]
    # The retry re-asks with a stricter nudge appended to the transcript.
    assert "exactly these keys" in model.calls[1][1]["content"]


async def test_degenerate_twice_raises_loudly():
    model = FakeModel("still not json", "still not json")

    with pytest.raises(ValueError):
        await LLMDigest(model).extract(EXCHANGES)

    assert len(model.calls) == 2  # exactly one retry, then give up


async def test_legitimately_empty_array_returns_empty_after_retry():
    model = FakeModel("[]")

    facts = await LLMDigest(model).extract(EXCHANGES)

    assert facts == []
    assert len(model.calls) == 2  # empty is retried once, then accepted as empty


# --- digest() ---------------------------------------------------------------


def write_day_file(tmp_path, records, name="events_2026-07-07.jsonl"):
    day_file = tmp_path / name
    day_file.write_text("\n".join(records) + "\n", encoding="utf-8")
    return day_file


DAY_RECORDS = [
    json.dumps(
        {
            "ts": "2026-07-07T23:03:07+00:00",
            "role": "exchange",
            "user": "Who won?",
            "assistant": "**Argentina** won 3-2.",
            "events": [{"type": "delegation", "tool": "web_search"}],
        }
    ),
    json.dumps({"ts": "2026-07-07T23:05:00+00:00", "role": "event", "type": "speech_interrupted"}),
    '{"torn line',
]


async def test_digest_cleans_exchanges_and_persists(tmp_path):
    day_file = write_day_file(tmp_path, DAY_RECORDS)
    extractor = FakeExtractor([make_fact()])
    out_dir = tmp_path / "digests"

    result = await digest(day_file, extractor, out_dir=out_dir)

    # Only the exchange survived; markdown stripped; delegation became a tool name.
    [exchanges] = extractor.calls
    assert exchanges == [
        {
            "ts": "2026-07-07T23:03:07+00:00",
            "user": "Who won?",
            "assistant": "Argentina won 3-2.",
            "tools": ["web_search"],
        }
    ]
    assert result.date == "2026-07-07"
    assert result.source_file == day_file.name
    assert result.extractor == "fake"
    on_disk = json.loads((out_dir / "digest_2026-07-07.json").read_text())
    assert on_disk["facts"][0]["subject"] == "user_home_location"


async def test_digest_is_cached_until_forced(tmp_path):
    day_file = write_day_file(tmp_path, DAY_RECORDS)
    extractor = FakeExtractor([make_fact()])
    out_dir = tmp_path / "digests"

    first = await digest(day_file, extractor, out_dir=out_dir)
    cached = await digest(day_file, extractor, out_dir=out_dir)
    forced = await digest(day_file, extractor, force=True, out_dir=out_dir)

    assert len(extractor.calls) == 2  # first + forced; the cached call cost nothing
    assert cached == first
    assert forced.facts == first.facts


async def test_zero_fact_result_never_destroys_good_cache(tmp_path):
    day_file = write_day_file(tmp_path, DAY_RECORDS)
    out_dir = tmp_path / "digests"
    good = await digest(day_file, FakeExtractor([make_fact()]), out_dir=out_dir)

    kept = await digest(day_file, FakeExtractor([]), force=True, out_dir=out_dir)

    assert kept.facts == good.facts
    on_disk = json.loads((out_dir / "digest_2026-07-07.json").read_text())
    assert len(on_disk["facts"]) == 1  # the verified cache survived
    rejected = json.loads((out_dir / "digest_2026-07-07.rejected.json").read_text())
    assert rejected["facts"] == []  # the degenerate output is kept for inspection


async def test_zero_facts_write_normally_when_no_prior_cache(tmp_path):
    day_file = write_day_file(tmp_path, DAY_RECORDS)
    out_dir = tmp_path / "digests"

    result = await digest(day_file, FakeExtractor([]), out_dir=out_dir)

    assert result.facts == []
    assert (out_dir / "digest_2026-07-07.json").exists()


async def test_lower_fact_count_never_destroys_better_cache(tmp_path):
    # The zero-fact guard, generalized: union recall only stabilizes upward,
    # so a re-roll that comes back SMALLER than the cache is a degenerate run.
    day_file = write_day_file(tmp_path, DAY_RECORDS)
    out_dir = tmp_path / "digests"
    two = [make_fact(), make_fact(subject="user_editor", fact="The user prefers vim.")]
    good = await digest(day_file, FakeExtractor(two), out_dir=out_dir)

    kept = await digest(day_file, FakeExtractor([make_fact()]), force=True, out_dir=out_dir)

    assert kept.facts == good.facts
    on_disk = json.loads((out_dir / "digest_2026-07-07.json").read_text())
    assert len(on_disk["facts"]) == 2  # the better cache survived
    rejected = json.loads((out_dir / "digest_2026-07-07.rejected.json").read_text())
    assert len(rejected["facts"]) == 1  # the smaller output kept for inspection


async def test_equal_fact_count_still_overwrites_on_force(tmp_path):
    # The guard blocks REGRESSIONS only — a same-size re-roll is a legitimate
    # refresh (e.g. after an extractor change) and must land.
    day_file = write_day_file(tmp_path, DAY_RECORDS)
    out_dir = tmp_path / "digests"
    await digest(day_file, FakeExtractor([make_fact()]), out_dir=out_dir)

    refreshed = await digest(
        day_file,
        FakeExtractor([make_fact(fact="The user lives in Buffalo, New York.")]),
        force=True,
        out_dir=out_dir,
    )

    on_disk = json.loads((out_dir / "digest_2026-07-07.json").read_text())
    assert on_disk["facts"][0]["fact"] == "The user lives in Buffalo, New York."
    assert refreshed.facts[0].fact == "The user lives in Buffalo, New York."


async def test_zero_fact_cache_is_reextracted_not_trusted(tmp_path):
    day_file = write_day_file(tmp_path, DAY_RECORDS)
    out_dir = tmp_path / "digests"
    await digest(day_file, FakeExtractor([]), out_dir=out_dir)  # empty cache on disk

    extractor = FakeExtractor([make_fact()])
    healed = await digest(day_file, extractor, out_dir=out_dir)  # no force needed

    assert len(extractor.calls) == 1  # the empty cache did not satisfy the read
    assert len(healed.facts) == 1
    on_disk = json.loads((out_dir / "digest_2026-07-07.json").read_text())
    assert len(on_disk["facts"]) == 1  # and the heal was persisted


async def test_malformed_records_lose_at_most_themselves_never_the_day(tmp_path):
    # Adversarial sweep finding: a JSONL line holding valid-but-non-object JSON
    # ("42") crashed the WHOLE day at record.get; null/typed-wrong fields would
    # crash strip_markdown next. Every malformed shape must cost at most its
    # own record.
    hostile = [
        json.dumps({"ts": "2026-07-07T01:00:00+00:00", "role": "exchange",
                    "user": "first good turn", "assistant": "ok", "events": []}),
        '{"torn line',
        "42",  # valid JSON, not an object
        '"just a string"',
        "[1, 2, 3]",
        json.dumps({"role": "exchange"}),  # exchange with no text at all
        json.dumps({"ts": "2026-07-07T02:00:00+00:00", "role": "exchange",
                    "user": "null assistant", "assistant": None, "events": []}),
        json.dumps({"ts": None, "role": "exchange", "user": 123,
                    "assistant": "typed-wrong user and ts", "events": "nope"}),
        json.dumps({"ts": "2026-07-07T04:00:00+00:00", "role": "exchange",
                    "user": "events list holds junk", "assistant": "x",
                    "events": [42, None, {"type": "delegation", "tool": "web_search"}]}),
        json.dumps({"ts": "2026-07-07T05:00:00+00:00", "role": "exchange",
                    "user": "last good turn", "assistant": "ok", "events": []}),
    ]
    day_file = write_day_file(tmp_path, hostile)
    extractor = FakeExtractor([make_fact()])

    await digest(day_file, extractor, out_dir=tmp_path / "d")

    [exchanges] = extractor.calls
    users = [e["user"] for e in exchanges]
    assert users == [
        "first good turn",
        "null assistant",
        "",  # typed-wrong user coerced empty; the assistant text was kept
        "events list holds junk",
        "last good turn",
    ]
    assert exchanges[3]["tools"] == ["web_search"]  # junk event elements skipped


async def test_digest_writes_are_atomic_no_tmp_left_behind(tmp_path):
    # write-sibling + os.replace: a crash or concurrent writer can never leave
    # a TORN digest (which would silently drop the day from every merge).
    day_file = write_day_file(tmp_path, DAY_RECORDS)
    out_dir = tmp_path / "digests"

    await digest(day_file, FakeExtractor([make_fact()]), out_dir=out_dir)
    # A shrinking re-roll exercises the rejected-file write path too.
    await digest(day_file, FakeExtractor([]), force=True, out_dir=out_dir)

    leftovers = [p.name for p in out_dir.iterdir() if p.suffix == ".tmp"]
    assert leftovers == []
    json.loads((out_dir / "digest_2026-07-07.json").read_text())
    json.loads((out_dir / "digest_2026-07-07.rejected.json").read_text())


async def test_digest_missing_day_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        await digest(
            tmp_path / "events_2026-01-01.jsonl", FakeExtractor(), out_dir=tmp_path / "d"
        )


def test_digest_path_derives_from_date():
    assert digest_path("2026-07-07").name == "digest_2026-07-07.json"


# --- digest_all: idempotency is cache-file existence, not a progress cursor ----


async def test_digest_all_digests_missing_days_and_skips_existing(tmp_path):
    events = tmp_path / "events"
    events.mkdir()
    day6 = write_day_file(events, DAY_RECORDS, name="events_2026-07-06.jsonl")
    write_day_file(events, DAY_RECORDS, name="events_2026-07-07.jsonl")
    out_dir = tmp_path / "digests"
    await digest(day6, FakeExtractor([make_fact()]), out_dir=out_dir)  # 06 already done

    extractor = FakeExtractor([make_fact()])
    results = [r async for r in digest_all(extractor, events_dir=events, out_dir=out_dir)]

    assert results[0] == ("2026-07-06", None)  # skipped: its digest file exists
    assert results[1][0] == "2026-07-07" and len(results[1][1].facts) == 1
    assert len(extractor.calls) == 1  # only the missing day was extracted
    assert (out_dir / "digest_2026-07-07.json").exists()


async def test_digest_all_rerun_skips_everything(tmp_path):
    events = tmp_path / "events"
    events.mkdir()
    write_day_file(events, DAY_RECORDS, name="events_2026-07-06.jsonl")
    write_day_file(events, DAY_RECORDS, name="events_2026-07-07.jsonl")
    out_dir = tmp_path / "digests"
    extractor = FakeExtractor([make_fact()])
    [r async for r in digest_all(extractor, events_dir=events, out_dir=out_dir)]

    rerun = [r async for r in digest_all(extractor, events_dir=events, out_dir=out_dir)]

    assert [day for _, day in rerun] == [None, None]
    assert len(extractor.calls) == 2  # nothing re-extracted on the rerun


async def test_digest_all_force_reextracts_despite_caches(tmp_path):
    events = tmp_path / "events"
    events.mkdir()
    write_day_file(events, DAY_RECORDS, name="events_2026-07-06.jsonl")
    out_dir = tmp_path / "digests"
    extractor = FakeExtractor([make_fact()])
    [r async for r in digest_all(extractor, events_dir=events, out_dir=out_dir)]

    forced = [r async for r in digest_all(extractor, events_dir=events, out_dir=out_dir, force=True)]

    assert forced[0][0] == "2026-07-06" and forced[0][1] is not None
    assert len(extractor.calls) == 2


async def test_digest_all_ignores_stray_files(tmp_path):
    events = tmp_path / "events"
    events.mkdir()
    (events / "events_notes.jsonl").write_text("not a day file\n", encoding="utf-8")
    write_day_file(events, DAY_RECORDS, name="events_2026-07-07.jsonl")

    results = [
        r async for r in digest_all(FakeExtractor([make_fact()]), events_dir=events, out_dir=tmp_path / "d")
    ]

    assert [date for date, _ in results] == ["2026-07-07"]


# --- source grounding: assistant-side trust comes from the event log -----------

GROUNDING_RECORDS = [
    json.dumps(
        {
            "ts": "2026-07-06T20:00:00+00:00",
            "role": "exchange",
            "user": "Where is USA vs Belgium?",
            "assistant": "Seattle, WA.",
            "events": [{"type": "delegation", "tool": "web_search"}],
        }
    ),
    json.dumps(
        {
            "ts": "2026-07-06T21:00:00+00:00",
            "role": "exchange",
            "user": "And kickoff time?",
            "assistant": "8 PM ET, I believe.",
            "events": [],
        }
    ),
]


async def test_digest_grounds_assistant_trust_in_delegation_events(tmp_path):
    day_file = write_day_file(tmp_path, GROUNDING_RECORDS, name="events_2026-07-06.jsonl")
    extractor = FakeExtractor(
        [
            # Floored/misjudged on a tool-backed turn -> upgraded by the log.
            make_fact(
                subject="venue", source="assistant_claimed", turn_ts="2026-07-06T20:00:00+00:00"
            ),
            # Model claimed tool backing on a turn that used none -> downgraded.
            make_fact(
                subject="kickoff", source="tool_derived", turn_ts="2026-07-06T21:00:00+00:00"
            ),
            # user_asserted is the model's semantic call — grounding never touches it.
            make_fact(
                subject="user_home_location",
                source="user_asserted",
                turn_ts="2026-07-06T20:00:00+00:00",
            ),
            # ts copied without tz (observed drift) still matches its turn.
            make_fact(
                subject="venue_short_ts",
                source="assistant_claimed",
                turn_ts="2026-07-06T20:00:00",
            ),
            # A ts matching no turn: keep the floored source rather than guess.
            make_fact(
                subject="mangled", source="assistant_claimed", turn_ts="not-a-timestamp"
            ),
        ]
    )

    result = await digest(day_file, extractor, out_dir=tmp_path / "digests")

    by_subject = {f.subject: f.source for f in result.facts}
    assert by_subject == {
        "venue": "tool_derived",
        "kickoff": "assistant_claimed",
        "user_home_location": "user_asserted",
        "venue_short_ts": "tool_derived",
        "mangled": "assistant_claimed",
    }


# --- conflict linking ---------------------------------------------------------


def test_unlinked_same_subject_disagreement_gets_one_group():
    a = make_fact(subject="match_result", fact="Argentina won 3-2.")
    b = make_fact(subject="match_result", fact="Argentina won 3-1.")

    linked = _link_conflicts([a, b])

    assert linked[0].conflict_group is not None
    assert linked[0].conflict_group == linked[1].conflict_group


def test_extractor_assigned_ids_are_rebuilt_deterministically():
    # Chunked extraction reuses ids like "c1" across chunks for UNRELATED
    # conflicts — model ids are hints for emitting both sides, never kept.
    a = make_fact(subject="match_result", fact="Argentina won 3-2.", conflict_group="c1")
    b = make_fact(subject="match_result", fact="Argentina won 3-1.")
    other = make_fact(subject="final_venue", fact="MetLife Stadium.", conflict_group="c1")
    other2 = make_fact(subject="final_venue", fact="Azteca Stadium.")

    linked = _link_conflicts([a, b, other, other2])

    assert [f.conflict_group for f in linked] == [
        "conflict:match_result",
        "conflict:match_result",
        "conflict:final_venue",
        "conflict:final_venue",
    ]


def test_agreeing_and_unrelated_facts_stay_unlinked():
    dupe_a = make_fact(subject="match_result", fact="Argentina won 3-2.")
    # Trailing-period/case wording drift is agreement, not conflict; a spurious
    # model-assigned group on an agreeing fact is cleared.
    dupe_b = make_fact(subject="match_result", fact="argentina won 3-2", conflict_group="c9")
    other = make_fact(subject="user_home_location", conflict_group="c1")

    linked = _link_conflicts([dupe_a, dupe_b, other])

    assert all(f.conflict_group is None for f in linked)
