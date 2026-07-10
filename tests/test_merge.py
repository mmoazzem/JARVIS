"""Memory Layer-2 merge invariants (core/memory/merge.py) — pure logic, no model.

Pinned behaviors:
  * Ephemeral categories are dropped at MERGE-READ only; durable ones survive
    with their turn_ts + source provenance intact.
  * Override keys on `subject`: identical values collapse keeping the
    strongest provenance; distinct values are ALL kept, conflict-linked,
    never resolved (Layer-3 seam).
  * working_view picks ONE fact per subject — highest trust, newest at equal
    trust — and render_profile emits one line per subject.
  * load_day_digests skips .rejected.json dumps and unreadable files.
"""
import json
from pathlib import Path

from core.constants import EPHEMERAL_CATEGORIES, FACT_CATEGORIES, PROFILE_PATH
from core.memory.base_digest import DayDigest, FactRecord
from core.memory.merge import (
    Profile,
    load_day_digests,
    load_profile,
    merge,
    save_profile,
    working_view,
)
from core.orchestrator.personality import render_profile


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


def make_day(date: str, facts) -> DayDigest:
    return DayDigest(
        date=date,
        source_file=f"events_{date}.jsonl",
        extracted_at=f"{date}T23:59:59+00:00",
        extractor="fake",
        facts=list(facts),
    )


# --- the drop-list ------------------------------------------------------------


def test_ephemeral_categories_dropped_durable_retained():
    day = make_day(
        "2026-07-05",
        [make_fact(subject=f"s_{cat}", category=cat) for cat in FACT_CATEGORIES],
    )

    profile = merge([day])

    kept = {f.category for f in profile.facts}
    assert kept == set(FACT_CATEGORIES) - set(EPHEMERAL_CATEGORIES)
    assert not kept & set(EPHEMERAL_CATEGORIES)


def test_drop_list_starts_with_exactly_four_categories():
    # The tuning contract: widen only when real noise survives into the
    # profile — never pre-emptively (over-dropping is the failure mode).
    assert set(EPHEMERAL_CATEGORIES) == {
        "current_state",
        "weather_lookup",
        "reference_lookup",
        "puzzle_or_task",
    }


def test_assistant_self_reports_dropped_unless_user_asserted():
    # "The assistant is functioning well" arrives under DURABLE categories
    # (category drifts run to run), so the rule keys on subject + source.
    day = make_day(
        "2026-07-05",
        [
            make_fact(
                subject="assistant_functioning_status",
                fact="The assistant is functioning well",
                category="personal_fact",
                source="assistant_claimed",
            ),
            make_fact(
                subject="assistant_capabilities",
                fact="The assistant can search the web.",
                category="world_fact",
                source="tool_derived",
            ),
            # What the USER says about the assistant is memory, not self-report.
            make_fact(
                subject="assistant_nickname",
                fact="The user calls the assistant J.",
                category="user_preference",
                source="user_asserted",
            ),
            make_fact(),
        ],
    )

    profile = merge([day])

    assert {f.subject for f in profile.facts} == {
        "assistant_nickname",
        "user_home_location",
    }


def test_provenance_survives_merge():
    day = make_day("2026-07-07", [make_fact()])

    [fact] = merge([day]).facts

    assert fact.turn_ts == "2026-07-07T23:03:07+00:00"
    assert fact.source == "user_asserted"


# --- subject-keyed override ---------------------------------------------------


def test_agreeing_values_collapse_keeping_strongest_provenance():
    old = make_fact(
        fact="the user lives in Buffalo, NY",  # same value modulo case/period
        source="assistant_claimed",
        turn_ts="2026-07-05T10:00:00+00:00",
    )
    new = make_fact(source="user_asserted", turn_ts="2026-07-07T23:03:07+00:00")

    profile = merge([make_day("2026-07-05", [old]), make_day("2026-07-07", [new])])

    [fact] = profile.facts
    assert fact.source == "user_asserted"
    assert fact.conflict_group is None  # agreement is not conflict


def test_disagreeing_values_all_kept_and_conflict_linked():
    tool = make_fact(
        subject="match_venue", fact="Seattle, WA", source="tool_derived",
        turn_ts="2026-07-06T01:00:00+00:00",
    )
    user = make_fact(
        subject="match_venue", fact="Dallas, TX", source="user_asserted",
        turn_ts="2026-07-06T02:00:00+00:00",
    )

    profile = merge([make_day("2026-07-06", [tool, user])])

    assert len(profile.facts) == 2  # conflicts are preserved, never resolved
    assert {f.conflict_group for f in profile.facts} == {"conflict:match_venue"}


def test_working_view_picks_highest_trust_then_newest():
    facts = [
        make_fact(subject="venue", fact="Seattle", source="user_asserted",
                  turn_ts="2026-07-06T01:00:00+00:00"),
        make_fact(subject="venue", fact="Dallas", source="tool_derived",
                  turn_ts="2026-07-07T01:00:00+00:00"),  # newer but lower trust
        make_fact(subject="score", fact="3-1", source="tool_derived",
                  turn_ts="2026-07-06T01:00:00+00:00"),
        make_fact(subject="score", fact="3-2", source="tool_derived",
                  turn_ts="2026-07-07T01:00:00+00:00"),  # equal trust: newest wins
    ]
    profile = merge([make_day("2026-07-07", facts)])

    view = {f.subject: f.fact for f in working_view(profile)}

    assert view == {"venue": "Seattle", "score": "3-2"}
    assert len(profile.facts) == 4  # storage still holds every side


# --- render contract ------------------------------------------------------------


def test_render_profile_one_line_per_subject():
    facts = [
        make_fact(subject="user_home_location"),
        make_fact(subject="user_editor", fact="The user prefers vim.",
                  category="user_preference"),
    ]

    text = render_profile(facts)

    assert len([line for line in text.splitlines() if line.startswith("- ")]) == 2
    assert "user_home_location: The user lives in Buffalo, NY." in text
    assert render_profile([]) == ""  # no memory, no Layer-3 block


# --- storage ------------------------------------------------------------------


def test_load_day_digests_skips_rejected_unreadable_and_strays(tmp_path):
    good = make_day("2026-07-07", [make_fact()])
    (tmp_path / "digest_2026-07-07.json").write_text(good.model_dump_json())
    (tmp_path / "digest_2026-07-06.rejected.json").write_text(good.model_dump_json())
    (tmp_path / "digest_2026-07-05.json").write_text("{corrupt")
    (tmp_path / "digest_notes.json").write_text("a stray the glob sweeps in")
    (tmp_path / "notes.txt").write_text("a stray the glob never sees")

    digests = load_day_digests(tmp_path)

    assert [d.date for d in digests] == ["2026-07-07"]


def test_trust_escalation_across_days_resolves_to_newest_user_value():
    # Adversarial sweep shape: same subject, ascending trust day over day —
    # storage keeps all three conflict-linked; the working view shows only
    # the day-3 user assertion.
    days = [
        make_day("2026-07-01", [make_fact(subject="fav_team", fact="Boca",
                                          source="assistant_claimed",
                                          turn_ts="2026-07-01T01:00:00+00:00")]),
        make_day("2026-07-02", [make_fact(subject="fav_team", fact="River",
                                          source="tool_derived",
                                          turn_ts="2026-07-02T01:00:00+00:00")]),
        make_day("2026-07-03", [make_fact(subject="fav_team", fact="Racing",
                                          source="user_asserted",
                                          turn_ts="2026-07-03T01:00:00+00:00")]),
    ]

    profile = merge(days)

    assert {f.conflict_group for f in profile.facts} == {"conflict:fav_team"}
    assert len(profile.facts) == 3
    [view] = working_view(profile)
    assert (view.fact, view.source) == ("Racing", "user_asserted")


def test_profile_roundtrip_missing_and_corrupt(tmp_path):
    path = tmp_path / "profile.json"
    profile = merge([make_day("2026-07-07", [make_fact()])])

    save_profile(profile, path)

    assert load_profile(path) == profile
    assert not list(tmp_path.glob("*.tmp"))  # atomic write left no sibling
    assert load_profile(tmp_path / "absent.json") is None
    path.write_text('{"merged_at": 3, "facts": "corrupt"')
    assert load_profile(path) is None  # boot degrades to empty memory
    path.write_text("null")
    assert load_profile(path) is None


def test_unicode_and_huge_facts_survive_storage_and_render(tmp_path):
    emoji = make_fact(subject="user_cat_name", fact="The cat is named 🐱ミケ — Ω≈ç√∫")
    huge = make_fact(subject="user_novel", fact="A" * 10_000 + " 終")
    path = tmp_path / "profile.json"

    save_profile(merge([make_day("2026-07-07", [emoji, huge])]), path)
    stored = load_profile(path)
    text = render_profile(working_view(stored))

    assert "🐱ミケ" in text  # short fact renders verbatim, no ellipsis
    assert "…" not in text.split("user_cat_name: ")[1].splitlines()[0]
    # Huge fact: render is capped with an ellipsis, storage keeps it whole.
    [rendered_huge] = [l for l in text.splitlines() if l.startswith("- user_novel:")]
    assert rendered_huge.endswith("A" * 50 + "…")
    assert len(rendered_huge) < 600
    [stored_huge] = [f for f in stored.facts if f.subject == "user_novel"]
    assert stored_huge.fact == "A" * 10_000 + " 終"


def test_committed_example_profile_matches_schema():
    example = Path(__file__).resolve().parent.parent / "data" / "profile.example.json"
    profile = Profile.model_validate_json(example.read_text(encoding="utf-8"))
    assert profile.facts  # the seed shows at least one fact and one conflict pair
    assert any(f.conflict_group for f in profile.facts)
