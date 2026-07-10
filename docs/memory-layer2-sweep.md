# Memory Layer 2 — adversarial sweep report (2026-07-09)

Goal: zero UNKNOWN bugs in the digest → merge → profile → live-recall pipeline.
Every input below was thrown at the real system (qwen3:14b live where extraction
semantics matter; synthetic day-files in temp dirs — real logs untouched).
Verdicts: **fixed** (was a crash/corruption, now has a regression test),
**known limitation** (graceful but imperfect, deliberately deferred), **fine**.

| # | Input | Observed behavior | Verdict |
|---|-------|-------------------|---------|
| 1 | Empty event log (0 bytes; also only non-exchange records) | 0 facts, **0 model calls** (extractor short-circuits), merge contribution empty | fine |
| 2 | Only-ephemeral day (time + weather queries) | 8 facts extracted, all `current_state`/`weather_lookup`, **0 leaked into profile** | fine |
| 3 | Malformed JSONL mid-file (torn line, non-object JSON like `42`, `null` assistant, wrong-typed fields, junk in `events`) | WAS: a non-object line or null field **crashed the whole day** (`AttributeError`/`TypeError` in `_read_exchanges`). NOW: each malformed line/field costs at most its own record; usable halves kept | **fixed** (`test_malformed_records_lose_at_most_themselves_never_the_day`) |
| 4 | One message asserting 3+ facts (peanut allergy + sister Nadia + job at Moog) | All three captured as `user_asserted` (multi-pass union; Part 1b) | fine |
| 5 | Same day digested concurrently / twice rapidly | Concurrent: both extract (double model cost), final file always valid JSON; rapid double: second run served from cache, 0 extractions. WAS: `write_text` truncate-then-write could leave a **torn cache** on crash/concurrent write (day silently dropped from merges until re-digest). NOW: all digest/profile writes are atomic (`os.replace`) | **fixed** (atomic writes) + known limitation: no cross-process lock, so two processes forcing the same day duplicate LLM cost — harmless, last writer wins |
| 6 | Very large day (200 exchanges, facts planted at exchanges 5/100/195) | No crash, no timeout, 0 failed chunks (29 chunks); **all 3 planted facts recalled** incl. mid-transcript; 226 facts in 816s at 1 pass (~41 min at default `digest_passes: 3`) | fine + 2 known limitations: (a) cost is linear in exchanges × passes — acceptable for on-demand `/digest`; (b) trivia-answer flood: every one-shot assistant answer ("47 is even") extracts as durable `world_fact`, so a chatty day bloats the profile with low-value facts (same family as the real profile's `2+2=4`; retrieval-side relevance filtering, deferred to RAG) |
| 7 | Same subject, ascending trust across days (assistant d1 → tool d2 → user d3) | Storage keeps all three conflict-linked; working view = day-3 user value | fine (`test_trust_escalation_across_days_resolves_to_newest_user_value`) |
| 8 | Unicode / emoji / very long fact | 🐱ミケ and crème brûlée survive extraction → storage → render; 10k-char fact round-trips. WAS: `render_profile` had no length cap — a huge fact landed in the system prompt verbatim. NOW: each fact is capped at `PROFILE_FACT_RENDER_MAX` (500 chars + ellipsis) at RENDER only; profile.json keeps the full fact | **fixed** (`test_unicode_and_huge_facts_survive_storage_and_render`) |
| 9 | `.rejected.json` + stray files in `data/digests/` (`digest_notes.json`, `notes.txt`, corrupt digest) | All skipped; corrupt digest warns and is skipped | fine (`test_load_day_digests_skips_rejected_unreadable_and_strays`) |
| 10 | `profile.json` missing / corrupt at boot | Warning logged, orchestrator boots with empty memory; live answer honestly says it doesn't know (no hallucinated recall); missing profile also boots; `/merge` rebuilds | fine (`test_profile_roundtrip_missing_and_corrupt`) |

## Known limitations (documented, deferred by decision)

- **Subject near-duplicate drift** (`FIFA_2026_final_date` vs
  `2026_FIFA_World_Cup_final_date`) — parallel profile entries per wording;
  multi-pass union amplifies the fact COUNT inflation. Deferred to RAG
  (`# KNOWN, deferred to RAG` in `merge.py`); recall is unaffected and the
  working view still renders one line per subject string.
- **Category instability across runs** (same fact, different on-enum category) —
  deferred to RAG (`# KNOWN` in `llm_digest.py`); the assistant-self-report drop
  rule deliberately keys on subject+source, not category, to be immune to it.
- **No cross-process digest lock** (item 5, above). (Item 8's render length cap
  was subsequently fixed: `PROFILE_FACT_RENDER_MAX`.)
- **Large-day cost scaling** and **trivia-answer profile bloat** (item 6, above).
  Bloat is the one to watch: it degrades the Layer-3 prompt (every durable fact
  renders into the system prompt today), and RAG-style retrieval is the planned
  fix — until then, a noisy day is cheap to re-filter because raw digests keep
  everything and merge re-runs are free.
- Facts extracted from wrong assistant claims are stored as claimed
  (garbage in, garbage out) — by design: trust ranking (`user_asserted` >
  `tool_derived` > `assistant_claimed`) is the correction mechanism, verified
  live in item 7 and by the ARG–EGT 3-2 recall check.

