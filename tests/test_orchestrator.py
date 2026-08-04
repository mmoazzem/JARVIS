"""
Orchestrator.respond() owns memory-Layer-1 capture: the event log rides the
core stream, so EVERY interface (CLI, WebSocket, future frontend) logs a turn
identically without wiring anything. These tests pin that contract — if
capture ever moves back into an interface loop, browser turns silently stop
becoming digestible memory.
"""
import gc
import json

import pytest

from core.memory.event_log import EventLog
from core.orchestrator.orchestrator import Orchestrator
from setup.config import JarvisConfig


@pytest.fixture
def orchestrator(tmp_path):
    # tools_enabled=False keeps construction to model + agent; the event log
    # is redirected at a tmp dir so tests never touch the real events files.
    orch = Orchestrator(JarvisConfig(tools_enabled=False))
    orch._event_log = EventLog(enabled=True, log_dir=tmp_path)

    async def fake_stream(user_text, conversation=None):
        yield {"type": "thinking"}
        yield {"type": "token", "content": "Buffalo, "}
        yield {"type": "token", "content": "NY."}
        yield {"type": "done"}

    orch._agent.respond = fake_stream
    return orch


async def _drain(stream):
    return [event async for event in stream]


@pytest.mark.asyncio
async def test_respond_captures_the_turn_without_any_interface(tmp_path, orchestrator):
    # Any bare consumer of respond() — this IS the WebSocket server's shape.
    events = await _drain(orchestrator.respond("which city do I live in?"))

    assert [e["type"] for e in events] == ["thinking", "token", "token", "done"]
    [day_file] = list(tmp_path.glob("events_*.jsonl"))
    record = json.loads(day_file.read_text(encoding="utf-8"))
    assert record["user"] == "which city do I live in?"
    assert record["assistant"] == "Buffalo, NY."


@pytest.mark.asyncio
async def test_two_turns_append_two_records_same_file(tmp_path, orchestrator):
    await _drain(orchestrator.respond("first"))
    await _drain(orchestrator.respond("second"))

    [day_file] = list(tmp_path.glob("events_*.jsonl"))
    users = [json.loads(line)["user"] for line in day_file.read_text().splitlines()]
    assert users == ["first", "second"]


# --- per-turn buffers ---------------------------------------------------------
# The turn buffer used to live on the EventLog, so two turns in flight shared it
# and one wrongly-paired exchange reached disk. These three pin the fix; the
# format assertion is a literal captured from the code BEFORE the change.

# Exactly what the pre-handle implementation wrote, ts masked.
PRE_CHANGE_RECORD = (
    '{"ts": "<TS>", "role": "exchange", "user": "which city do I live in?", '
    '"assistant": "Buffalo, NY.", "events": []}'
)


@pytest.mark.asyncio
async def test_concurrent_turns_write_two_correctly_paired_records(tmp_path, orchestrator):
    # The ALPHA/BRAVO shape: both turns open at once, each echoing its own word.
    async def echo(user_text, conversation=None):
        yield {"type": "thinking"}
        yield {"type": "token", "content": user_text.split()[-1]}
        yield {"type": "done"}

    orchestrator._agent.respond = echo
    alpha = orchestrator.respond("reply with ALPHA")
    bravo = orchestrator.respond("reply with BRAVO")
    await anext(alpha)  # both turns are now open — the corrupting condition
    await anext(bravo)
    await _drain(alpha)
    await _drain(bravo)

    [day_file] = list(tmp_path.glob("events_*.jsonl"))
    records = [json.loads(line) for line in day_file.read_text().splitlines()]
    assert {(r["user"], r["assistant"]) for r in records} == {
        ("reply with ALPHA", "ALPHA"),
        ("reply with BRAVO", "BRAVO"),
    }


@pytest.mark.asyncio
async def test_abandoned_turn_writes_no_record_and_strands_no_buffer(tmp_path, orchestrator):
    stream = orchestrator.respond("a question the client will not wait for")
    await anext(stream)  # thinking
    await anext(stream)  # one token has reached the client
    await stream.aclose()  # ...and the client vanishes
    gc.collect()

    # A half-assembled exchange must not land on disk looking like a whole one.
    assert list(tmp_path.iterdir()) == []
    # Nor may its buffer outlive it inside the log (private on purpose: there is
    # no public way to observe a leak that has no other symptom).
    assert len(orchestrator.event_log._open) == 0


@pytest.mark.asyncio
async def test_single_turn_record_is_byte_identical_to_the_pre_change_format(
    tmp_path, orchestrator
):
    # The CLI path: one turn, start to finish. The digest reads these files, so
    # the bytes must not have moved.
    await _drain(orchestrator.respond("which city do I live in?"))

    [day_file] = list(tmp_path.glob("events_*.jsonl"))
    line = day_file.read_text(encoding="utf-8").rstrip("\n")
    assert line.replace(json.loads(line)["ts"], "<TS>") == PRE_CHANGE_RECORD


def test_new_conversation_hands_out_private_histories(orchestrator):
    # Per-connection state without a second Orchestrator: the expensive parts
    # (model, tools, warmup) stay shared, only the past is handed out fresh.
    first, second = orchestrator.new_conversation(), orchestrator.new_conversation()

    assert first is not second
    first.add_user("my codeword is PELICAN")
    assert "PELICAN" not in str(second.to_messages("system"))


@pytest.mark.asyncio
async def test_disabled_event_log_streams_but_writes_nothing(tmp_path, orchestrator):
    orchestrator._event_log = EventLog(enabled=False, log_dir=tmp_path)

    events = await _drain(orchestrator.respond("hello"))

    assert any(e["type"] == "token" for e in events)  # the stream itself is untouched
    assert list(tmp_path.iterdir()) == []
