"""
Orchestrator.respond() owns memory-Layer-1 capture: the event log rides the
core stream, so EVERY interface (CLI, WebSocket, future frontend) logs a turn
identically without wiring anything. These tests pin that contract — if
capture ever moves back into an interface loop, browser turns silently stop
becoming digestible memory.
"""
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

    async def fake_stream(user_text):
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


@pytest.mark.asyncio
async def test_disabled_event_log_streams_but_writes_nothing(tmp_path, orchestrator):
    orchestrator._event_log = EventLog(enabled=False, log_dir=tmp_path)

    events = await _drain(orchestrator.respond("hello"))

    assert any(e["type"] == "token" for e in events)  # the stream itself is untouched
    assert list(tmp_path.iterdir()) == []
