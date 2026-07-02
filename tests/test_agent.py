"""Agent run-loop invariants (core/orchestrator/agent.py) — mocked model, no Ollama.

Two documented behaviors are pinned here:
  * respond() yields STRUCTURED EVENT DICTS (token/done/error/thinking), never raw
    strings — the envelope a future frontend consumes.
  * The zero-content gotcha: reasoning can eat the whole budget and leave no
    content. The agent must detect that, retry EXACTLY ONCE with the recovery
    instruction, and if still empty emit the honest fallback — never silence.
"""
from typing import AsyncIterator

from core.constants import FALLBACK_MESSAGE, RECOVERY_INSTRUCTION, ROLE_SYSTEM
from core.orchestrator.agent import Agent
from models.base import BaseModel, ModelResponse, WarmupResult
from setup.config import JarvisConfig

EVENT_TYPES = {"thinking", "token", "done", "error"}


class MockModel(BaseModel):
    """Scripted model: each stream() call plays the next token list (or raises)."""

    def __init__(self, scripts: list):
        self._scripts = list(scripts)
        self.calls: list[list[dict]] = []  # messages received, call by call

    async def stream(self, messages: list[dict], **opts) -> AsyncIterator[str]:
        self.calls.append(messages)
        script = self._scripts.pop(0) if self._scripts else []
        if isinstance(script, Exception):
            raise script
        for token in script:
            yield token

    async def complete(self, messages, **opts) -> ModelResponse:
        raise AssertionError("conversational turns must be streaming-only")

    async def health_check(self) -> bool:
        return True

    async def warmup(self) -> WarmupResult:
        return WarmupResult(success=True, model_id="mock", elapsed_s=0.0)

    async def unload(self) -> None:
        pass


def _agent(scripts: list) -> tuple[Agent, MockModel]:
    model = MockModel(scripts)
    config = JarvisConfig()  # defaults: enable_thinking=False, budgets as locked
    identity = {"identity": "You are a test persona."}
    return Agent(model, config, identity), model


async def _collect(agent: Agent, text: str = "hello") -> list[dict]:
    return [event async for event in agent.respond(text)]


# --- event shape --------------------------------------------------------------


async def test_respond_yields_only_structured_event_dicts():
    agent, _ = _agent([["Hel", "lo."]])

    events = await _collect(agent)

    assert events, "respond() yielded nothing"
    for event in events:
        assert isinstance(event, dict), f"raw value leaked: {event!r}"
        assert event["type"] in EVENT_TYPES


async def test_token_events_carry_the_streamed_content_in_order():
    agent, _ = _agent([["The ", "answer ", "is 42."]])

    events = await _collect(agent)

    tokens = [e["content"] for e in events if e["type"] == "token"]
    assert tokens == ["The ", "answer ", "is 42."]


async def test_successful_turn_ends_with_done():
    agent, _ = _agent([["ok"]])

    events = await _collect(agent)

    assert events[-1] == {"type": "done"}


async def test_model_exception_becomes_error_event_not_raise():
    agent, _ = _agent([RuntimeError("connection lost")])

    events = await _collect(agent)  # must not raise out of respond()

    assert events[-1]["type"] == "error"
    assert "connection lost" in events[-1]["message"]


async def test_model_receives_system_prompt_first():
    agent, model = _agent([["ok"]])

    await _collect(agent, "what time is it?")

    messages = model.calls[0]
    assert messages[0]["role"] == ROLE_SYSTEM
    assert "test persona" in messages[0]["content"]
    assert messages[-1]["content"] == "what time is it?"


# --- zero-content detection & recovery -----------------------------------------


async def test_zero_content_triggers_exactly_one_recovery():
    # First call: reasoning ate the budget (no content). Second call: recovers.
    agent, model = _agent([[], ["Recovered answer."]])

    events = await _collect(agent)

    assert len(model.calls) == 2  # exactly one retry, no retry-storm
    assert model.calls[1][-1]["content"] == RECOVERY_INSTRUCTION
    tokens = "".join(e["content"] for e in events if e["type"] == "token")
    assert tokens == "Recovered answer."
    assert events[-1] == {"type": "done"}


async def test_whitespace_only_content_counts_as_zero_content():
    agent, model = _agent([["  ", "\n"], ["Real answer."]])

    await _collect(agent)

    assert len(model.calls) == 2  # whitespace is not an answer — recovery ran


async def test_still_empty_after_recovery_emits_honest_fallback():
    agent, model = _agent([[], []])  # both attempts come back empty

    events = await _collect(agent)

    assert len(model.calls) == 2  # one recovery, then stop — never a third call
    tokens = [e["content"] for e in events if e["type"] == "token"]
    assert tokens == [FALLBACK_MESSAGE]  # visible fallback, not silence
    assert events[-1] == {"type": "done"}


async def test_turn_is_never_silently_empty():
    # Whatever the model does — answer, empty, empty twice — the user always sees
    # either content, the fallback message, or an error event.
    for scripts in ([["hi"]], [[], ["recovered"]], [[], []], [RuntimeError("boom")]):
        agent, _ = _agent(list(scripts))
        events = await _collect(agent)
        visible = [e for e in events if e["type"] in ("token", "error")]
        assert visible, f"silent turn for scripts {scripts!r}"


async def test_content_turn_does_not_trigger_recovery():
    agent, model = _agent([["fine."]])

    await _collect(agent)

    assert len(model.calls) == 1  # a good answer is never re-asked
