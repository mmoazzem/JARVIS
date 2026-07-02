"""System-prompt assembly invariants (core/orchestrator/personality.py)."""
from core.constants import NO_THINK_DIRECTIVE
from core.orchestrator.personality import build_system_prompt

IDENTITY_TEXT = "You are Jarvis, a test persona.\nComposed. Precise."
IDENTITY = {"identity": IDENTITY_TEXT}
STATE = {"Current date and time": "2026-07-02 10:00", "Mode": "local"}
PROFILE = "The user prefers concise answers."


def test_identity_text_is_present_verbatim():
    prompt = build_system_prompt(IDENTITY, STATE)

    # Verbatim — not paraphrased, not stripped of internal newlines.
    assert IDENTITY_TEXT in prompt


def test_three_layers_compose_in_order():
    prompt = build_system_prompt(IDENTITY, STATE, profile=PROFILE)

    identity_at = prompt.index(IDENTITY_TEXT)
    state_at = prompt.index("Current system state")
    profile_at = prompt.index(PROFILE)
    assert identity_at < state_at < profile_at


def test_state_layer_renders_each_entry():
    prompt = build_system_prompt(IDENTITY, STATE)

    assert "- Current date and time: 2026-07-02 10:00" in prompt
    assert "- Mode: local" in prompt


def test_falsy_state_values_are_omitted():
    prompt = build_system_prompt(IDENTITY, {"Mode": "", "GPU": None})

    assert "Mode" not in prompt
    assert "Current system state" not in prompt  # no entries → no layer at all


def test_no_think_appended_when_thinking_disabled():
    prompt = build_system_prompt(IDENTITY, STATE, enable_thinking=False)

    assert prompt.endswith(NO_THINK_DIRECTIVE)


def test_no_think_absent_when_thinking_enabled():
    prompt = build_system_prompt(IDENTITY, STATE, enable_thinking=True)

    assert NO_THINK_DIRECTIVE not in prompt


def test_empty_profile_slot_leaves_no_blank_layer():
    with_profile = build_system_prompt(IDENTITY, STATE, profile=PROFILE)
    empty = build_system_prompt(IDENTITY, STATE, profile="")
    whitespace = build_system_prompt(IDENTITY, STATE, profile="   \n ")

    assert PROFILE in with_profile
    # Empty/whitespace profile contributes nothing — no dangling separator.
    assert empty == whitespace
    assert "\n\n\n" not in empty
    assert not empty.endswith("\n\n")


def test_missing_identity_key_yields_no_crash_and_no_empty_layer():
    prompt = build_system_prompt({}, STATE)

    assert prompt.startswith("Current system state")
