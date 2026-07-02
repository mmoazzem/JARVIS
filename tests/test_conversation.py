"""Conversation trimming invariants (core/orchestrator/conversation.py).

Budgets below are in TOKENS; the estimator is len(text) // CHARS_PER_TOKEN, so a
message of N * CHARS_PER_TOKEN characters costs exactly N tokens — every test
builds messages with exact, known token costs.
"""
from core.constants import CHARS_PER_TOKEN, ROLE_ASSISTANT, ROLE_SYSTEM, ROLE_USER
from core.orchestrator.conversation import Conversation


def _text(tokens: int) -> str:
    """A string that costs exactly `tokens` under the estimator."""
    return "x" * (tokens * CHARS_PER_TOKEN)


def _conversation(budget: int, pairs: int, tokens_each: int) -> Conversation:
    """A conversation of `pairs` completed exchanges, each message `tokens_each`."""
    conv = Conversation(context_token_budget=budget)
    for i in range(pairs):
        conv.add_user(_text(tokens_each))
        conv.add_assistant(_text(tokens_each))
    return conv


def test_under_budget_history_is_untouched():
    conv = _conversation(budget=1000, pairs=3, tokens_each=10)
    conv.add_user(_text(10))  # current turn

    messages = conv.to_messages(system_prompt=_text(10))

    # system + 3 full pairs + current user turn, nothing dropped
    assert len(messages) == 8
    assert messages[0]["role"] == ROLE_SYSTEM


def test_exactly_at_budget_is_untouched():
    # system(10) + 2 pairs of 10+10 + current(10) = 60 tokens, budget exactly 60
    conv = _conversation(budget=60, pairs=2, tokens_each=10)
    conv.add_user(_text(10))

    messages = conv.to_messages(system_prompt=_text(10))

    assert len(messages) == 6  # at-budget is inside budget — no trim


def test_one_token_over_budget_trims_exactly_one_pair():
    # Same shape as the at-budget case but budget one token short: 60 needed, 59 given.
    conv = _conversation(budget=59, pairs=2, tokens_each=10)
    conv.add_user(_text(10))

    messages = conv.to_messages(system_prompt=_text(10))

    # One (and only one) oldest pair dropped: system + 1 pair + current turn.
    assert len(messages) == 4


def test_trimming_drops_oldest_pairs_first():
    conv = Conversation(context_token_budget=35)
    conv.add_user("old question".ljust(40, "."))
    conv.add_assistant("old answer".ljust(40, "."))
    conv.add_user("recent question".ljust(40, "."))
    conv.add_assistant("recent answer".ljust(40, "."))
    conv.add_user("current question".ljust(40, "."))

    messages = conv.to_messages(system_prompt=_text(5))

    contents = [m["content"] for m in messages]
    assert not any(c.startswith("old") for c in contents)
    assert any(c.startswith("recent question") for c in contents)
    assert contents[-1].startswith("current question")


def test_trimming_never_drops_system_prompt():
    # System prompt ALONE blows the budget — it must still be message[0].
    conv = _conversation(budget=10, pairs=3, tokens_each=10)
    conv.add_user(_text(10))
    system = _text(50)

    messages = conv.to_messages(system_prompt=system)

    assert messages[0] == {"role": ROLE_SYSTEM, "content": system}


def test_trimming_never_splits_a_user_assistant_pair():
    conv = _conversation(budget=45, pairs=3, tokens_each=10)
    conv.add_user(_text(10))

    messages = conv.to_messages(system_prompt=_text(5))
    history = messages[1:]

    # Whatever survived, it must be whole pairs plus the current user turn: the
    # first surviving message is a USER message (never an orphaned assistant reply)
    # and roles alternate user/assistant all the way down.
    assert history[0]["role"] == ROLE_USER
    for i, msg in enumerate(history):
        expected = ROLE_USER if i % 2 == 0 else ROLE_ASSISTANT
        assert msg["role"] == expected


def test_current_user_turn_survives_even_when_over_budget_alone():
    conv = Conversation(context_token_budget=5)
    conv.add_user(_text(100))  # the turn being asked right now

    messages = conv.to_messages(system_prompt=_text(5))

    assert messages[-1] == {"role": ROLE_USER, "content": _text(100)}


def test_to_messages_does_not_mutate_stored_history():
    conv = _conversation(budget=25, pairs=3, tokens_each=10)
    conv.add_user(_text(10))

    trimmed = conv.to_messages(system_prompt=_text(5))
    assert len(trimmed) < 8  # trimming happened in the returned view...

    # ...but the stored history is intact: a bigger budget later sees everything.
    conv._budget = 10_000
    assert len(conv.to_messages(system_prompt=_text(5))) == 8
