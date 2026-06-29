"""
Conversation history as structured data — pure logic, no I/O.

Holds the user/assistant turn sequence and assembles the messages array the model
consumes. Trimming keeps the array inside the configured token budget so long
sessions never overflow context, while honouring two invariants: the system prompt
is never dropped, and a user/assistant exchange is never split.
"""
from __future__ import annotations

from core.constants import CHARS_PER_TOKEN, ROLE_ASSISTANT, ROLE_SYSTEM, ROLE_USER


def _estimate_tokens(text: str) -> int:
    """Cheap token estimate (no tokenizer dependency) — see CHARS_PER_TOKEN."""
    return len(text) // CHARS_PER_TOKEN


class Conversation:
    def __init__(self, context_token_budget: int) -> None:
        self._budget = context_token_budget
        # Flat sequence of {role, content}; pairs are (user, assistant) in order.
        self._history: list[dict] = []

    def add_user(self, text: str) -> None:
        self._history.append({"role": ROLE_USER, "content": text})

    def add_assistant(self, text: str) -> None:
        self._history.append({"role": ROLE_ASSISTANT, "content": text})

    def to_messages(self, system_prompt: str) -> list[dict]:
        """Return [system] + trimmed history, within the token budget.

        Trims OLDEST exchanges first, two messages at a time so a user/assistant
        pair is never split, and always keeps the system prompt plus the most
        recent (current) user turn.
        """
        history = list(self._history)
        system_tokens = _estimate_tokens(system_prompt)

        # Drop oldest pairs while over budget, but never strip the final turn.
        while len(history) > 1 and self._total_tokens(system_tokens, history) > self._budget:
            del history[:2]

        return [{"role": ROLE_SYSTEM, "content": system_prompt}, *history]

    @staticmethod
    def _total_tokens(system_tokens: int, history: list[dict]) -> int:
        return system_tokens + sum(_estimate_tokens(m["content"]) for m in history)
