"""
Terminal chat surface — presentation only.

No logic lives here that isn't reachable elsewhere: this loop reads input, consumes
the orchestrator's structured events, and decides how to display them. It prints
`token` events live, surfaces `error` events, and ignores event types it does not
render yet (`thinking`, `delegation`).

prompt_toolkit's PromptSession enables bracketed paste, so a multi-line paste
arrives as ONE message and the prompt never prints mid-paste (CLAUDE.md gotcha).
"""
from __future__ import annotations

from prompt_toolkit import PromptSession

from core.constants import (
    ASSISTANT_DISPLAY_NAME,
    CLI_PROMPT,
    EXIT_COMMANDS,
    GOODBYE_MESSAGE,
    MODEL_UNAVAILABLE_NEXT_STEP,
    OLLAMA_READY_MSG,
    OLLAMA_STARTING_MSG,
    STAGE_DAEMON_FAILED,
    STAGE_DAEMON_READY,
    STAGE_MODEL_MISSING,
    STAGE_MODEL_READY,
    STAGE_NOT_INSTALLED,
    STAGE_PULL_DECLINED,
    STAGE_PULL_FAILED,
    STAGE_PULLING_MODEL,
    STAGE_STARTING_DAEMON,
    STAGE_WARMING,
    STAGE_WARMUP_FAILED,
    STAGE_WARMUP_READY,
)
from core.runtime.ollama_manager import BootEvent


class BootRenderer:
    """Renders structured BootEvents as terminal lines — the ONLY place boot wording
    lives (FIX C). The manager reports stages as data; this decides how they look, so
    a future frontend can consume the same events and draw spinners/pills instead.

    Holds just presentation state: whether an in-place progress line is open (so the
    next non-progress line starts cleanly).
    """

    def __init__(self) -> None:
        self._progress_open = False

    def _close_progress(self) -> None:
        if self._progress_open:
            print()  # terminate the in-place `\r` progress line
            self._progress_open = False

    def __call__(self, ev: BootEvent) -> None:
        stage = ev.stage
        if stage == STAGE_STARTING_DAEMON:
            print(OLLAMA_STARTING_MSG)
        elif stage == STAGE_DAEMON_READY:
            print(OLLAMA_READY_MSG)
        elif stage == STAGE_NOT_INSTALLED:
            print(ev.detail)
        elif stage == STAGE_DAEMON_FAILED:
            print(f"{ev.detail}. Start Ollama (or run ./setup.sh), then start Jarvis again.")
        elif stage == STAGE_MODEL_MISSING:
            print(f"{ev.detail}.")
        elif stage == STAGE_PULLING_MODEL:
            pct = f" {ev.progress:5.1f}%" if ev.progress is not None else ""
            print(f"\r  pulling: {ev.detail}{pct}", end="", flush=True)
            self._progress_open = True
        elif stage == STAGE_PULL_FAILED:
            self._close_progress()
            print(f"Pull failed: {ev.detail}")
            print(MODEL_UNAVAILABLE_NEXT_STEP.format(model=ev.model))
        elif stage == STAGE_PULL_DECLINED:
            print("Skipped.")
            print(MODEL_UNAVAILABLE_NEXT_STEP.format(model=ev.model))
        elif stage == STAGE_MODEL_READY:
            self._close_progress()
            if ev.detail:  # e.g. "<model> pulled"; silent when it was already present
                print(ev.detail)
        elif stage == STAGE_WARMING:
            print(f"Warming {ev.model}… ", end="", flush=True)
        elif stage == STAGE_WARMUP_READY:
            print(f"ready ({ev.elapsed_s:.0f}s)")
        elif stage == STAGE_WARMUP_FAILED:
            print(f"warmup failed: {ev.detail}")


def confirm_pull(model: str) -> bool:
    """Ask whether to pull the missing model. Interaction/presentation only — the
    boot coordinator decides what the answer means."""
    answer = input(f"Pull {model} now? [Y/n] ").strip().lower()
    return answer in ("", "y", "yes")


async def run_chat(orchestrator, session: PromptSession | None = None) -> None:
    # Default to a real session; an injected one lets tests drive input/output.
    session = session or PromptSession()

    while True:
        try:
            user_text = await session.prompt_async(CLI_PROMPT)
        except (EOFError, KeyboardInterrupt):
            print(f"\n{GOODBYE_MESSAGE}")
            return

        user_text = user_text.strip()
        if not user_text:
            continue
        if user_text.lower() in EXIT_COMMANDS:
            print(GOODBYE_MESSAGE)
            return

        # Assistant prefix prints once, before any streamed output for this turn.
        print(f"{ASSISTANT_DISPLAY_NAME}: ", end="", flush=True)

        async for event in orchestrator.respond(user_text):
            kind = event["type"]
            if kind == "token":
                print(event["content"], end="", flush=True)
            elif kind == "error":
                print(f"\n[error] {event['message']}", flush=True)
            # "thinking" / "delegation" / "done" carry no text to render yet.

        print()  # close the turn's line
