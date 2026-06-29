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
)


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
