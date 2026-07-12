"""
Terminal chat surface — presentation only.

No logic lives here that isn't reachable elsewhere: this loop reads input, consumes
the orchestrator's structured events, and decides how to display them. It prints
`token` events live, renders `delegation` events as a status line above the
answer, surfaces `error` events, and ignores event types it does not render yet
(`thinking`, `recovery`, and the speech events).

One subscriber rides the event stream beside the renderer: the speech pipeline
(when voice is on). The event log is core-owned — Orchestrator.respond()
captures every turn, so the CLI only feeds it speech-interruption events.
With voice OFF the loop is the
original sequential prompt→stream→prompt — byte-identical to the pre-TTS app.
Voice ON renders through the SAME sequential path (printing never depends on
the TTS pipeline's sentence batching); it only adds a raw-mode key listener for
the duration of the turn, so Enter can stop speech at any moment — mid-stream
included. A live prompt is NOT kept open while tokens stream: prompt_toolkit
can only reprint complete lines above an active prompt, so a partial-line token
stream gets erased on every prompt redraw (the original "only the tail renders"
bug). Keys typed during a turn buffer unechoed: Enter submits them as the next
turn, leftovers prefill the next prompt.

prompt_toolkit's PromptSession enables bracketed paste, so a multi-line paste
arrives as ONE message and the prompt never prints mid-paste (CLAUDE.md gotcha).
"""
from __future__ import annotations

import asyncio
import logging

from prompt_toolkit import PromptSession
from prompt_toolkit.keys import Keys

from core.constants import (
    ASSISTANT_DISPLAY_NAME,
    CLI_PROMPT,
    DELEGATION_LINE_FORMAT,
    DIGEST_COMMAND,
    DIGEST_FLAG_ALL,
    DIGEST_FLAG_FORCE,
    EXIT_COMMANDS,
    GOODBYE_MESSAGE,
    LOGGER_ROOT,
    MERGE_COMMAND,
    PROFILE_PATH,
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
    VOICE_COMMAND,
)
from core.memory.base_digest import DayDigest
from core.memory.digest import digest_path
from core.runtime.ollama_manager import BootEvent
from interface.audio import AudioUnavailableError, PulsePlayer
from interface.speech import SpeechController
from models.tts import create_tts


def _digest_summary(day: DayDigest) -> str:
    """One line per digested day — the full fact list lives in the JSON file."""
    conflicts = {f.conflict_group for f in day.facts if f.conflict_group}
    return (
        f"Digested {day.date}: {len(day.facts)} facts, "
        f"{len(conflicts)} conflict group(s) → {digest_path(day.date)}"
    )

log = logging.getLogger(LOGGER_ROOT)


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


async def run_chat(orchestrator, config, session: PromptSession | None = None) -> None:
    # Default to a real session; an injected one lets tests drive input/output.
    session = session or PromptSession()

    # Turn capture moved into Orchestrator.respond() — the CLI only subscribes
    # its speech side-channel to the same core-owned log.
    event_log = orchestrator.event_log
    speech: SpeechController | None = None
    voice_on = False
    # Keystrokes swallowed by raw mode while a voiced turn streamed: completed
    # (Enter-terminated) inputs run as the next turns; the unfinished remainder
    # prefills the next prompt so nothing the user typed is lost.
    typed_keys: list[str] = []
    queued_turns: list[str] = []

    async def enable_voice() -> bool:
        """Build the speech pipeline on first use; report (not raise) if audio
        can't work — voice stays off and the app remains text-only."""
        nonlocal speech
        if speech is None:
            try:
                player = PulsePlayer()
                await asyncio.to_thread(player._lib)  # fail fast on a machine with no audio
                speech = SpeechController(
                    create_tts(config.tts_engine, config.tts_voice),
                    player,
                    on_event=event_log.feed_speech,
                    preroll_ms=config.tts_preroll_ms,
                )
            except (AudioUnavailableError, ValueError) as exc:
                print(f"Voice unavailable: {exc}")
                return False
        speech.ensure_started()
        return True

    async def run_turn(user_text: str) -> None:
        """One full turn: render events and feed the speech subscriber. The
        event log is NOT fed here — Orchestrator.respond() captures the turn.
        BOTH voice modes render through this path, so the printed text is EXACTLY
        what the pre-TTS loop printed — voice only adds audio beside it."""
        if voice_on and speech is not None:
            speech.begin_turn()

        # The assistant prefix prints once, ahead of the first answer text — but
        # AFTER any delegation status lines, which is why it waits for the first
        # token instead of printing up front. Non-tool turns render byte-identically.
        prefix_printed = False

        def ensure_prefix() -> None:
            nonlocal prefix_printed
            if not prefix_printed:
                print(f"{ASSISTANT_DISPLAY_NAME}: ", end="", flush=True)
                prefix_printed = True

        async for event in orchestrator.respond(user_text):
            kind = event["type"]
            if kind == "token":
                ensure_prefix()
                print(event["content"], end="", flush=True)
            elif kind == "delegation":
                print(DELEGATION_LINE_FORMAT.format(status=event["status"]), flush=True)
            elif kind == "error":
                ensure_prefix()
                print(f"\n[error] {event['message']}", flush=True)
            # "thinking" / "recovery" / "done" carry no text to render yet.
            if voice_on and speech is not None:
                speech.feed(event)

        ensure_prefix()
        print()  # close the turn's line

    async def run_turn_voiced(user_text: str) -> None:
        """Voice-on turn: rendering is run_turn unchanged — every token prints
        as it arrives, independent of the TTS layer's sentence batching. A
        raw-mode key listener rides along so Enter stops speech the instant it
        lands, even while text is still streaming; the turn's TEXT completes."""
        aborted = False
        turn = asyncio.ensure_future(run_turn(user_text))

        def on_keys() -> None:
            nonlocal aborted
            for press in key_input.read_keys():
                if press.key in (Keys.Enter, Keys.ControlJ):
                    if speech is not None:
                        speech.interrupt()  # Enter ALWAYS stops speech
                    text = "".join(typed_keys).strip()
                    typed_keys.clear()
                    if text:
                        queued_turns.append(text)
                elif press.key in (Keys.ControlC, Keys.ControlD):
                    aborted = True
                    turn.cancel()
                elif press.key == Keys.Backspace:
                    if typed_keys:
                        typed_keys.pop()
                elif press.key == Keys.BracketedPaste:
                    typed_keys.append(press.data)
                elif len(press.data) == 1 and press.data.isprintable():
                    typed_keys.append(press.data)

        key_input = session.app.input
        output = session.app.output
        output.enable_bracketed_paste()  # a mid-turn paste stays ONE message
        output.flush()
        try:
            with key_input.raw_mode(), key_input.attach(on_keys):
                await turn
        except asyncio.CancelledError:
            if not aborted:
                raise
        finally:
            output.disable_bracketed_paste()
            output.flush()
        if aborted:
            raise KeyboardInterrupt  # Ctrl+C mid-turn == Ctrl+C at the prompt

    async def shutdown(spoken_goodbye: bool) -> None:
        if speech is not None:
            if spoken_goodbye:
                try:
                    await speech.say(GOODBYE_MESSAGE)
                except Exception as exc:  # goodbye audio is best-effort
                    log.warning("goodbye speech failed: %s", exc)
            await speech.aclose()

    if config.tts_enabled:
        voice_on = await enable_voice()

    while True:
        if queued_turns:
            user_text = queued_turns.pop(0)
            print(f"{CLI_PROMPT}{user_text}")  # echo what raw mode swallowed
        else:
            try:
                if voice_on:
                    # Speech may still be playing (it outlives the turn's text);
                    # the prompt is live for it — Enter interrupts, typed or empty.
                    user_text = await session.prompt_async(
                        CLI_PROMPT, default="".join(typed_keys)
                    )
                    typed_keys.clear()
                    if speech is not None:
                        speech.interrupt()
                else:
                    user_text = await session.prompt_async(CLI_PROMPT)
            except (EOFError, KeyboardInterrupt):
                print(f"\n{GOODBYE_MESSAGE}")
                await shutdown(spoken_goodbye=voice_on)
                return

        user_text = user_text.strip()
        if not user_text:
            continue

        if user_text.lower().startswith(VOICE_COMMAND):
            arg = user_text[len(VOICE_COMMAND):].strip().lower()
            if arg == "on":
                voice_on = await enable_voice()
                print("Voice on." if voice_on else "Voice stays off.")
            elif arg == "off":
                if speech is not None:
                    speech.interrupt()
                voice_on = False
                print("Voice off.")
            else:
                print(f"Usage: {VOICE_COMMAND} on|off")
            continue

        if user_text.lower().startswith(DIGEST_COMMAND):
            args = user_text[len(DIGEST_COMMAND):].split()
            force = DIGEST_FLAG_FORCE in args
            day_arg = next((a for a in args if not a.startswith("--")), None)
            if DIGEST_FLAG_ALL in args:
                try:
                    async for date, day in orchestrator.digest_all(force=force):
                        if day is None:
                            print(f"Skipped {date} (already digested)", flush=True)
                        else:
                            print(_digest_summary(day), flush=True)
                except Exception as exc:  # a failed digest must never end the chat loop
                    # Finished days are cached, so rerunning --all resumes here.
                    print(f"Digest failed: {exc}")
                continue
            print(f"Digesting {day_arg or 'today'}…", flush=True)
            try:
                day = await orchestrator.digest_day(day_arg, force=force)
            except FileNotFoundError:
                print(f"No event log for {day_arg or 'today'} — nothing to digest.")
            except Exception as exc:  # a failed digest must never end the chat loop
                print(f"Digest failed: {exc}")
            else:
                print(_digest_summary(day))
            continue

        if user_text.lower().startswith(MERGE_COMMAND):
            try:
                profile = orchestrator.merge_profile()
            except Exception as exc:  # a failed merge must never end the chat loop
                print(f"Merge failed: {exc}")
            else:
                groups = {f.conflict_group for f in profile.facts if f.conflict_group}
                print(
                    f"Merged {len(profile.source_days)} day digest(s): "
                    f"{len(profile.facts)} durable facts, "
                    f"{len(groups)} conflict group(s) → {PROFILE_PATH}"
                )
            continue

        if user_text.lower() in EXIT_COMMANDS:
            print(GOODBYE_MESSAGE)
            await shutdown(spoken_goodbye=voice_on)
            return

        if voice_on:
            try:
                await run_turn_voiced(user_text)
            except KeyboardInterrupt:
                print(f"\n{GOODBYE_MESSAGE}")
                await shutdown(spoken_goodbye=True)
                return
        else:
            await run_turn(user_text)
