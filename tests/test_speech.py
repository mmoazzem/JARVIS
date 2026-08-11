"""
The two things speech can get wrong without ever looking wrong.

1. `speaking_started` bound to the token stream instead of to playback. The
   frontend would show SPEAKING over silence and every happy-path run would
   still pass, because the text and the audio both arrive eventually.
2. An interruption filed against the wrong turn. The old rule inferred the turn
   ("attach if exactly one is open") and was correct for the CLI, which has one
   of everything; with browser speech it writes a fact about the wrong exchange
   into the memory pipeline, where nothing downstream can tell it is false.
"""
import asyncio
import threading
import time

from core.constants import (
    EVENT_SPEAKING_STARTED,
    EVENT_SPEECH_DONE,
    EVENT_SPEECH_INTERRUPTED,
    WS_SPEECH_END,
    WS_SPEECH_PENDING,
    WS_SPEECH_START,
)
from core.memory.event_log import EventLog
from interface.speech import SpeechController
from interface.speech_session import SpeechDirector
from models.tts.base import AudioClip
from setup.config import JarvisConfig


class StubTTS:
    async def synthesize(self, text: str) -> AudioClip:
        return AudioClip(pcm=b"\x20\x11" * 512, sample_rate=22050, sample_width=2, channels=1)


class StubPlayer:
    """A player that can be held at the instant before the first sound.

    The real one only signals on_start once audio is AUDIBLE; this reproduces
    that gap on demand so the test can look at what has been emitted while a
    clip is in the sink but nothing can be heard yet.
    """

    # Every wait is bounded: a failed assertion must not strand a worker thread
    # and hang the run instead of reporting.
    LIMIT_S = 5.0

    def __init__(self) -> None:
        self.entered = threading.Event()  # play() has the clip
        self.audible = threading.Event()  # ...and may now make sound
        self.finished = threading.Event()  # ...and has reached the end of it

    def play(self, clip, stop, on_start=None, on_level=None) -> bool:
        self.entered.set()
        if not self._hold(self.audible, stop):
            return False
        if on_start is not None:
            on_start()
        if on_level is not None:
            on_level(0.5)
        return self._hold(self.finished, stop)

    def _hold(self, flag: threading.Event, stop: threading.Event) -> bool:
        """Wait for `flag`; False if `stop` (or the safety limit) lands first."""
        deadline = time.monotonic() + self.LIMIT_S
        while not flag.wait(0.01):
            if stop.is_set() or time.monotonic() > deadline:
                return False
        return True


async def _until(predicate, timeout: float = 3.0) -> None:
    deadline = asyncio.get_running_loop().time() + timeout
    while not predicate():
        assert asyncio.get_running_loop().time() < deadline, "timed out waiting"
        await asyncio.sleep(0.01)


async def test_speaking_started_follows_playback_not_the_first_token():
    seen: list[tuple] = []
    player = StubPlayer()
    speech = SpeechController(StubTTS(), player, on_event=lambda h, e: seen.append((h, e["type"])))
    speech.ensure_started()
    speech.begin_turn("turn-1")

    # A whole sentence, synthesized, handed to the player — everything short of
    # sound. This is exactly the moment a token-bound implementation would have
    # already announced that Jarvis is speaking.
    speech.feed({"type": "token", "content": "Buffalo, New York. "})
    assert await asyncio.to_thread(player.entered.wait, 3.0)
    assert seen == []

    player.audible.set()
    await _until(lambda: any(kind == EVENT_SPEAKING_STARTED for _, kind in seen))
    assert seen[0] == ("turn-1", EVENT_SPEAKING_STARTED)

    player.finished.set()
    speech.feed({"type": "done"})
    await _until(lambda: any(kind == EVENT_SPEECH_DONE for _, kind in seen))
    await speech.aclose()


async def test_interruption_ends_the_bracket_and_names_its_own_turn(tmp_path):
    event_log = EventLog(enabled=True, log_dir=tmp_path)
    # Two turns open at once — the shape the old inference could not survive.
    other_tab = event_log.begin_turn("what is the capital of France")
    mine = event_log.begin_turn("what is the weather")

    player = StubPlayer()

    async def build(config, on_event):
        return SpeechController(StubTTS(), player, on_event=on_event)

    sent: list[str] = []

    async def send(event: dict) -> None:
        sent.append(event["type"])

    director = SpeechDirector(JarvisConfig(tts_enabled=True), event_log, build=build)
    session = await director.open(mine, send)
    assert sent == [WS_SPEECH_PENDING]

    session.feed({"type": "token", "content": "Rain, mostly. "})
    assert await asyncio.to_thread(player.entered.wait, 3.0)
    player.audible.set()
    await _until(lambda: WS_SPEECH_START in sent)

    director.abort(session)  # the tab closed while the answer was being read out

    await _until(lambda: WS_SPEECH_END in sent)
    assert [e["type"] for e in mine.record["events"]] == [EVENT_SPEECH_INTERRUPTED]
    assert other_tab.record["events"] == []
