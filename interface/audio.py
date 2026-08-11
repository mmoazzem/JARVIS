"""
Local audio playback via PulseAudio (WSLg) — presentation only.

Plays AudioClips through PulseAudio's simple API with ctypes; no Python audio
package needed. The client stack is the system one — `libpulse0` is a documented
setup prerequisite. (Historical: a no-root WSL bootstrap once extracted the libs
into vendor/pulse; that workaround is retired, see CLAUDE.md.)

Interruption contract: play() writes PCM in small chunks and checks a
threading.Event between chunks; when set, the server-side buffer is FLUSHED
(discarded), not drained — the sound stops at once, mid-word if need be.

Metering contract: play() reports what is AUDIBLE, not what it has written.
Writes run ahead of the sound by the whole server buffer, so a meter fed at
write time would lead the ear by ~300 ms; every reading here is taken at
`written - latency`, the sample the sink is playing right now. That same
correction is what makes on_start honest: it fires when the first non-silent
sample reaches the ear, so a caller can say "speaking" and be telling the truth.
"""
from __future__ import annotations

import array
import ctypes
import ctypes.util
import logging
import threading
import time
from typing import Callable, Optional

from core.constants import (
    LOGGER_SPEECH,
    PLAYBACK_BUFFER_MS,
    PLAYBACK_CHUNK_MS,
    SPEECH_LEVEL_INTERVAL_MS,
    SPEECH_LEVEL_REFERENCE_RMS,
    SPEECH_LEVEL_WINDOW_MS,
)
from models.tts.base import AudioClip

logger = logging.getLogger(LOGGER_SPEECH)

_INT16_FULL_SCALE = 32768.0


def window_rms(pcm: bytes, end: int, width: int, sample_width: int) -> float:
    """Meter reading for the `width` bytes of PCM ending at `end`, 0..1.

    Real energy, scaled against SPEECH_LEVEL_REFERENCE_RMS so a meter driven by
    it uses its whole travel. Anything that is not 16-bit PCM reads 0 rather
    than guessing at a format the sink is not being given.
    """
    if sample_width != 2 or end <= 0:
        return 0.0
    chunk = pcm[max(0, end - width) : end]
    if len(chunk) < 2:
        return 0.0
    samples = array.array("h")
    samples.frombytes(chunk[: len(chunk) - (len(chunk) % 2)])
    rms = (sum(s * s for s in samples) / len(samples)) ** 0.5 / _INT16_FULL_SCALE
    return min(1.0, rms / SPEECH_LEVEL_REFERENCE_RMS)

_PA_SAMPLE_S16LE = 3  # matches piper output: 16-bit little-endian PCM
_PA_STREAM_PLAYBACK = 1
_APP_NAME = b"jarvis"


class AudioUnavailableError(RuntimeError):
    """No usable PulseAudio client stack — voice stays off, app runs text-only."""


class _PaSampleSpec(ctypes.Structure):
    _fields_ = [
        ("format", ctypes.c_int),
        ("rate", ctypes.c_uint32),
        ("channels", ctypes.c_uint8),
    ]


class _PaBufferAttr(ctypes.Structure):
    _fields_ = [
        ("maxlength", ctypes.c_uint32),
        ("tlength", ctypes.c_uint32),
        ("prebuf", ctypes.c_uint32),
        ("minreq", ctypes.c_uint32),
        ("fragsize", ctypes.c_uint32),
    ]


_PA_DEFAULT = ctypes.c_uint32(-1).value  # "server decides" sentinel for buffer fields


def _load_libpulse_simple() -> ctypes.CDLL:
    """Load the system libpulse-simple. Missing library = voice off, app runs on."""
    try:
        return ctypes.CDLL("libpulse-simple.so.0")
    except OSError as exc:
        raise AudioUnavailableError(
            "no PulseAudio client libraries: install libpulse0 "
            "(sudo apt install libpulse0)"
        ) from exc


class PulsePlayer:
    """Blocking PCM playback (run it in an executor); interruptible between chunks."""

    def __init__(self) -> None:
        self._pa: ctypes.CDLL | None = None
        self._load_lock = threading.Lock()

    def _lib(self) -> ctypes.CDLL:
        with self._load_lock:
            if self._pa is None:
                pa = _load_libpulse_simple()
                pa.pa_simple_new.restype = ctypes.c_void_p
                pa.pa_simple_new.argtypes = [
                    ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p,
                    ctypes.c_char_p, ctypes.POINTER(_PaSampleSpec), ctypes.c_void_p,
                    ctypes.c_void_p, ctypes.POINTER(ctypes.c_int),
                ]
                pa.pa_simple_write.argtypes = [
                    ctypes.c_void_p, ctypes.c_char_p, ctypes.c_size_t,
                    ctypes.POINTER(ctypes.c_int),
                ]
                pa.pa_simple_flush.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int)]
                pa.pa_simple_drain.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int)]
                pa.pa_simple_get_latency.restype = ctypes.c_uint64
                pa.pa_simple_get_latency.argtypes = [
                    ctypes.c_void_p, ctypes.POINTER(ctypes.c_int)
                ]
                pa.pa_simple_free.argtypes = [ctypes.c_void_p]
                self._pa = pa
                logger.info("pulseaudio client loaded")
        return self._pa

    def play(
        self,
        clip: AudioClip,
        stop: threading.Event,
        on_start: Optional[Callable[[], None]] = None,
        on_level: Optional[Callable[[float], None]] = None,
    ) -> bool:
        """Play a clip to the default sink. Returns False if interrupted.

        BLOCKING — call via asyncio.to_thread. `stop` may be set from any thread;
        the current chunk finishes (≤ one write step) and the rest is flushed.

        on_start fires once, when the first sample with any energy in it becomes
        AUDIBLE. on_level fires at SPEECH_LEVEL_INTERVAL_MS with the RMS of what
        is sounding at that moment. Both run ON THIS THREAD — a caller living in
        an event loop must hand them across itself.
        """
        if not clip.pcm:
            return True
        pa = self._lib()

        spec = _PaSampleSpec(_PA_SAMPLE_S16LE, clip.sample_rate, clip.channels)
        bytes_per_s = clip.sample_rate * clip.sample_width * clip.channels
        # Small server buffer: an interrupt only ever has to flush this much tail.
        attr = _PaBufferAttr(
            _PA_DEFAULT, int(bytes_per_s * PLAYBACK_BUFFER_MS / 1000),
            _PA_DEFAULT, _PA_DEFAULT, _PA_DEFAULT,
        )
        err = ctypes.c_int(0)
        stream = pa.pa_simple_new(
            None, _APP_NAME, _PA_STREAM_PLAYBACK, None, b"speech",
            ctypes.byref(spec), None, ctypes.byref(attr), ctypes.byref(err),
        )
        if not stream:
            raise AudioUnavailableError(f"pa_simple_new failed (error {err.value})")

        metered = on_start is not None or on_level is not None
        # A meter turns the write loop into the playback clock: once the sink's
        # buffer is full every write blocks for exactly one step, so stepping at
        # the meter's interval costs nothing and needs no second timer. The
        # interrupt check rides the same boundary and only gets finer.
        step_ms = SPEECH_LEVEL_INTERVAL_MS if metered else PLAYBACK_CHUNK_MS
        # Every byte count here is rounded to whole FRAMES. PulseAudio rejects a
        # write that splits a frame (50 ms of 22.05 kHz mono is 2205 bytes — odd),
        # and a meter window read off a half-frame boundary is noise, not audio.
        frame_bytes = clip.sample_width * clip.channels

        def whole_frames(byte_count: float) -> int:
            return (int(byte_count) // frame_bytes) * frame_bytes

        chunk_bytes = max(frame_bytes, whole_frames(bytes_per_s * step_ms / 1000))
        window_bytes = max(frame_bytes, whole_frames(bytes_per_s * SPEECH_LEVEL_WINDOW_MS / 1000))
        started = False
        interrupted = False

        def reading(written: int, latency_us: int) -> None:
            """One meter reading of the audio the sink is playing RIGHT NOW."""
            nonlocal started
            audible = whole_frames(written - int(bytes_per_s * latency_us / 1_000_000))
            rms = window_rms(clip.pcm, audible, window_bytes, clip.sample_width)
            if not started:
                if rms <= 0:
                    return  # still inside the pre-roll pad — nothing is heard yet
                started = True
                if on_start is not None:
                    on_start()
            if on_level is not None:
                on_level(rms)

        try:
            written = 0
            for offset in range(0, len(clip.pcm), chunk_bytes):
                if stop.is_set():
                    interrupted = True
                    break
                chunk = clip.pcm[offset : offset + chunk_bytes]
                if pa.pa_simple_write(stream, chunk, len(chunk), ctypes.byref(err)) < 0:
                    raise AudioUnavailableError(f"pa_simple_write failed (error {err.value})")
                written += len(chunk)
                if metered:
                    reading(written, pa.pa_simple_get_latency(stream, ctypes.byref(err)))

            # Everything is written; the buffered tail is still sounding. Wait it out
            # interruptibly — pa_simple_drain would block past the stop signal.
            # NOTE: the WSLg RDP sink reports a constant ~128ms device-latency floor
            # that never reaches zero, so "tail played out" is detected as the
            # latency PLATEAUING, not as it hitting a fixed threshold. The plateau
            # is compared across PLAYBACK_CHUNK_MS as it always was — metering ticks
            # faster inside that, but never moves the samples being compared.
            per_check = max(1, round(PLAYBACK_CHUNK_MS / step_ms))
            previous_us = None
            tick = 0
            while not interrupted:
                latency_us = pa.pa_simple_get_latency(stream, ctypes.byref(err))
                if tick % per_check == 0:
                    drained = latency_us <= PLAYBACK_CHUNK_MS * 1000 or (
                        previous_us is not None and latency_us >= previous_us
                    )
                    if drained:
                        time.sleep(min(latency_us, PLAYBACK_CHUNK_MS * 1000) / 1_000_000)
                        break
                    previous_us = latency_us
                if metered:
                    reading(written, latency_us)
                tick += 1
                if stop.wait(step_ms / 1000):
                    interrupted = True

            if interrupted:
                pa.pa_simple_flush(stream, ctypes.byref(err))  # discard, don't drain
        finally:
            pa.pa_simple_free(stream)
        return not interrupted
