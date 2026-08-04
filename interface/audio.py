"""
Local audio playback via PulseAudio (WSLg) — presentation only.

Plays AudioClips through PulseAudio's simple API with ctypes; no Python audio
package needed. The client stack is the system one — `libpulse0` is a documented
setup prerequisite. (Historical: a no-root WSL bootstrap once extracted the libs
into vendor/pulse; that workaround is retired, see CLAUDE.md.)

Interruption contract: play() writes PCM in small chunks and checks a
threading.Event between chunks; when set, the server-side buffer is FLUSHED
(discarded), not drained — the sound stops at once, mid-word if need be.
"""
from __future__ import annotations

import ctypes
import ctypes.util
import logging
import threading
import time

from core.constants import (
    LOGGER_SPEECH,
    PLAYBACK_BUFFER_MS,
    PLAYBACK_CHUNK_MS,
)
from models.tts.base import AudioClip

logger = logging.getLogger(LOGGER_SPEECH)

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

    def play(self, clip: AudioClip, stop: threading.Event) -> bool:
        """Play a clip to the default sink. Returns False if interrupted.

        BLOCKING — call via asyncio.to_thread. `stop` may be set from any thread;
        the current chunk finishes (≤ PLAYBACK_CHUNK_MS) and the rest is flushed.
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

        chunk_bytes = int(bytes_per_s * PLAYBACK_CHUNK_MS / 1000)
        interrupted = False
        try:
            for offset in range(0, len(clip.pcm), chunk_bytes):
                if stop.is_set():
                    interrupted = True
                    break
                chunk = clip.pcm[offset : offset + chunk_bytes]
                if pa.pa_simple_write(stream, chunk, len(chunk), ctypes.byref(err)) < 0:
                    raise AudioUnavailableError(f"pa_simple_write failed (error {err.value})")

            # Everything is written; the buffered tail is still sounding. Wait it out
            # interruptibly — pa_simple_drain would block past the stop signal.
            # NOTE: the WSLg RDP sink reports a constant ~128ms device-latency floor
            # that never reaches zero, so "tail played out" is detected as the
            # latency PLATEAUING, not as it hitting a fixed threshold.
            previous_us = None
            while not interrupted:
                latency_us = pa.pa_simple_get_latency(stream, ctypes.byref(err))
                drained = latency_us <= PLAYBACK_CHUNK_MS * 1000 or (
                    previous_us is not None and latency_us >= previous_us
                )
                if drained:
                    time.sleep(min(latency_us, PLAYBACK_CHUNK_MS * 1000) / 1_000_000)
                    break
                previous_us = latency_us
                if stop.wait(PLAYBACK_CHUNK_MS / 1000):
                    interrupted = True

            if interrupted:
                pa.pa_simple_flush(stream, ctypes.byref(err))  # discard, don't drain
        finally:
            pa.pa_simple_free(stream)
        return not interrupted
