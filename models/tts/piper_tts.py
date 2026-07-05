"""
Piper TTS engine behind the BaseTTS seam.

piper-tts (the piper1-gpl project) is GPL-licensed — a known, deliberate choice.

Inference is CPU-only ON PURPOSE: the GPU is fully occupied by the 14B chat model
(CLAUDE.md: two large models cannot co-reside in 16 GB). Piper's medium voices run
several times faster than realtime on CPU, which is plenty for sentence streaming.
"""
from __future__ import annotations

import asyncio
import logging
import threading

from core.constants import LOGGER_SPEECH, VOICES_DIR
from models.tts.base import AudioClip, BaseTTS

logger = logging.getLogger(LOGGER_SPEECH)


class PiperTTS(BaseTTS):
    def __init__(self, voice: str) -> None:
        self._voice_name = voice
        self._model_path = VOICES_DIR / f"{voice}.onnx"
        self._voice = None  # loaded lazily — costs ~1s, skip it when voice stays off
        self._load_lock = threading.Lock()

    def _load(self):
        """Load the onnx voice once, thread-safe (called from executor threads)."""
        with self._load_lock:
            if self._voice is None:
                # Imported here so the app boots even if piper isn't installed —
                # the failure surfaces on first synthesis, where it can be handled.
                from piper import PiperVoice

                if not self._model_path.exists():
                    raise FileNotFoundError(
                        f"voice model not found: {self._model_path} — download it with "
                        f"`python -m piper.download_voices {self._voice_name} "
                        f"--data-dir {VOICES_DIR}`"
                    )
                self._voice = PiperVoice.load(str(self._model_path))
                logger.info("piper voice loaded: %s", self._voice_name)
        return self._voice

    def _synthesize_blocking(self, text: str) -> AudioClip:
        voice = self._load()
        chunks = list(voice.synthesize(text))
        if not chunks:
            return AudioClip(pcm=b"", sample_rate=22050, sample_width=2, channels=1)
        first = chunks[0]
        return AudioClip(
            pcm=b"".join(c.audio_int16_bytes for c in chunks),
            sample_rate=first.sample_rate,
            sample_width=first.sample_width,
            channels=first.sample_channels,
        )

    async def synthesize(self, text: str) -> AudioClip:
        """Synthesize off the event loop — onnx inference is CPU-bound."""
        return await asyncio.to_thread(self._synthesize_blocking, text)
