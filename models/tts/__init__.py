"""TTS engine factory — config names an engine, this maps it to a class.

The registry keeps engine names out of core logic (CLAUDE.md: generalize by
protocol); a future kokoro engine is one entry + one file away.
"""
from __future__ import annotations

from core.constants import TTS_ENGINE_PIPER
from models.tts.base import AudioClip, BaseTTS
from models.tts.piper_tts import PiperTTS


def create_tts(engine: str, voice: str) -> BaseTTS:
    if engine == TTS_ENGINE_PIPER:
        return PiperTTS(voice)
    raise ValueError(f"unknown tts_engine: {engine!r} (supported: {TTS_ENGINE_PIPER!r})")
