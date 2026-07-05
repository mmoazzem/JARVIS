"""
Provider-agnostic TTS interface.

Mirrors models/base.py's seam: the rest of the app depends only on `BaseTTS` and
`AudioClip`, never on a specific engine. Synthesis returns AUDIO DATA — it never
plays. Playback is a presentation concern (the CLI plays locally today; a future
frontend streams the same bytes to a browser).
"""
from __future__ import annotations

from abc import ABC, abstractmethod

from pydantic import BaseModel as PydanticBaseModel


class AudioClip(PydanticBaseModel):
    """One synthesized utterance as raw PCM plus the format needed to play it."""

    pcm: bytes
    sample_rate: int
    sample_width: int  # bytes per sample (2 = 16-bit)
    channels: int

    @property
    def duration_s(self) -> float:
        frame_bytes = self.sample_width * self.channels
        return len(self.pcm) / (self.sample_rate * frame_bytes) if frame_bytes else 0.0


class BaseTTS(ABC):
    """Async interface every TTS engine implements.

    Implementations own a single voice (a constructor parameter from config) and
    must be safe to call sequentially from the event loop; CPU-bound synthesis
    belongs in an executor inside the implementation.
    """

    @abstractmethod
    async def synthesize(self, text: str) -> AudioClip:
        """Turn one piece of text into an AudioClip. Never plays audio."""
        raise NotImplementedError
