"""The audio data source — a seam the background visuals draw from.

The screen's background reads ``AudioSampleTopic`` (bilateral role, 5Hz, rms/peak +
spectrum bins + downsampled waveform, no PCM). Today the only producer is
``MockAudioSource``, which fabricates samples from a demo mode; the real producer —
a subscription to ``types/topics/audio.py`` — plugs into the same ``sample()`` shape
later. Nothing downstream knows the source is a mock.
"""

from __future__ import annotations

import math
import random
from typing import Protocol

from ghoshell_moss.types.topics.audio import AudioSampleTopic

__all__ = ["AudioSource", "MockAudioSource", "DEMO_MODES"]

DEMO_MODES = ("idle", "ghost", "user")


class AudioSource(Protocol):
    """Anything that produces an ``AudioSampleTopic`` on demand."""

    def sample(self) -> AudioSampleTopic: ...


class MockAudioSource:
    """Fabricates ``AudioSampleTopic`` samples from a demo mode.

    ``idle`` emits a silent sample (below the gate), so the background freezes;
    ``ghost`` emits speech with energy biased to low bins (character rain);
    ``user`` emits speech with flat energy (the listening ECG).
    """

    def __init__(self, seed: int | None = None) -> None:
        self._mode = "idle"
        self._rng = random.Random(seed)

    @property
    def mode(self) -> str:
        return self._mode

    def set_mode(self, mode: str) -> None:
        if mode not in DEMO_MODES:
            raise ValueError(f"unknown audio mode {mode!r} — {DEMO_MODES}")
        self._mode = mode

    def sample(self) -> AudioSampleTopic:
        if self._mode == "idle":
            return AudioSampleTopic(
                role="ghost",
                rms_db=-80.0,
                spectrum_bins=[-96.0] * 16,
                waveform=[0.0] * 128,
            )
        return self._speech(self._mode)

    def _speech(self, role: str) -> AudioSampleTopic:
        rng = self._rng
        if role == "ghost":
            rms_db = -26.0 + rng.random() * 14.0
            peak = 0.4 + rng.random() * 0.6
            spectrum_bins = [
                -70.0 + (rng.random() * 50.0 if i < 8 else rng.random() * 25.0)
                for i in range(16)
            ]
            waveform = [
                math.sin(i / 4 + rng.random()) * peak * (0.4 + rng.random() * 0.6)
                for i in range(128)
            ]
        else:  # user
            rms_db = -30.0 + rng.random() * 12.0
            peak = 0.3 + rng.random() * 0.5
            spectrum_bins = [-72.0 + rng.random() * 45.0 for _ in range(16)]
            waveform = [(rng.random() - 0.5) * 2 * peak for _ in range(128)]
        return AudioSampleTopic(
            role=role,
            rms_db=rms_db,
            peak=peak,
            spectrum_bins=spectrum_bins,
            waveform=waveform,
        )
