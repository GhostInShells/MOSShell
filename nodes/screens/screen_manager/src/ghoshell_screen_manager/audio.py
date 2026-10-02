"""The audio data source — a seam the background visuals draw from.

The screen's background reads ``AudioSampleTopic`` (bilateral role, 5Hz, rms/peak +
spectrum bins + downsampled waveform, no PCM). Two producers plug into the same
``sample()`` shape:

- ``TopicAudioSource`` — the live bus, and the default. The listening side publishes
  ``role=user``, the speaking side ``role=ghost``; both reach this node across
  processes over the Matrix topic bridge. Speech arriving *is* the switch — there is
  no demo mode to flip first.
- ``MockAudioSource`` — a fabricated feed for demos and headless runs, behind
  ``--mock-audio``.

Nothing downstream knows which one it is.
"""

from __future__ import annotations

import asyncio
import contextlib
import math
import random
import time
from typing import TYPE_CHECKING, Protocol

from ghoshell_moss.types.topics.audio import AudioSampleTopic

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ghoshell_moss.core.concepts.topic import TopicService

__all__ = [
    "AudioSource",
    "MockAudioSource",
    "TopicAudioSource",
    "DEMO_MODES",
    "IDLE_TIMEOUT",
]

DEMO_MODES = ("idle", "ghost", "user")

IDLE_TIMEOUT = 0.4
"""No fresh sample for this long reads as silence.

Speech that stopped must stop the visuals on screen: the last sample would otherwise
be replayed forever by the 5Hz pump and the rain would never lift.
"""


def silent_sample() -> AudioSampleTopic:
    """A below-gate sample — the shape ``idle`` uses, shared by both sources."""
    return AudioSampleTopic(
        role="ghost",
        rms_db=-80.0,
        spectrum_bins=[-96.0] * 16,
        waveform=[0.0] * 128,
    )


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
            return silent_sample()
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


class TopicAudioSource:
    """The live source: subscribe ``AudioSampleTopic`` off the bus, latest wins.

    The pump draws at 5Hz; this holds the newest sample it has seen and hands it
    back. Cross-process by construction — the listening side and the speaking side
    publish in their own processes, and the subscription rides the Matrix topic
    bridge into this node.

    A sample older than ``idle_timeout`` reads as silence, so the background drains
    the moment speech stops instead of freezing on its last frame.
    """

    def __init__(
        self,
        topics: TopicService,
        *,
        idle_timeout: float = IDLE_TIMEOUT,
    ) -> None:
        self._topics = topics
        self._idle_timeout = idle_timeout
        self._latest: AudioSampleTopic | None = None
        self._latest_at = 0.0
        self._seen = 0
        self._task: asyncio.Task | None = None

    @property
    def running(self) -> bool:
        return self._task is not None and not self._task.done()

    def stats(self) -> dict:
        """手上这帧的真实样子 —— 给模型自查用，不给人看。

        可视化出问题时，第一件事是分清"数据是什么"和"画法做了什么"。
        这个读数只回答前者：几个桶、各桶多少分贝、多久之前到的。
        """
        s = self._latest
        return {
            "role": getattr(s, "role", ""),
            "rms_db": round(getattr(s, "rms_db", 0.0), 1),
            "n_bins": len(getattr(s, "spectrum_bins", []) or []),
            "bins": [round(v, 1) for v in (getattr(s, "spectrum_bins", []) or [])],
            "age_ms": int((time.monotonic() - self._latest_at) * 1000) if s else -1,
            "subscribed": self.running,
        }

    def sample(self) -> AudioSampleTopic:
        if self._latest is None or time.monotonic() - self._latest_at > self._idle_timeout:
            return silent_sample()
        return self._latest

    async def start(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(self._run(), name="screen-manager-audio")

    async def stop(self) -> None:
        if self._task is None:
            return
        self._task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await self._task
        self._task = None

    def _log_every(self, *, now_mono: float, every: int = 10) -> None:
        """每 N 帧往 stdout 打一行真实频谱 —— 模型自己读日志，不用人去念屏幕。"""
        self._seen += 1
        if self._seen % every:
            return
        s = self._latest
        bins = getattr(s, "spectrum_bins", []) or []
        print(
            f"[audio] #{self._seen} role={getattr(s, 'role', '')} "
            f"rms={getattr(s, 'rms_db', 0.0):.1f} n={len(bins)} "
            f"bins={','.join(f'{v:.0f}' for v in bins)}",
            flush=True,
        )

    async def _run(self) -> None:
        subscriber = self._topics.subscribe_model(AudioSampleTopic, maxsize=1)
        async with subscriber:
            while True:
                try:
                    sample = await subscriber.poll_model(timeout=self._idle_timeout)
                except asyncio.TimeoutError:
                    # Quiet bus: keep the last sample only as long as it is fresh —
                    # ``sample()`` decides, so there is nothing to do here.
                    continue
                if sample is not None:
                    self._latest = sample
                    self._latest_at = time.monotonic()
                    self._log_every(now_mono=self._latest_at)
