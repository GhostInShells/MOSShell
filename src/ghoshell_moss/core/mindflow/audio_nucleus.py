from ghoshell_container import IoCContainer

from ghoshell_moss.core.blueprint.mindflow import (
    NucleusMeta,
    Nucleus,
    SignalMeta,
    Priority,
    Signal,
)
from ghoshell_moss.core.mindflow.audio_signal import AudioAction, AudioSignal
from ghoshell_moss.core.mindflow.buffer_nucleus import BufferNucleus

__all__ = [
    "AudioNucleus",
    "AudioNucleusMeta",
    "audio_nucleus_factory",
]


class AudioNucleus(BufferNucleus):
    """Audio signal nucleus with SPEECH_STARTED filtering.

    Current policy: only SPEECH_FINAL triggers impulse. SPEECH_STARTED is
    dropped at the nucleus boundary.

    Why keep the path: when we implement TTS preemption (Step 13), a
    SPEECH_STARTED arriving while Ghost is speaking should immediately call
    Preemptable.attenuate() — without waiting for the full sentence. At that
    point this filter becomes a routing gate, not a drop.
    """

    def add_signal(self, signal: Signal) -> None:
        audio_meta = AudioSignal.from_signal(signal)
        if audio_meta and audio_meta.action == AudioAction.SPEECH_STARTED:
            # Reserved for preemption hook — see class docstring.
            self._logger.debug("Dropping SPEECH_STARTED — preemption hook reserved for future")
            return
        super().add_signal(signal)


class AudioNucleusMeta(NucleusMeta):
    """音频感知核工厂 — 生产监听 audio 信号的 AudioNucleus。"""

    def name(self) -> str:
        return "audio_nucleus"

    def description(self) -> str:
        return "audio perception signal nucleus — aggregates audio signals from ASR/listener"

    def signals(self) -> list[type[SignalMeta]]:
        return [AudioSignal]

    def factory(self, container: IoCContainer) -> Nucleus:
        return AudioNucleus(
            name="audio_nucleus",
            description="audio perception signal nucleus",
            target_signal="audio",
            default_prompt="User spoke via voice input. Process the speech.",
            suppress_seconds=0.5,
            buffer_size=5,
            min_priority=Priority.WARNING,
            pulse_beat_interval=3.0,
            logger=container.force_fetch_logger(),
        )


audio_nucleus_factory = AudioNucleusMeta()
