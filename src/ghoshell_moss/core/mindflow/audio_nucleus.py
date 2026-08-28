import time

from ghoshell_common.contracts import LoggerItf
from ghoshell_container import IoCContainer

from ghoshell_moss.core.blueprint.mindflow import (
    NucleusMeta,
    Nucleus,
    SignalMeta,
    Priority,
    Signal,
    Impulse,
)
from ghoshell_moss.core.mindflow.audio_signal import AudioAction, AudioSignal
from ghoshell_moss.core.mindflow.buffer_nucleus import BufferNucleus

__all__ = [
    "AudioNucleus",
    "AudioNucleusMeta",
]


class AudioNucleus(BufferNucleus):
    """Audio signal nucleus — aggregate ASR signals into attention impulses.

    SPEECH_STARTED (incomplete) signals preempt attention immediately on the
    first packet — the incomplete Impulse carries interrupt=True, triggers
    shell.stop_interpretation(), and occupies attention via complete=False.
    Subsequent signals in the same session accumulate silently into the buffer.

    SPEECH_FINAL purges incomplete predecessors and produces a complete
    Impulse (interrupt=False) that delivers the full speech content to the
    already-occupied attention.  If no STARTED preceded it (standalone FINAL),
    the complete impulse goes through normal arbitration without interrupt.

    Reverse suppress (aligned with InterruptNucleus): pop_impulse starts a
    victory-side cooldown; suppress only clears the buffer on the failure side.
    """

    async def _process_signal(self, signal: Signal) -> None:
        audio_meta = AudioSignal.from_signal(signal)
        if audio_meta and audio_meta.action == AudioAction.SPEECH_FINAL:
            # Purge incomplete signals (SPEECH_STARTED) so FINAL produces
            # a complete Impulse.  _process_signal runs under self._lock.
            self._signals = [s for s in self._signals if s.complete]
        await super()._process_signal(signal)

    def _rebuild_impulse(self) -> Impulse | None:
        impulse = super()._rebuild_impulse()
        if impulse is not None and impulse.complete:
            # 首包打断: incomplete impulse preempts attention, claims it
            # via complete=False.  Complete (FINAL) delivers content to
            # the occupied attention without re-interrupting.
            impulse.interrupt = True
        return impulse

    def suppress(self, suppress_by: Impulse, suppressed: Impulse | None = None) -> None:
        # 失败侧不进冷静期 — 只清理 buffer.
        # 与 InterruptNucleus 对齐: impulse 仲裁失败只可能是 same-id absorb
        # 或 stale, 这两种都不需要冷却. 覆写 BufferNucleus.suppress 避免
        # 在失败侧设置 _suppress_until.
        self._signals.clear()
        self._impulse_cache = None

    def attended(self, impulse: Impulse) -> None:
        # 反向 suppress: 仲裁胜利后启动冷静期, 防止 shell churn.
        # 语音打断成功后短时间内不再通知, 避免反复 stop_interpretation
        # + 重建 attention 的抖动. 与 InterruptNucleus 对齐.
        if not self.is_running():
            return
        self._suppress_until = time.monotonic() + self._suppress_seconds
        self._event_loop.create_task(self._atomic_clear_buffer())


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
            logger=container.force_fetch(LoggerItf),
        )
