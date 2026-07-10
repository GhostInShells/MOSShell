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

    SPEECH_STARTED (incomplete) signals occupy attention for the current
    utterance id via complete=False. Whether they interrupt an existing shell
    execution is controlled by the produced Impulse/Signal semantics rather
    than being forced here.

    SPEECH_FINAL purges incomplete predecessors and produces a complete
    Impulse that delivers the full speech content to the already-occupied
    attention.  For compatibility, complete impulses interrupt by default;
    aEther full-duplex voice mode disables this via interrupt_on_complete=False.

    Reverse suppress (aligned with InterruptNucleus): pop_impulse starts a
    victory-side cooldown; suppress only clears the buffer on the failure side.
    """

    def __init__(self, *, interrupt_on_complete: bool = True, **kwargs):
        super().__init__(**kwargs)
        # 兼容默认语义：历史上 AudioNucleus 会把 complete 的语音 impulse
        # 标记为 interrupt，用于旧 listener/show 场景在最终语音到达时抢占当前
        # attention。aEther 的全双工语音不适合这个默认值，因此由
        # AudioNucleusMeta(interrupt_on_complete=False) 在 aEther mode 内显式关闭。
        self._interrupt_on_complete = interrupt_on_complete

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
            # 只在 complete impulse 上保留旧行为开关；incomplete impulse 的
            # interrupt 语义仍由 BufferNucleus/Signal 自身决定，避免扩大改动面。
            impulse.interrupt = self._interrupt_on_complete
        return impulse

    def suppress(self, suppress_by: Impulse) -> None:
        # 失败侧不进冷静期 — 只清理 buffer.
        # 与 InterruptNucleus 对齐: impulse 仲裁失败只可能是 same-id absorb
        # 或 stale, 这两种都不需要冷却. 覆写 BufferNucleus.suppress 避免
        # 在失败侧设置 _suppress_until.
        self._signals.clear()
        self._impulse_cache = None

    def pop_impulse(self, impulse: Impulse) -> None:
        # 反向 suppress: 仲裁胜利后启动冷静期, 防止 shell churn.
        # 语音打断成功后短时间内不再通知, 避免反复 stop_interpretation
        # + 重建 attention 的抖动. 与 InterruptNucleus 对齐.
        if not self.is_running():
            return
        self._suppress_until = time.monotonic() + self._suppress_seconds
        self._event_loop.create_task(self._atomic_clear_buffer())


class AudioNucleusMeta(NucleusMeta):
    """音频感知核工厂 — 生产监听 audio 信号的 AudioNucleus。"""

    def __init__(self, *, interrupt_on_complete: bool = True):
        self._interrupt_on_complete = interrupt_on_complete

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
            interrupt_on_complete=self._interrupt_on_complete,
        )
