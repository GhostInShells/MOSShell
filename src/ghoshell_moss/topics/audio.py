"""
Audio and speech topic models.

These are implementation-layer topics (like channels/bridges), not contracts.
They are published/consumed via TopicService at runtime.
"""
from typing import Literal

from pydantic import Field

from ghoshell_moss.core.concepts.topic import TopicModel

__all__ = [
    "AudioSampleTopic",
    "ClauseTopic",
]


class ClauseTopic(TopicModel):
    """One clause (分句) of a spoken conversation — a single finalized sentence.

    Bilateral: the listening side publishes a clause when ASR finalizes it, the
    speaking side when TTS renders it. A ``TopicWindow[ClauseTopic]`` over recent
    clauses is the interleaved conversation trajectory — who said what, in order,
    across both sides.

    The unit is the clause, not the turn: a turn may hold several clauses, and
    only clauses interleave cleanly between speakers. Each topic is self-contained
    and final — no delta/incremental updates.

    This model carries only the semantic content of the clause. Anything tied to
    a specific transport or storage shape — audio references, ASR segment
    linkage — does NOT belong in a field; attach it via ``additional``.
    """

    text: str = Field(default="", description="The clause's own text.")
    speaker_id: str = Field(default="", description="Stable identity of who spoke the clause.")
    speaker_name: str = Field(default="", description="Display name of who spoke the clause.")
    role: str | Literal['ghost', 'user'] = Field(
        default='',
        description="Which side of the conversation produced the clause. Orthogonal to speaker "
                    "identity: several speakers can share one role.",
    )
    lang: str = Field(default="", description="Language of the clause.")

    @classmethod
    def topic_type(cls) -> str:
        return "clause"

    @classmethod
    def default_topic_name(cls) -> str:
        return "clause"


class AudioSampleTopic(TopicModel):
    """一个 ~200ms 声音事件的可视化采样 — 双边 (听/说), 以 ``role`` 区分.

    预计算频谱摘要 (rms/peak + spectrum_bins + 下采样 waveform), GUI 订阅后直接绘制:
      - ``waveform`` → 心跳线/ECG (单帧即画, 无需 window)
      - ``rms_db`` → 分贝轨迹 (TopicWindow 累积历史)
      - ``spectrum_bins`` → 柱状跳跃 (最新帧)
    每个 topic 自包含、无 delta. 无原始 PCM (对齐 AudioPlaybackTopic 的 no-PCM 惯例).
    生产 cadence 约 5Hz (``contracts.audio.AUDIO_SAMPLE_INTERVAL``).
    """

    role: Literal["user", "ghost"] = Field(
        description="谁的声音: user=听侧/麦克风, ghost=说侧/TTS.",
    )
    sample_rate: int = Field(default=0, description="采样率 (Hz).")
    duration: float = Field(default=0.0, description="本窗口时长 (秒), ~0.2.")
    rms_db: float = Field(default=0.0, description="RMS 响度 (dB).")
    peak: float = Field(default=0.0, description="峰值振幅 (0.0–1.0).")
    spectrum_bins: list[float] = Field(default_factory=list, description="N 个频段能量 (dB).")
    n_spectrum_bins: int = Field(default=16, description="spectrum_bins 的桶数.")
    waveform: list[float] = Field(default_factory=list, description="下采样有符号振幅, 供 ECG.")
    n_waveform: int = Field(default=128, description="waveform 的目标点数.")

    @classmethod
    def topic_type(cls) -> str:
        return "audio/sample"

    @classmethod
    def default_topic_name(cls) -> str:
        return "audio/sample"
