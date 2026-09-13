"""
Audio and speech topic models.

These are implementation-layer topics (like channels/bridges), not contracts.
They are published/consumed via TopicService at runtime.
"""
from typing import Literal
from pydantic import Field
from ghoshell_moss.core.concepts.topic import TopicModel

__all__ = [
    "AudioRuntimeTopic",
    "AudioPlaybackTopic",
    "ClauseTopic",
]


class AudioRuntimeTopic(TopicModel):
    """Audio capture runtime state broadcast via TopicWindow (max_size=1).

    A continuously updatable topic — consumers get heartbeat, running state, and
    stream location without polling the filesystem.
    """

    running: bool = False
    device_name: str = ""
    device_explain: str = ""
    started_at: float = 0.0
    last_heartbeat: float = 0.0

    @classmethod
    def topic_type(cls) -> str:
        return "audio/runtime"

    @classmethod
    def default_topic_name(cls) -> str:
        return "audio/runtime"


class AudioPlaybackTopic(TopicModel):
    """Real-time audio playback visualization frame.

    Published at ~20 Hz during active playback. Carries pre-computed
    spectrum bins for visualizer consumers — CLI spectrogram, dashboards,
    remote monitoring. Consumers subscribe via TopicWindow(max_size=1)
    for latest-only display.

    Published via AudioTransport alongside AudioRuntimeTopic (speaker
    gate). Detachable: no transport = no topic, no computation overhead.
    """

    stream_id: str = ""
    fragment_id: str = ""
    sample_rate: int = 0

    # Loudness summary
    rms_db: float = 0.0
    peak: float = 0.0

    # Frequency spectrum — N equal-width bins across 0..Nyquist, dB values.
    # Consumer renders directly — no need for its own FFT.
    spectrum_bins: list[float] = Field(default_factory=list)
    n_spectrum_bins: int = 16

    @classmethod
    def topic_type(cls) -> str:
        return "audio/playback"

    @classmethod
    def default_topic_name(cls) -> str:
        return "audio/playback"


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
