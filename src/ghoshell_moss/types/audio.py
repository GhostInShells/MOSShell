"""
Audio topic types — cross-side protocol schemas for voice conversation.

These are the data contracts for the three audio topics settled in
voice-input-state-machine (2026-08-12 会话补充(二)):

  - ConversationTopic: sentence-level bilateral conversation segments (event face)
  - AudioPlaybackTopic: playback metadata broadcast, no PCM (event face)

AudioRuntimeTopic (half-duplex gate) is state face → Parameter, not a Topic.
Defined in the parameter layer, not here.
"""
from typing import Literal

from pydantic import Field

from ghoshell_moss.core.concepts.topic import TopicModel

__all__ = [
    "ConversationTopic",
    "AudioPlaybackTopic",
]


class ConversationTopic(TopicModel):
    """A spoken sentence segment in a voice conversation — bilateral, shared
    between listener and speech sides.

    Each ConversationTopic is a finished sentence. ASR streams intermediate
    results internally; only a completed (or interrupted) segment is published.
    Every event is self-contained — no deltas.

    A TopicWindow[ConversationTopic] over recent N segments forms the dialogue
    context window for the current voice interaction.
    """

    role: Literal["ghost", "user"] = Field(
        default="user",
        description="Who spoke this sentence",
    )
    name: str = Field(
        default="",
        description="Display name of the speaker",
    )
    sentence_id: str = Field(
        default="",
        description="Unique identifier for this sentence segment",
    )
    batch_id: str = Field(
        default="",
        description="Link to L2 command-stream cid (say task.cid for ghost, "
                    "ASR session id for user)",
    )
    seq: int = Field(
        default=0,
        description="Monotonic sequence number within the conversation",
    )
    text: str = Field(
        default="",
        description="The spoken text",
    )
    lang: str = Field(
        default="",
        description="Language code of the utterance",
    )
    interrupted: bool = Field(
        default=False,
        description="True if this sentence was cut off mid-utterance "
                    "(barge-in, playback cancelled, etc.)",
    )
    spoken_at: float = Field(
        default=0.0,
        description="When the sentence was spoken (seconds, time.monotonic). "
                    "Distinct from meta.created_at which is the topic publish time",
    )
    resource: str = Field(
        default="",
        description="Reference to the persisted resource (audio file, "
                    "matrix-resources URI, etc.)",
    )

    @classmethod
    def topic_type(cls) -> str:
        return "conversation"

    @classmethod
    def default_topic_name(cls) -> str:
        return "conversation"


class AudioPlaybackTopic(TopicModel):
    """Real-time audio playback metadata broadcast at ~20 Hz during active
    playback.

    **Hard constraint: NO binary PCM.** This carries pre-computed spectrum
    metadata only — rms, peak, frequency bins. Consumers that need raw audio
    samples use the local PlaybackSample observe callback.

    Cross-cell consumers:
      - Digital human lip-sync (mouth shape from spectrum)
      - Ghost audio waveform visualization
      - Remote monitoring dashboards

    Consumers subscribe via TopicWindow(max_size=1) for latest-only display.
    Published via the speaker's playback transport. Detachable: no transport
    = no topic, no computation overhead.
    """

    stream_id: str = Field(
        default="",
        description="Playback stream identifier",
    )
    fragment_id: str = Field(
        default="",
        description="Fragment within the stream",
    )
    sample_rate: int = Field(
        default=0,
        description="Audio sample rate in Hz",
    )

    # Loudness summary
    rms_db: float = Field(
        default=0.0,
        description="RMS level in dB",
    )
    peak: float = Field(
        default=0.0,
        description="Peak amplitude (0.0–1.0)",
    )

    # Frequency spectrum — N equal-width bins across 0..Nyquist, dB values.
    # Consumer renders directly — no need for its own FFT.
    spectrum_bins: list[float] = Field(
        default_factory=list,
        description="Frequency spectrum bins in dB",
    )
    n_spectrum_bins: int = Field(
        default=16,
        description="Number of spectrum bins",
    )

    @classmethod
    def topic_type(cls) -> str:
        return "audio/playback"

    @classmethod
    def default_topic_name(cls) -> str:
        return "audio/playback"
