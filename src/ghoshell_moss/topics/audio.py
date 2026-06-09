"""
Audio and speech topic models.

These are implementation-layer topics (like channels/bridges), not contracts.
They are published/consumed via TopicService at runtime.
"""
from pydantic import Field

from ghoshell_moss.core.concepts.topic import TopicModel

__all__ = [
    "AudioRuntimeTopic",
    "SpeechTopic",
]


class AudioRuntimeTopic(TopicModel):
    """Audio capture runtime state broadcast via TopicWindow (max_size=1).

    Replaces the old tmp_storage one-shot write with a continuously
    updatable topic — consumers get heartbeat, running state, and stream
    location without polling the filesystem.
    """

    running: bool = False
    stream_key: str = Field(default="", description="Identifier for the audio stream location")
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


class SpeechTopic(TopicModel):
    """A completed utterance in a voice conversation stream.

    Each SpeechTopic is a finished sentence segment — spoken by human, ghost,
    assistant, or system. ASR streams intermediate results internally but only
    publishes to this topic once segmentation completes. No delta/incremental
    updates; every event is self-contained.

    A TopicWindow[SpeechTopic] over recent N utterances forms the conversation
    context window for the current voice interaction.
    """

    text: str = ""
    speaker_id: str = ""
    speaker_name: str = ""
    role: str = ""

    batch_id: str = ""
    timestamp: float = 0.0

    lang: str = "zh"
    audio_key: str | None = Field(default=None, description="Reference key to the audio recording, if stored")

    @classmethod
    def topic_type(cls) -> str:
        return "speech"

    @classmethod
    def default_topic_name(cls) -> str:
        return "speech"
