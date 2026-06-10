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
    "SpeechTopic",
]


class AudioRuntimeTopic(TopicModel):
    """Audio capture runtime state broadcast via TopicWindow (max_size=1).

    Replaces the old tmp_storage one-shot write with a continuously
    updatable topic — consumers get heartbeat, running state, and stream
    location without polling the filesystem.
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


class SpeechTopic(TopicModel):
    """A completed utterance in a voice conversation stream.

    Each SpeechTopic is a finished sentence segment — spoken by human, ghost,
    assistant, or system. ASR streams intermediate results internally but only
    publishes to this topic once segmentation completes. No delta/incremental
    updates; every event is self-contained.

    A TopicWindow[SpeechTopic] over recent N utterances forms the conversation
    context window for the current voice interaction.
    """

    # todo: all properties has no Filed with description
    text: str = ""
    speaker_id: str = ""
    speaker_name: str = ""
    role: str | Literal['ghost', 'user'] = Field(
        default='',
        description='role of the speaker one',
    )

    batch_id: str = ""
    # todo: remove the timestamp, it must be useless since assigned with time.monotonic
    #    also topic already has timestamp in topic.meta.created_at
    timestamp: float = 0.0

    lang: str = Field(default="", description='language of the utterance')
    # todo: normalize with audio resource (which is not implemented yet)
    audio_key: str | None = Field(default=None, description="Reference key to the audio recording, if stored")

    @classmethod
    def topic_type(cls) -> str:
        return "speech"

    @classmethod
    def default_topic_name(cls) -> str:
        return "speech"
