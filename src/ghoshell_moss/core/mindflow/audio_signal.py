"""
Audio signal definitions for mindflow integration.

SignalMeta lives in core.blueprint.mindflow; this module defines the
audio-specific signal payload that carries ASR results into the
attention arbitration pipeline.
"""
from enum import Enum

from ghoshell_moss.core.blueprint.mindflow import Priority, SignalMeta
from ghoshell_moss.topics.audio import SpeechTopic

__all__ = [
    "AudioAction",
    "AudioSignal",
]


class AudioAction(str, Enum):
    SPEECH_STARTED = "speech_started"
    SPEECH_FINAL = "speech_final"
    WAKE_WORD = "wake_word"
    AUDIO_ALERT = "audio_alert"


class AudioSignal(SignalMeta):
    """Audio perception signal — listener → mindflow attention preemption.

    ASR emits SPEECH_FINAL when a completed sentence is ready. The signal
    carries the SpeechTopic payload and challenges the Ghost's current
    attention via mindflow.
    """

    action: AudioAction
    speech_topic: SpeechTopic | None = None

    @classmethod
    def signal_name(cls) -> str:
        return "audio"

    @classmethod
    def priority(cls) -> Priority:
        return Priority.WARNING
