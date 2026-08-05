"""Two-axis state machine — utterance lifecycle × gate/buffer.

Axis 1 (utterance):  idle → armed → capturing → finalizing → idle
Axis 2 (gate):       staged → committed / dropped

The state machine is driven by audio/ASR events from the controller loop.
It dispatches typed lifecycle events through an EventBus.
"""
from __future__ import annotations

import logging
import time
import uuid

from ghoshell_moss.host.speech.voice.contracts import (
    AsrFinal,
    AsrPartial,
    BufferUpdated,
    StreamState,
    StreamStateChanged,
    VoiceConfig,
    VoiceMode,
    VoiceNodeRuntime,
)
from ghoshell_moss.host.speech.voice.handlers import EventBus

__all__ = ["VoiceStateMachine"]


class VoiceStateMachine:
    """Orchestrates the voice input lifecycle — capture → ASR → gate → commit."""

    def __init__(
        self,
        config: VoiceConfig,
        event_bus: EventBus,
        *,
        logger: logging.Logger | None = None,
    ):
        self._config = config
        self._bus = event_bus
        self._log = logger or logging.getLogger("moss.voice.state")
        # core state
        self._state: StreamState = StreamState.IDLE
        self._utterance_id: str | None = None
        self._buffer: str = ""
        # runtime
        self._asr_partial: str = ""
        self._attention_occupied: bool = False

    # ── state query ──

    @property
    def state(self) -> StreamState:
        return self._state

    @property
    def utterance_id(self) -> str | None:
        return self._utterance_id

    @property
    def buffer(self) -> str:
        return self._buffer

    def snapshot(self, *, capture_device: str = "", capture_rate: int = 0) -> VoiceNodeRuntime:
        return VoiceNodeRuntime(
            running=self._state != StreamState.IDLE,
            device_name=capture_device,
            device_sample_rate=capture_rate,
            mode=self._config.mode.value,
            stream_state=self._state.value,
            barge_in_target=self._config.barge_in_target,
            asr_partial=self._asr_partial,
            staged_text=self._buffer,
            gate=self._config.gate,
            attention_occupied=self._attention_occupied,
            buffer_depth=len(self._buffer),
        )

    # ── control ──

    def start(self) -> None:
        self._transition(StreamState.ARMED)

    def stop(self) -> None:
        self._transition(StreamState.IDLE)
        self._utterance_id = None
        self._asr_partial = ""
        self._attention_occupied = False

    def set_config(self, config: VoiceConfig) -> None:
        self._config = config

    def set_mode(self, mode: VoiceMode) -> None:
        self._config.mode = mode

    # ── ASR input (called by controller loop) ──

    def on_asr_partial(self, utterance_id: str, text: str) -> None:
        """First non-empty partial triggers armed→capturing. Subsequent partials update buffer."""
        if self._state == StreamState.ARMED:
            self._utterance_id = utterance_id
            self._transition(StreamState.CAPTURING)
        if self._state == StreamState.CAPTURING:
            self._asr_partial = text
            self._bus.dispatch(AsrPartial(utterance_id=utterance_id, text=text, timestamp=time.monotonic()))

    def on_asr_final(self, utterance_id: str, text: str) -> None:
        """ASR final result — enters staged, then gate decides."""
        if self._state not in (StreamState.CAPTURING, StreamState.FINALIZING):
            return
        # Transition through finalizing → staged
        self._transition(StreamState.FINALIZING)
        self._asr_partial = ""
        if not text:
            self._drop("empty")
            return
        self._transition(StreamState.STAGED)
        # Append to buffer (#3)
        self._buffer = self._buffer + ("\n" if self._buffer else "") + text
        self._bus.dispatch(AsrFinal(utterance_id=utterance_id, text=text, timestamp=time.monotonic()))
        self._bus.dispatch(BufferUpdated(content=self._buffer, timestamp=time.monotonic()))
        # Gate decision
        if self._config.gate == "auto":
            self._commit()
        else:
            # manual gate: staged holds, buffer editable (Round 2)
            pass

    def on_silence_timeout(self) -> None:
        """Called when silence timeout fires mid-capture."""
        if self._state == StreamState.CAPTURING:
            self._transition(StreamState.FINALIZING)

    def on_arm(self) -> None:
        """Called when mode becomes active and TTS is not playing."""
        if self._state == StreamState.IDLE:
            self._transition(StreamState.ARMED)

    # ── internals ──

    def _commit(self) -> None:
        self._transition(StreamState.COMMITTED)
        self._reset_utterance()

    def _drop(self, reason: str) -> None:
        self._log.info("Dropping utterance: %s", reason)
        self._transition(StreamState.DROPPED)
        self._reset_utterance()

    def _reset_utterance(self) -> None:
        self._utterance_id = None
        self._asr_partial = ""

    def _transition(self, to: StreamState) -> None:
        old = self._state
        self._state = to
        self._bus.dispatch(StreamStateChanged(state=to, timestamp=time.monotonic()))
        self._log.debug("State: %s → %s", old.value, to.value)
