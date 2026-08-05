"""Voice input contracts — controller ABC, strongly-typed lifecycle events, handler protocol.

Dependency discipline: depends on contracts layer (ConfigType, pydantic). No matrix, no channel,
no transport. The implementation wires concrete deps (miniaudio, VolcengineASR).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from typing import Callable, Protocol

from pydantic import BaseModel, Field

from ghoshell_moss.contracts.configs import ConfigType

__all__ = [
    # enums
    "StreamState",
    "VoiceMode",
    # events (strongly typed, never dict)
    "VoiceLifecycleEvent",
    "StreamStateChanged",
    "AsrPartial",
    "AsrFinal",
    "BufferUpdated",
    # handler
    "EventHandler",
    # controller
    "VoiceController",
    # data
    "VoiceConfig",
    "DeviceConfig",
    "VoiceNodeRuntime",
]

Disposer = Callable[[], None]


# ── enums ──


class StreamState(str, Enum):
    """L2 stream-layer utterance lifecycle state."""
    IDLE = "idle"
    ARMED = "armed"
    CAPTURING = "capturing"
    FINALIZING = "finalizing"
    STAGED = "staged"
    COMMITTED = "committed"
    DROPPED = "dropped"


class VoiceMode(str, Enum):
    """Interaction mode — L2 trigger + L4 parameter combo. KD2: named config preset, not a separate state machine."""
    OFF = "off"
    PTT = "ptt"
    ENTER = "enter"
    TURN_TAKING = "turn_taking"
    DUPLEX = "duplex"


# ── config models (persistent, read from node config.toml) ──


class DeviceConfig(BaseModel):
    """Audio device selection — three-tier fallback."""
    device_pattern: str = ""  # name substring match (empty = skip)
    device_index: int | None = None  # enumeration position
    # fallback: system default (when neither matches)


class VoiceConfig(ConfigType):
    """10-switch orthogonal config. Stored as JSON (voice.json), env-resolved ($VAR).

    Extends ConfigType for conf_name() path convention + ConfigStore compatibility.
    Serialization uses JSON to avoid YAML enum friction.
    """

    @classmethod
    def conf_name(cls) -> str:
        return "voice"

    @classmethod
    def load_json(cls, storage) -> "VoiceConfig":
        """Load from Storage (voice.json). Returns default instance if absent."""
        name = f"{cls.conf_name()}.json"
        if storage.exists(name):
            return cls.model_validate_json(storage.get(name))
        return cls()

    def save_json(self, storage) -> None:
        """Persist to Storage as voice.json."""
        name = f"{self.conf_name()}.json"
        storage.put(name, self.model_dump_json(indent=2, exclude_none=True).encode())

    # ── 10 orthogonal switches ──
    listening: bool = True  # 1
    control_owner: str = "auto"  # 2: auto / ghost / human (human-handover Round 2+)
    barge_in: bool = True  # 4: first-packet TTS interrupt
    barge_in_target: str = "speech"  # 4: speech / all / none
    attention: bool = True  # 5: first-packet attention preempt
    audio_store: bool = False  # 6: persist audio (future, depends on matrix-resources)
    gate: str = "auto"  # 7: auto / manual (manual Round 2+)
    rewrite: str = "off"  # 8: off / vad / stream (flash agent, Round 3+)
    priority: str = "warning"  # 9: signal priority value
    user_identity: str = ""  # 10: speaker label (editable, no voiceprint yet)
    # ── interaction mode ──
    mode: VoiceMode = VoiceMode.TURN_TAKING
    # ── device ──
    device: DeviceConfig = Field(default_factory=DeviceConfig)
    # ── asr ──
    asr_end_window_ms: int = 500


# ── lifecycle events (strongly typed) ──


class VoiceLifecycleEvent(BaseModel):
    """Base for all voice lifecycle events. Subclass at the EventHandler method level."""
    timestamp: float = 0.0


class StreamStateChanged(VoiceLifecycleEvent):
    state: StreamState


class AsrPartial(VoiceLifecycleEvent):
    utterance_id: str
    text: str


class AsrFinal(VoiceLifecycleEvent):
    utterance_id: str
    text: str


class BufferUpdated(VoiceLifecycleEvent):
    content: str  # current staged block (editable in manual mode)


# ── handler (one method per event type) ──


class EventHandler(Protocol):
    """Protocol-layer handler — one typed method per lifecycle event. Never dict."""

    def on_stream_state_changed(self, e: StreamStateChanged) -> None: ...

    def on_asr_partial(self, e: AsrPartial) -> None: ...

    def on_asr_final(self, e: AsrFinal) -> None: ...

    def on_buffer_updated(self, e: BufferUpdated) -> None: ...


# ── runtime snapshot (for VoiceNodeRuntimeTopic) ──


class VoiceNodeRuntime(BaseModel):
    """Snapshot returned by controller.snapshot() — published as VoiceNodeRuntimeTopic."""
    running: bool = False

    # L1 device
    device_name: str = ""
    device_sample_rate: int = 0

    # L2 stream
    mode: str = "off"
    stream_state: str = "idle"
    barge_in_target: str = "speech"

    # L3 processing
    asr_partial: str = ""
    staged_text: str = ""
    gate: str = "auto"

    # L4 nucleus / buffer
    attention_occupied: bool = False
    buffer_depth: int = 0


# ── controller ABC ──


class VoiceController(ABC):
    """Voice input core contract. Channels pull this from IoC; impl lives in host layer.
    Does not depend on matrix, channel, or transport."""

    # -- control (low-frequency, called by channel commands) --

    @abstractmethod
    async def start(self) -> None:
        """Open microphone, begin capture→ASR→state machine loop."""

    @abstractmethod
    async def stop(self) -> None:
        """Close microphone, stop loop."""

    @abstractmethod
    async def set_mode(self, mode: VoiceMode) -> None:
        """Switch interaction mode (L2 trigger + L4 parameters)."""

    @abstractmethod
    async def set_config(self, config: VoiceConfig) -> None:
        """Replace current config. Writes back to config.toml."""

    # -- lifecycle events (protocol layer subscribes) --

    @abstractmethod
    def add_handler(self, handler: EventHandler) -> Disposer:
        """Register an EventHandler. Returns a disposer to unregister."""

    # -- state query --

    @abstractmethod
    def snapshot(self) -> VoiceNodeRuntime:
        """Return current state snapshot for VoiceNodeRuntimeTopic broadcast."""
