import time
from abc import ABC
from typing import Any, ClassVar, Optional

from ghoshell_common.helpers import uuid
from pydantic import BaseModel, Field
from typing_extensions import Self, TypedDict

from ghoshell_moss.core.concepts.channel import ChannelMeta
from ghoshell_moss.core.concepts.errors import CommandErrorCode

__all__ = [
    "ChannelEvent",
    "ChannelEventModel",
    "ChannelMetaUpdateEvent",
    "ClearEvent",
    "CommandCallEvent",
    "CommandCancelEvent",
    "CommandDoneEvent",
    "CreateSessionEvent",
    "HeartbeatEvent",
    "ProviderErrorEvent",
    "ReconnectSessionEvent",
    "SessionCreatedEvent",
    "SyncChannelMetasEvent",
]

"""
Duplex Channel 双工通讯事件.
需要确认几个基本概念.

Duplex Channel 包含 Provider 和 Proxy 两个角色.
Provider 提供 Channel 能力
Proxy 消费并且重构 Channel 能力. 
"""


class ChannelEvent(TypedDict):
    event_id: str
    event_type: str
    session_id: Optional[str]
    timestamp: float
    data: Optional[dict[str, Any]]


class ChannelEventModel(BaseModel, ABC):
    event_type: ClassVar[str] = ""

    event_id: str = Field(default_factory=uuid, description="event id for transport")
    session_id: str = Field(default="", description="channel proxy id")
    timestamp: float = Field(default_factory=lambda: round(time.time(), 4), description="timestamp")

    def to_channel_event(self) -> ChannelEvent:
        data = self.model_dump(exclude_none=True, exclude={"event_type", "channel_id", "channel_name", "event_id"})
        return ChannelEvent(
            event_id=self.event_id,
            event_type=self.event_type,
            session_id=self.session_id,
            data=data,
            timestamp=self.timestamp,
        )

    @classmethod
    def from_channel_event(cls, channel_event: ChannelEvent) -> Optional[Self]:
        if cls.event_type != channel_event["event_type"]:
            return None
        data = channel_event.get("data", {})
        data["event_id"] = channel_event["event_id"]
        data["session_id"] = channel_event["session_id"]
        data["timestamp"] = channel_event["timestamp"]
        return cls(**data)

    def __str__(self):
        value = super.__str__(self)
        return value[:200]


class HeartbeatEvent(ChannelEventModel):
    """心跳事件，由客户端发送，服务器响应"""

    event_type: ClassVar[str] = "moss.heartbeat"
    direction: str = Field(default="request", description="请求或响应: request/response")


# --- proxy event --- #


class ClearEvent(ChannelEventModel):
    """发出讯号给某个 channel, 执行状态清空的逻辑"""

    event_type: ClassVar[str] = "moss.channel.proxy.clear"
    chan: str = Field(description="channel name")


class CommandCallEvent(ChannelEventModel):
    """发起一个 command 的调用."""

    # todo: 未来要加一个用 command_id 轮询 provider 状态的事件. 用来避免通讯丢失.

    event_type: ClassVar[str] = "moss.channel.proxy.command.call"
    name: str = Field(description="command name")
    chan: str = Field(description="channel name")
    command_id: str = Field(default_factory=uuid, description="command id")
    args: list[Any] = Field(default_factory=list, description="command args")
    kwargs: dict[str, Any] = Field(default_factory=dict, description="kwargs of the command")
    tokens: str = Field("", description="command tokens")
    context: dict[str, Any] = Field(default_factory=dict, description="context of the command")
    call_id: str = Field(default="")

    def not_available(self, msg: str = "") -> "CommandDoneEvent":
        return CommandDoneEvent(
            session_id=self.session_id,
            command_id=self.command_id,
            errcode=CommandErrorCode.NOT_AVAILABLE.value,
            errmsg=msg or f"command `{self.chan}:{self.name}` not available",
            result=None,
            chan=self.chan,
        )

    def cancel(self) -> "CommandCancelEvent":
        return CommandCancelEvent(
            session_id=self.session_id,
            command_id=self.command_id,
            chan=self.chan,
        )

    def done(self, result: Any, errcode: int, errmsg: str) -> "CommandDoneEvent":
        return CommandDoneEvent(
            session_id=self.session_id,
            command_id=self.command_id,
            errcode=errcode,
            errmsg=errmsg,
            chan=self.chan,
            result=result,
        )

    def not_found(self, msg: str = "") -> "CommandDoneEvent":
        return CommandDoneEvent(
            session_id=self.session_id,
            command_id=self.command_id,
            errcode=CommandErrorCode.NOT_FOUND.value,
            errmsg=msg or f"command `{self.chan}:{self.name}` not found",
            chan=self.chan,
            result=None,
        )


class CommandCancelEvent(ChannelEventModel):
    """通知 channel 指定的 command 被取消."""

    event_type: ClassVar[str] = "moss.channel.proxy.command.cancel"
    chan: str = Field(description="channel name")
    command_id: str = Field(description="command id")


class SyncChannelMetasEvent(ChannelEventModel):
    """要求同步 channel 的 meta 信息."""

    event_type: ClassVar[str] = "moss.channel.proxy.meta.sync"


class ReconnectSessionEvent(ChannelEventModel):
    """
    Proxy 告知 Provider 传送的事件 Session Id 未对齐, 需要重新建立 session, 双方清空状态.
    """

    event_type: ClassVar[str] = "moss.channel.proxy.session.reconnect"


class SessionCreatedEvent(ChannelEventModel):
    """
    proxy 告知 provider session 已经确认并创建了.
    握手后期待服务端发送 UpdateChannelMeta 事件进行同步.
    """

    event_type: ClassVar[str] = "moss.channel.proxy.session.created"


# --- provider event --- #


class CreateSessionEvent(ChannelEventModel):
    """
    握手事件, provider 侧尝试与 proxy 进行握手, 确定 Session.
    """

    event_type: ClassVar[str] = "moss.channel.provider.session.create"


class CommandDoneEvent(ChannelEventModel):
    event_type: ClassVar[str] = "moss.channel.provider.command.done"
    chan: str = Field(description="channel name")
    command_id: str = Field(description="command id")
    errcode: int = Field(default=0, description="command errcode")
    errmsg: Optional[str] = Field(default=None, description="command errmsg")
    result: Any = Field(default=None, description="result of the command")


class ChannelMetaUpdateEvent(ChannelEventModel):
    event_type: ClassVar[str] = "moss.channel.meta.update"
    metas: dict[str, ChannelMeta] = Field(default_factory=dict, description="channel meta")
    root_chan: str = Field(description="channel name")
    all: bool = Field(default=True, description="是否更新了所有 channel")


class ProviderErrorEvent(ChannelEventModel):
    event_type: ClassVar[str] = "moss.channel.provider.error"
    errcode: int = Field(description="error code")
    errmsg: str = Field(description="error message")
