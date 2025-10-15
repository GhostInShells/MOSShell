from typing import TypedDict, Dict, Any, ClassVar, Optional, List
from typing_extensions import Self
from abc import ABC

from ghoshell_moss.concepts.channel import ChannelMeta
from ghoshell_moss.concepts.errors import CommandErrorCode
from ghoshell_common.helpers import uuid
from pydantic import BaseModel, Field
import time

__all__ = [
    'ChannelEvent', 'ChannelEventModel',
    'CommandPeekEvent', 'CommandCallEvent', 'CommandCancelEvent', 'CommandDoneEvent',
    'ChannelMetaUpdateEvent', 'SyncChannelMetasEvent',
    'PausePolicyDoneEvent', 'RunPolicyDoneEvent', 'PausePolicyEvent', 'RunPolicyEvent',
    'ClearCallEvent', 'ClearDoneEvent',
    'ServerErrorEvent',
    'HeartbeatEvent',
]


class ChannelEvent(TypedDict):
    event_id: str
    event_type: str
    session_id: Optional[str]
    timestamp: float
    data: Optional[Dict[str, Any]]


class ChannelEventModel(BaseModel, ABC):
    event_type: ClassVar[str] = ""

    event_id: str = Field(default_factory=uuid, description="event id for transport")
    session_id: str = Field(default="", description="channel client id")
    timestamp: float = Field(default_factory=lambda: round(time.time(), 4), description="timestamp")

    def to_channel_event(self) -> ChannelEvent:
        data = self.model_dump(exclude_none=True, exclude={'event_type', 'channel_id', 'channel_name', 'event_id'})
        return ChannelEvent(
            event_id=self.event_id,
            event_type=self.event_type,
            session_id=self.session_id,
            data=data,
            timestamp=self.timestamp,
        )

    @classmethod
    def from_channel_event(cls, channel_event: ChannelEvent) -> Optional[Self]:
        if cls.event_type != channel_event['event_type']:
            return None
        data = channel_event.get('data', {})
        data['event_id'] = channel_event['event_id']
        data['session_id'] = channel_event['session_id']
        data['timestamp'] = channel_event['timestamp']
        return cls(**data)


class HeartbeatEvent(ChannelEventModel):
    """心跳事件，由客户端发送，服务器响应"""
    event_type: ClassVar[str] = "moss.heartbeat"
    direction: str = Field(default="request", description="请求或响应: request/response")


# --- client event --- #


class RunPolicyEvent(ChannelEventModel):
    """开始运行 channel 的 policy"""
    event_type: ClassVar[str] = "moss.channel.client.policy.run"
    chan: str = Field(description="channel name")


class PausePolicyEvent(ChannelEventModel):
    """暂停某个 channel 的 policy 运行状态"""
    event_type: ClassVar[str] = "moss.channel.client.policy.pause"
    chan: str = Field(description="channel name")


class ClearCallEvent(ChannelEventModel):
    """发出讯号给某个 channel, 执行状态清空的逻辑"""
    event_type: ClassVar[str] = "moss.channel.client.clear.call"
    chan: str = Field(description="channel name")


class CommandCallEvent(ChannelEventModel):
    """发起一个 command 的调用. """

    # todo: 未来要加一个用 command_id 轮询 server 状态的事件. 用来避免通讯丢失.

    event_type: ClassVar[str] = "moss.channel.client.command.call"
    name: str = Field(description="command name")
    chan: str = Field(description="channel name")
    command_id: str = Field(default_factory=uuid, description="command id")
    args: List[Any] = Field(default_factory=list, description="command args")
    kwargs: Dict[str, Any] = Field(default_factory=dict, description="kwargs of the command")
    tokens: str = Field("", description="command tokens")
    context: Dict[str, Any] = Field(default_factory=dict, description="context of the command")

    def not_available(self, msg: str = "") -> "CommandDoneEvent":
        return CommandDoneEvent(
            session_id=self.session_id,
            command_id=self.command_id,
            errcode=CommandErrorCode.NOT_AVAILABLE.value,
            errmsg=msg or f"command `{self.chan}:{self.name}` not available",
            data=None,
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
            data=result,
            chan=self.chan,
        )

    def not_found(self, msg: str = "") -> "CommandDoneEvent":
        return CommandDoneEvent(
            session_id=self.session_id,
            command_id=self.command_id,
            errcode=CommandErrorCode.NOT_FOUND.value,
            errmsg=msg or f"command `{self.chan}:{self.name}` not found",
            chan=self.chan,
            data=None,
        )


class CommandPeekEvent(ChannelEventModel):
    event_type: ClassVar[str] = "moss.channel.client.command.peek"
    chan: str = Field(description="channel name")
    command_id: str = Field(description="command id")


class CommandCancelEvent(ChannelEventModel):
    """通知 channel 指定的 command 被取消. """
    event_type: ClassVar[str] = "moss.channel.client.command.cancel"
    chan: str = Field(description="channel name")
    command_id: str = Field(description="command id")


class SyncChannelMetasEvent(ChannelEventModel):
    """要求同步 channel 的 meta 信息. """
    event_type: ClassVar[str] = "moss.channel.meta.sync"
    channels: List[str] = Field(default_factory=list, description="channel names to sync")


# --- server event --- #

class CommandDoneEvent(ChannelEventModel):
    event_type: ClassVar[str] = "moss.channel.server.command.done"
    chan: str = Field(description="channel name")
    command_id: str = Field(description="command id")
    errcode: int = Field(default=0, description="command errcode")
    errmsg: Optional[str] = Field(default=None, description="command errmsg")
    data: Any = Field(description="result of the command")


class ClearDoneEvent(ChannelEventModel):
    event_type: ClassVar[str] = "moss.channel.server.clear.done"
    chan: str = Field(description="channel name")


class RunPolicyDoneEvent(ChannelEventModel):
    event_type: ClassVar[str] = "moss.channel.server.policy.run_done"


class PausePolicyDoneEvent(ChannelEventModel):
    event_type: ClassVar[str] = "moss.channel.server.policy.pause_done"
    chan: str = Field(description="channel name")


class ChannelMetaUpdateEvent(ChannelEventModel):
    event_type: ClassVar[str] = "moss.channel.meta.update"
    metas: List[ChannelMeta] = Field(default_factory=list, description="channel meta")
    root_chan: str = Field(description="channel name")


class ServerErrorEvent(ChannelEventModel):
    event_type: ClassVar[str] = "moss.channel.server.error"
    errcode: int = Field(description="error code")
    errmsg: str = Field(description="error message")
