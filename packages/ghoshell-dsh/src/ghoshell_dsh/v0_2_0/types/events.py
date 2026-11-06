"""
dsh 0.1.5 远程流帧面: `/api/remote.mux` 传输帧 + `$events` 逻辑流下行帧.

镜像 dsh-api-gateway 的 stream-protocol.ts。一条物理 WS (`/api/remote.mux`) 多路复用
逻辑流: 客户端 `{type:'open', streamId, endpoint, payload}` 开流, 服务端回
`{type:'item', streamId, value}` / `{type:'error', ...}` / `{type:'end', ...}`。

`$events` 是应用级转发事件流 (endpoint `$events`, payload `{args:{}}`), 其 `item.value`
是四种下行帧: `ready`(绑定 clientId) / `emit`(单向, 位置参数 args) / `waterfall`(需回话)
/ `cancel`(取消 pending waterfall)。session 事件流 (`session/follow`) 另属后续增量。

帧用「判别符 type + 全字段 permissive」建模 (同 session_events 的 StreamChunk 手法):
消费方按 `.type` 分派, 未知变体不崩。
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from .session_events import Message
from .rpc import RpcError

__all__ = [
    "ApprovalOutcome",
    "ApprovalRequestId",
    "SubagentStopReason",
    "AskUserQuestionOption",
    "AskUserQuestionIntent",
    "AskUserQuestionItem",
    "AskUserQuestionAnswerItem",
    "AskUserQuestionAnswer",
    "ToolEventView",
    "QueuedInboxItem",
    "RemoteEventReady",
    "RemoteEventEmit",
    "RemoteEventWaterfall",
    "RemoteEventCancel",
    "RemoteEventDownlink",
    "RemoteStreamItem",
    "RemoteStreamError",
    "RemoteStreamEnd",
]

ApprovalRequestId = str
ApprovalOutcome = Literal["allowed-once", "rejected", "cancelled", "unavailable"]
SubagentStopReason = Literal["completed", "aborted", "error", "max-tokens", "refusal"]


class AskUserQuestionOption(BaseModel):
    model_config = ConfigDict(extra="allow")

    label: str = Field(default="")
    description: str | None = Field(default=None)


class AskUserQuestionIntent(BaseModel):
    """caller 声明的呈现意图: 只改呈现, 不改协议."""

    model_config = ConfigDict(extra="allow")

    kind: str | Literal["plan-review"] = Field(default="plan-review")
    approve: str = Field(default="")


class AskUserQuestionItem(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: str = Field(default="")
    question: str = Field(default="")
    detail: str | None = Field(default=None)
    header: str | None = Field(default=None)
    options: list[AskUserQuestionOption] | None = Field(default=None)
    multiSelect: bool | None = Field(default=None)
    intent: AskUserQuestionIntent | None = Field(default=None)


class AskUserQuestionAnswerItem(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: str = Field(default="")
    selected: list[str] = Field(default_factory=list)
    custom: str | None = Field(default=None)


class AskUserQuestionAnswer(BaseModel):
    model_config = ConfigDict(extra="allow")

    answers: list[AskUserQuestionAnswerItem] = Field(default_factory=list)


class ToolEventView(BaseModel):
    """host 在 emit 时算出的 render intent, 永不持久化. `view` 是不透明呈现载荷."""

    model_config = ConfigDict(extra="allow")

    for_: str | Literal["call", "result"] = Field(default="call", alias="for")
    view: dict[str, Any] | None = Field(default=None)


class QueuedInboxItem(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: str = Field(default="")
    placement: str | Literal["queued", "steering", "context"] = Field(default="queued")
    message: Message = Field(default_factory=Message)


class RemoteEventReady(BaseModel):
    """`$events` 逻辑流首帧: 绑定 clientId (后续 `$events/result` 回话凭据) + host 事实."""

    model_config = ConfigDict(extra="allow")

    type: Literal["ready"] = "ready"
    clientId: str = Field(default="")
    host: dict[str, Any] = Field(default_factory=dict, description="{home: str}")


class RemoteEventEmit(BaseModel):
    """`$events` 单向通知帧: 应用级 cordis event, args 为位置参数."""

    model_config = ConfigDict(extra="allow")

    type: Literal["emit"] = "emit"
    event: str = Field(default="")
    args: list[Any] = Field(default_factory=list)


class RemoteEventWaterfall(BaseModel):
    """`$events` 需回话帧: eventId 关联 `$events/result`, request 为 JSON-safe 载荷."""

    model_config = ConfigDict(extra="allow")

    type: Literal["waterfall"] = "waterfall"
    event: str = Field(default="")
    eventId: str = Field(default="")
    agentId: str = Field(default="")
    request: dict[str, Any] = Field(default_factory=dict)


class RemoteEventCancel(BaseModel):
    """取消一个 pending waterfall (同一 eventId)."""

    model_config = ConfigDict(extra="allow")

    type: Literal["cancel"] = "cancel"
    eventId: str = Field(default="")


# `$events` 逻辑流 item.value 的判别联合.
RemoteEventDownlink = RemoteEventReady | RemoteEventEmit | RemoteEventWaterfall | RemoteEventCancel


class RemoteStreamItem(BaseModel):
    """mux 下行 `item` 帧: 一条逻辑流的一个 value."""

    model_config = ConfigDict(extra="allow")

    type: Literal["item"] = "item"
    streamId: str = Field(default="")
    value: Any = Field(default=None, description="$events 下行帧 (或后续 session/follow 帧).")


class RemoteStreamError(BaseModel):
    """mux 下行 `error` 帧: 逻辑流失败."""

    model_config = ConfigDict(extra="allow")

    type: Literal["error"] = "error"
    streamId: str = Field(default="")
    error: RpcError | None = Field(default=None)


class RemoteStreamEnd(BaseModel):
    """mux 下行 `end` 帧: 逻辑流正常结束."""

    model_config = ConfigDict(extra="allow")

    type: Literal["end"] = "end"
    streamId: str = Field(default="")
