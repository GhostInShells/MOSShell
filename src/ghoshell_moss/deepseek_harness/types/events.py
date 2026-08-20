"""
apiproxy 事件面: MuxFrame / HostFrame 两条 WS 下行流的帧联合 + 交互共享类型.

镜像 events.ts. MuxFrame 是 session 粒度的流 (session/event 帧包裹 SessionEvent,
其余是控制帧); HostFrame 是 host 级生命周期流, 不包裹 SessionEvent.

帧用「判别符 type + 全字段 permissive」建模 (同 session_events 的 StreamChunk 手法):
消费方按 `.type` 分派, 未知变体不崩 (type 为 str | Literal, extra="allow").
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from .session_events import Message, SessionEvent
from .nouns import JobView, WorkspaceView
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
    "MuxFrame",
    "HostFrame",
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


class MuxFrame(BaseModel):
    """session 粒度下行流帧. `type` 是判别符; 未知变体不崩, 消费方按 type 分派."""

    model_config = ConfigDict(extra="allow")

    type: str | Literal[
        "session/event",
        "session/subscribed",
        "approval/requested",
        "approval/resolved",
        "question/requested",
        "question/resolved",
        "session/queue",
        "session/jobs",
        "session/projection",
        "stream/error",
    ] = Field(default="session/event")

    # session/event
    sessionId: str = Field(default="")
    event: SessionEvent | None = Field(default=None)
    view: ToolEventView | None = Field(default=None)

    # session/subscribed
    lastSeq: int = Field(default=0)

    # approval/requested · resolved
    approvalId: str = Field(default="")
    toolName: str = Field(default="")
    callId: str | None = Field(default=None)
    reason: str | None = Field(default=None)
    outcome: ApprovalOutcome | str | None = Field(default=None)

    # question/requested · resolved
    questions: list[AskUserQuestionItem] | None = Field(default=None)
    questionRpcId: str = Field(default="")
    # question/resolved outcome: 'answered' | 'cancelled' (复用 outcome 槽)

    # session/queue
    items: list[QueuedInboxItem] | None = Field(default=None)

    # session/jobs
    jobs: list[JobView] | None = Field(default=None)

    # session/projection
    key: str = Field(default="")
    value: Any | None = Field(default=None)
    seq: int = Field(default=0)

    # stream/error
    error: RpcError | None = Field(default=None)


class HostFrame(BaseModel):
    """host 级生命周期流帧."""

    model_config = ConfigDict(extra="allow")

    type: str | Literal[
        "host/session-added",
        "host/session-removed",
        "host/session-status",
        "host/agent-error",
        "host/workspace-changed",
        "host/workspace-removed",
        "host/workspace-order-changed",
        "host/archived-sessions-changed",
        "host/remote-event",
        "stream/error",
    ] = Field(default="host/session-status")

    # session-added
    sessionId: str = Field(default="")
    blank: bool = Field(default=False)
    parentSessionId: str | None = Field(default=None)
    origin: str | None = Field(default=None, description="'subagent' 或空.")
    cwd: str | None = Field(default=None)
    agentPreset: str | None = Field(default=None)

    # session-status
    running: bool = Field(default=False)

    # agent-error
    message: str = Field(default="")

    # workspace-*
    workspace: WorkspaceView | None = Field(default=None)
    workspaceId: str = Field(default="")
    workspaceIds: list[str] | None = Field(default=None)
    archivedSessionIds: list[str] | None = Field(default=None)

    # remote-event
    event: str = Field(default="", description="host 自身 cordis event 名.")
    args: list[Any] | None = Field(default=None)

    # stream/error
    error: RpcError | None = Field(default=None)
