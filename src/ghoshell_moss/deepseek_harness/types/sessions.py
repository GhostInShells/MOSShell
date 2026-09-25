"""
session 域: dsh 0.1.5 Remote 的 session.* 动词的请求载荷/响应值类型 + session 名词.

镜像 sessions.ts. 每个动词的 params (请求载荷) 与 value (响应值, 成功分支) 各建一个
模型; 值是裸名词时直接用名词. 品牌类型为 str.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from ghoshell_moss.message import unique_id

from .session_events import ContentBlock

__all__ = [
    "SessionListMetadata",
    "SessionSummary",
    "SessionProjectionsBlock",
    "PromptContentPart",
    "ModelSelection",
    "ModelReasoningEffort",
    "ModelReasoning",
    "ModelCatalogModel",
    "ModelProviderGroup",
    "ModelCatalogFailure",
    "ModelCatalog",
    "QueueAction",
    # 会话动词 params/value
    "SessionListParams", "SessionListValue",
    "SessionCreateParams", "SessionCreateValue",
    "SessionAddress", "SessionPageParams", "SessionPageRecord", "SessionPageValue",
    "SessionSelectModelParams", "SessionSelectModelValue",
    "SessionRenameParams", "SessionRenameValue",
    "SessionForkParams", "SessionForkValue",
    "SessionPromptParams", "SessionPromptValue",
    "SessionAttachmentParams", "SessionAttachmentValue",
    "SessionUpdateQueueParams", "SessionUpdateQueueValue",
    "SessionCancelParams", "SessionCancelValue",
]


class SessionListMetadata(BaseModel):
    model_config = ConfigDict(extra="allow")

    blank: bool = Field(default=False)
    lastPromptAt: int | None = Field(default=None)


class SessionSummary(BaseModel):
    """一个 session 列表项 (session/list 的 value.item). agentPreset 不在顶层 —
    它随 follow 快照 header.agentPreset 或 projections.values.agentPreset 到达."""

    model_config = ConfigDict(extra="allow")

    sessionId: str = Field(default="")
    updatedAt: int = Field(default=0)
    running: bool = Field(default=False)
    blank: bool = Field(default=False)
    parentSessionId: str | None = Field(default=None)
    origin: str | None = Field(default=None)
    cwd: str | None = Field(default=None)
    projections: "SessionProjectionsBlock | None" = Field(default=None)


class SessionProjectionsBlock(BaseModel):
    """历史尾页携带的 projection 基线."""

    model_config = ConfigDict(extra="allow")

    asOfSeq: int = Field(default=-1)
    values: dict[str, Any] = Field(default_factory=dict)


class PromptContentPart(BaseModel):
    """浏览器提交的 prompt 内容 (image 为 base64, host 再提升为 attachment ref)."""

    model_config = ConfigDict(extra="allow")

    type: str | Literal["text", "image"] = Field(default="text")
    text: str = Field(default="")
    mediaType: str | None = Field(default=None)
    data: str = Field(default="", description="image 的 base64 字节.")
    name: str | None = Field(default=None)


class ModelSelection(BaseModel):
    model_config = ConfigDict(extra="allow")

    provider: str = Field(default="")
    model: str = Field(default="")
    reasoningEffort: str | None = Field(default=None)


class ModelReasoningEffort(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: str = Field(default="")
    name: str = Field(default="")
    description: str | None = Field(default=None)


class ModelReasoning(BaseModel):
    model_config = ConfigDict(extra="allow")

    efforts: list[ModelReasoningEffort] = Field(default_factory=list)
    defaultEffort: str | None = Field(default=None)


class ModelCatalogModel(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: str = Field(default="")
    name: str = Field(default="")
    description: str | None = Field(default=None)
    reasoning: ModelReasoning | None = Field(default=None)


class ModelProviderGroup(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: str = Field(default="")
    name: str = Field(default="")
    models: list[ModelCatalogModel] = Field(default_factory=list)


class ModelCatalogFailure(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: str = Field(default="")
    name: str = Field(default="")
    message: str = Field(default="")


class ModelCatalog(BaseModel):
    """session/modelCatalog 的浏览器模型目录 — default + routableProviders + groups + failures.

    ``routableProviders`` 是「可路由的 provider id 列表」, 取代旧 session.models 的
    ``routable`` 布尔 (后者意为「当前路由可服务」).
    """

    model_config = ConfigDict(extra="allow")

    default: ModelSelection = Field(default_factory=ModelSelection)
    routableProviders: list[str] = Field(default_factory=list)
    groups: list[ModelProviderGroup] = Field(default_factory=list)
    failures: list[ModelCatalogFailure] = Field(default_factory=list)


class QueueAction(BaseModel):
    model_config = ConfigDict(extra="allow")

    kind: str | Literal["edit", "remove", "steer"] = Field(default="remove")
    content: list[ContentBlock] | None = Field(default=None)


# ---- session.* 12 动词 params / value ---- #


class SessionListParams(BaseModel):
    model_config = ConfigDict(extra="allow")
    cursor: str | None = Field(default=None)


class SessionListValue(BaseModel):
    model_config = ConfigDict(extra="allow")
    items: list[SessionSummary] = Field(default_factory=list)


class SessionCreateParams(BaseModel):
    model_config = ConfigDict(extra="allow")
    workspaceId: str | None = Field(default=None)
    cwd: str | None = Field(default=None)
    sessionId: str | None = Field(default=None)
    agentPreset: str | None = Field(default=None)


class SessionCreateValue(BaseModel):
    model_config = ConfigDict(extra="allow")
    sessionId: str = Field(default="")
    agentPreset: str | None = Field(default=None)


class SessionAddress(BaseModel):
    """session/page 的 durable 地址 (kind=session)."""

    model_config = ConfigDict(extra="allow")

    kind: Literal["session"] = Field(default="session")
    sessionId: str = Field(default="")


class SessionPageParams(BaseModel):
    """session/page 的向后历史分页请求.

    ``throughSeq`` 是「inclusive log cut」— 来自 follow 开流快照的 cursor; 分页从它
    向后 (更早) 读.
    """

    model_config = ConfigDict(extra="allow")

    address: SessionAddress = Field(default_factory=SessionAddress)
    throughSeq: int = Field(default=0)
    beforeSeq: int | None = Field(default=None)
    maxMessages: int | None = Field(default=None)


class SessionPageRecord(BaseModel):
    """session/page 的一条记录: 包裹一个 flat wire 事件 (经 SessionEvent.from_dict 解析)."""

    model_config = ConfigDict(extra="allow")

    type: str = Field(default="event")
    event: dict = Field(default_factory=dict)


class SessionPageValue(BaseModel):
    """session/page 的一页: records + hasMore."""

    model_config = ConfigDict(extra="allow")

    records: list[SessionPageRecord] = Field(default_factory=list)
    hasMore: bool = Field(default=False)


class SessionSelectModelParams(BaseModel):
    model_config = ConfigDict(extra="allow")
    sessionId: str = Field(default="")
    provider: str = Field(default="")
    model: str = Field(default="")
    reasoningEffort: str | None = Field(default=None)


class SessionSelectModelValue(BaseModel):
    model_config = ConfigDict(extra="allow")
    selected: ModelSelection = Field(default_factory=ModelSelection)


class SessionRenameParams(BaseModel):
    model_config = ConfigDict(extra="allow")
    sessionId: str = Field(default="")
    title: str = Field(default="")


class SessionRenameValue(BaseModel):
    model_config = ConfigDict(extra="allow")
    title: str = Field(default="")
    seq: int = Field(default=0)


class SessionForkParams(BaseModel):
    model_config = ConfigDict(extra="allow")
    sessionId: str = Field(default="")
    atSeq: int | None = Field(default=None)


class SessionForkValue(BaseModel):
    model_config = ConfigDict(extra="allow")
    sessionId: str = Field(default="")


class SessionPromptParams(BaseModel):
    model_config = ConfigDict(extra="allow")
    requestId: str = Field(default_factory=unique_id, description="client-minted identity persisted on the accepted user message.")
    sessionId: str = Field(default="")
    mode: str | Literal["queue", "steer"] = Field(default="queue")
    content: list[PromptContentPart] = Field(default_factory=list)
    clientTimeZone: str | None = Field(default=None)


class SessionPromptValue(BaseModel):
    model_config = ConfigDict(extra="allow")
    accepted: bool = Field(default=True)
    command: dict[str, Any] | None = Field(default=None, description="slash command 结果槽.")


class SessionAttachmentParams(BaseModel):
    model_config = ConfigDict(extra="allow")
    sessionId: str = Field(default="")
    attachmentId: str = Field(default="")


class SessionAttachmentValue(BaseModel):
    model_config = ConfigDict(extra="allow")
    attachment: dict[str, Any] = Field(default_factory=dict, description="ImageAttachmentRef, 不透明.")
    data: str = Field(default="")


class SessionUpdateQueueParams(BaseModel):
    model_config = ConfigDict(extra="allow")
    sessionId: str = Field(default="")
    itemId: str = Field(default="")
    action: QueueAction = Field(default_factory=QueueAction)


class SessionUpdateQueueValue(BaseModel):
    model_config = ConfigDict(extra="allow")
    accepted: bool = Field(default=True)


class SessionCancelParams(BaseModel):
    model_config = ConfigDict(extra="allow")
    sessionId: str = Field(default="")


class SessionCancelValue(BaseModel):
    model_config = ConfigDict(extra="allow")
    accepted: bool = Field(default=True)
