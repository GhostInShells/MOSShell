"""
session 域: apiproxy 的 session.* 12 个动词的请求载荷/响应值类型 + session 名词.

镜像 sessions.ts. 每个动词的 params (请求载荷) 与 value (响应值, 成功分支) 各建一个
模型; 值是裸名词时直接用名词. 品牌类型为 str.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from .session_events import ContentBlock, SessionEvent
from .events import ToolEventView

__all__ = [
    "SessionListMetadata",
    "SessionSummary",
    "SessionSearchItem",
    "HistoryEntry",
    "SessionProjectionsBlock",
    "PromptContentPart",
    "ModelSelection",
    "ModelReasoningEffort",
    "ModelReasoning",
    "ModelCatalogModel",
    "ModelProviderGroup",
    "ModelCatalogFailure",
    "SessionModels",
    "QueueAction",
    # 12 动词的 params/value
    "SessionListParams", "SessionListValue",
    "SessionSearchParams", "SessionSearchValue",
    "SessionCreateParams", "SessionCreateValue",
    "SessionHistoryParams", "SessionHistoryValue",
    "SessionModelsParams",
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
    """一个 session 列表项."""

    model_config = ConfigDict(extra="allow")

    sessionId: str = Field(default="")
    updatedAt: int = Field(default=0)
    running: bool = Field(default=False)
    blank: bool = Field(default=False)
    parentSessionId: str | None = Field(default=None)
    origin: str | None = Field(default=None)
    cwd: str | None = Field(default=None)
    agentPreset: str | None = Field(default=None)
    projections: "SessionProjectionsBlock | None" = Field(default=None)


class SessionSearchItem(BaseModel):
    model_config = ConfigDict(extra="allow")

    sessionId: str = Field(default="")
    snippet: str = Field(default="")


class HistoryEntry(BaseModel):
    """一个历史页条目: 原始事件 + 可选 render intent."""

    model_config = ConfigDict(extra="allow")

    event: SessionEvent = Field(default_factory=SessionEvent)
    view: ToolEventView | None = Field(default=None)


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


class SessionModels(BaseModel):
    """一个 session 的模型目录快照."""

    model_config = ConfigDict(extra="allow")

    current: ModelSelection = Field(default_factory=ModelSelection)
    routable: bool = Field(default=False)
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


class SessionSearchParams(BaseModel):
    model_config = ConfigDict(extra="allow")
    query: str = Field(default="")


class SessionSearchValue(BaseModel):
    model_config = ConfigDict(extra="allow")
    items: list[SessionSearchItem] = Field(default_factory=list)
    hasMore: bool = Field(default=False)


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


class SessionHistoryParams(BaseModel):
    model_config = ConfigDict(extra="allow")
    sessionId: str = Field(default="")
    beforeSeq: int | None = Field(default=None)
    maxMessages: int | None = Field(default=None)


class SessionHistoryValue(BaseModel):
    model_config = ConfigDict(extra="allow")
    events: list[HistoryEntry] = Field(default_factory=list)
    hasMore: bool = Field(default=False)
    projections: SessionProjectionsBlock | None = Field(default=None)


class SessionModelsParams(BaseModel):
    model_config = ConfigDict(extra="allow")
    sessionId: str = Field(default="")


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
