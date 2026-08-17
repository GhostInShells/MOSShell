"""
SDK JSON-RPC 协议类型 (sdk/protocol/src/types.ts).

三对请求/结果 + 四个 server→client 通知载荷. 独立于 apiproxy 的 web 面,
是 dsh SDK 进程外 stdio 协议 (serverInfo.name 恒为 deepseek-harness-sdk-runtime).

类型名加 `Sdk` 前缀, 与 apiproxy 面 (session.prompt 的 SessionPromptParams) 消歧.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from .session_events import ContentBlock, SessionEvent
from .events import SubagentStopReason

__all__ = [
    "SdkRunStatus",
    "SdkServerInfo",
    "SdkInitializeParams",
    "SdkInitializeResult",
    "SdkSessionPromptParams",
    "SdkSessionPromptResult",
    "SdkSessionEventNotification",
    "SdkSessionStatusNotification",
    "SdkSubagentStartedNotification",
    "SdkSubagentFinishedNotification",
]

SdkRunStatus = Literal["ok", "error"]


class SdkServerInfo(BaseModel):
    model_config = ConfigDict(extra="allow")
    name: str = Field(default="deepseek-harness-sdk-runtime")
    version: str = Field(default="")


class SdkInitializeParams(BaseModel):
    model_config = ConfigDict(extra="allow")
    cwd: str = Field(default="")
    provider: str = Field(default="")
    model: str = Field(default="")
    maxTokens: int | None = Field(default=None)


class SdkInitializeResult(BaseModel):
    model_config = ConfigDict(extra="allow")
    serverInfo: SdkServerInfo = Field(default_factory=SdkServerInfo)


class SdkSessionPromptParams(BaseModel):
    model_config = ConfigDict(extra="allow")
    sessionId: str = Field(default="")
    contentBlocks: list[ContentBlock] = Field(default_factory=list)


class SdkSessionPromptResult(BaseModel):
    model_config = ConfigDict(extra="allow")
    messageId: str = Field(default="")


class SdkSessionEventNotification(BaseModel):
    model_config = ConfigDict(extra="allow")
    sessionId: str = Field(default="")
    event: SessionEvent = Field(default_factory=SessionEvent)


class SdkSessionStatusNotification(BaseModel):
    model_config = ConfigDict(extra="allow")
    sessionId: str = Field(default="")
    status: str | Literal["idle", "running"] = Field(default="idle")


class SdkSubagentStartedNotification(BaseModel):
    model_config = ConfigDict(extra="allow")
    parentSessionId: str = Field(default="")
    childSessionId: str = Field(default="")


class SdkSubagentFinishedNotification(BaseModel):
    model_config = ConfigDict(extra="allow")
    provider: str = Field(default="")
    agentId: str = Field(default="")
    parentSessionId: str = Field(default="")
    childSessionId: str = Field(default="")
    status: SdkRunStatus | str = Field(default="ok")
    stopReason: SubagentStopReason | str = Field(default="completed")
    lastAssistantMessage: list[ContentBlock] | None = Field(default=None)
