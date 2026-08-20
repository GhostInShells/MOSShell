"""
剩余 apiproxy 域: host / workspace / subagents / skills / goals / agent-presets /
settings / credentials / llm 的请求载荷与响应值类型.

workspace 的 WorkspaceView 与 jobs 的 JobView 已抽到 nouns.py (被 events 引用),
此处 workspace 域只声明动词 params/value. 品牌类型为 str.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from .session_events import ContentBlock
from .nouns import WorkspaceView
from .sessions import (
    HistoryEntry,
    ModelCatalogFailure,
    ModelProviderGroup,
    SessionProjectionsBlock,
)

__all__ = [
    # host
    "DirectoryEntry", "DirectoryListing",
    "HostDescribeValue", "HostPickDirectoryValue", "HostListDirectoryParams",
    "HostCreateDirectoryParams", "HostCreateDirectoryValue", "HostOpenPathParams", "HostOpenPathValue",
    # workspace
    "WorkspaceListValue", "WorkspaceCreateParams", "WorkspaceCreateValue",
    "WorkspaceRenameParams", "WorkspaceRenameValue", "WorkspaceDeleteParams", "WorkspaceDeleteValue",
    "WorkspaceInsertBeforeParams", "WorkspaceInsertBeforeValue",
    "WorkspaceInsertSessionBeforeParams", "WorkspaceInsertSessionBeforeValue",
    "WorkspaceArchiveSessionParams", "WorkspaceArchiveSessionValue",
    # subagents
    "SubagentListEntry", "SubagentPromptReceipt", "SubagentInterruptReceipt",
    "SubagentAddress", "SubagentCatalog",
    "SubagentListParams", "SubagentHistoryParams", "SubagentHistoryValue",
    "SubagentPromptParams", "SubagentInterruptParams",
    # skills
    "SkillEntry", "SkillListParams", "SkillListValue",
    # goals
    "GoalRef", "GoalCreateParams", "GoalCreateValue", "GoalEditParams", "GoalRefValue",
    "GoalClearValue",
    # agent-presets
    "AgentPresetEntry", "AgentPresetListValue", "AgentPresetSelectParams", "AgentPresetSelectValue",
    "AgentPresetReadParams", "AgentPresetReadValue", "AgentPresetCopyParams", "AgentPresetCopyValue",
    "AgentPresetOpenDocumentParams", "AgentPresetOpenDocumentValue", "AgentPresetRemoveParams",
    # settings
    "SettingsSecretView", "SettingsNamespaceView", "SettingsPathOpView",
    "SettingsDescribeValue", "SettingsOpenDocumentValue", "SettingsUpdateParams",
    "SettingsReplaceParams", "SettingsMutateParams",
    # credentials
    "CredentialView", "CredentialDescribeParams", "CredentialDescribeValue",
    "CredentialSetParams", "CredentialUnsetParams",
    # llm
    "ConfigurableProviderView", "DiscoveredModelView",
    "LlmProvidersValue", "LlmModelsValue", "LlmDiscoverModelsParams", "LlmDiscoverModelsValue",
]


# ---- host ---- #


class DirectoryEntry(BaseModel):
    model_config = ConfigDict(extra="allow")
    name: str = Field(default="")
    path: str = Field(default="")
    hidden: bool = Field(default=False)


class DirectoryListing(BaseModel):
    model_config = ConfigDict(extra="allow")
    path: str = Field(default="")
    home: str = Field(default="")
    crumbs: list[DirectoryEntry] = Field(default_factory=list)
    entries: list[DirectoryEntry] = Field(default_factory=list)
    truncated: bool = Field(default=False)


class HostDescribeValue(BaseModel):
    model_config = ConfigDict(extra="allow")
    version: str = Field(default="")
    cwd: str = Field(default="")
    provider: str | None = Field(default=None)
    model: str | None = Field(default=None)
    attachedSessions: int = Field(default=0)
    canOpenPath: bool = Field(default=False)


class HostPickDirectoryValue(BaseModel):
    model_config = ConfigDict(extra="allow")
    path: str | None = Field(default=None)


class HostListDirectoryParams(BaseModel):
    model_config = ConfigDict(extra="allow")
    path: str | None = Field(default=None)


class HostCreateDirectoryParams(BaseModel):
    model_config = ConfigDict(extra="allow")
    path: str = Field(default="")
    name: str = Field(default="")


class HostCreateDirectoryValue(BaseModel):
    model_config = ConfigDict(extra="allow")
    path: str = Field(default="")


class HostOpenPathParams(BaseModel):
    model_config = ConfigDict(extra="allow")
    path: str = Field(default="")


class HostOpenPathValue(BaseModel):
    model_config = ConfigDict(extra="allow")
    opened: bool = Field(default=True)


# ---- workspace ---- #


class WorkspaceListValue(BaseModel):
    model_config = ConfigDict(extra="allow")
    items: list[WorkspaceView] = Field(default_factory=list)
    archivedSessionIds: list[str] = Field(default_factory=list)


class WorkspaceCreateParams(BaseModel):
    model_config = ConfigDict(extra="allow")
    path: str = Field(default="")


class WorkspaceCreateValue(BaseModel):
    model_config = ConfigDict(extra="allow")
    workspace: WorkspaceView = Field(default_factory=WorkspaceView)
    created: bool = Field(default=False)


class WorkspaceRenameParams(BaseModel):
    model_config = ConfigDict(extra="allow")
    workspaceId: str = Field(default="")
    title: str = Field(default="")


class WorkspaceRenameValue(BaseModel):
    model_config = ConfigDict(extra="allow")
    workspace: WorkspaceView = Field(default_factory=WorkspaceView)


class WorkspaceDeleteParams(BaseModel):
    model_config = ConfigDict(extra="allow")
    workspaceId: str = Field(default="")


class WorkspaceDeleteValue(BaseModel):
    model_config = ConfigDict(extra="allow")
    deleted: bool = Field(default=True)


class WorkspaceInsertBeforeParams(BaseModel):
    model_config = ConfigDict(extra="allow")
    workspaceId: str = Field(default="")
    beforeWorkspaceId: str | None = Field(default=None)


class WorkspaceInsertBeforeValue(BaseModel):
    model_config = ConfigDict(extra="allow")
    workspaceIds: list[str] = Field(default_factory=list)


class WorkspaceInsertSessionBeforeParams(BaseModel):
    model_config = ConfigDict(extra="allow")
    workspaceId: str = Field(default="")
    sessionId: str = Field(default="")
    beforeSessionId: str | None = Field(default=None)


class WorkspaceInsertSessionBeforeValue(BaseModel):
    model_config = ConfigDict(extra="allow")
    workspace: WorkspaceView = Field(default_factory=WorkspaceView)


class WorkspaceArchiveSessionParams(BaseModel):
    model_config = ConfigDict(extra="allow")
    sessionId: str = Field(default="")


class WorkspaceArchiveSessionValue(BaseModel):
    model_config = ConfigDict(extra="allow")
    archivedSessionIds: list[str] = Field(default_factory=list)


# ---- subagents ---- #


class SubagentListEntry(BaseModel):
    model_config = ConfigDict(extra="allow")
    kind: str | Literal["child", "diagnostic"] = Field(default="child")
    id: str = Field(default="")
    activity: str | Literal["running", "inactive"] | None = Field(default=None)
    hasChildren: bool = Field(default=False)
    mode: str | Literal["one-shot", "continuable"] | None = Field(default=None)
    label: str | None = Field(default=None)
    reason: str | Literal["corrupt", "unsupported", "unavailable"] | None = Field(default=None)


class SubagentPromptReceipt(BaseModel):
    model_config = ConfigDict(extra="allow")
    messageId: str = Field(default="")


class SubagentInterruptReceipt(BaseModel):
    model_config = ConfigDict(extra="allow")
    accepted: bool = Field(default=True)


class SubagentAddress(BaseModel):
    model_config = ConfigDict(extra="allow")
    parentSessionId: str = Field(default="")
    childSessionId: str = Field(default="")
    mode: str | Literal["one-shot", "continuable"] = Field(default="continuable")


class SubagentCatalog(BaseModel):
    model_config = ConfigDict(extra="allow")
    entries: list[SubagentListEntry] = Field(default_factory=list)
    parentAvailable: bool = Field(default=False)


class SubagentListParams(BaseModel):
    model_config = ConfigDict(extra="allow")
    parentSessionId: str = Field(default="")


class SubagentHistoryParams(BaseModel):
    model_config = ConfigDict(extra="allow")
    parentSessionId: str = Field(default="")
    childSessionId: str = Field(default="")
    mode: str | Literal["one-shot", "continuable"] = Field(default="continuable")
    beforeSeq: int | None = Field(default=None)
    maxMessages: int | None = Field(default=None)


class SubagentHistoryValue(BaseModel):
    model_config = ConfigDict(extra="allow")
    events: list[HistoryEntry] = Field(default_factory=list)
    hasMore: bool = Field(default=False)
    projections: SessionProjectionsBlock | None = Field(default=None)


class SubagentPromptParams(BaseModel):
    model_config = ConfigDict(extra="allow")
    parentSessionId: str = Field(default="")
    childSessionId: str = Field(default="")
    mode: str | Literal["continuable"] = Field(default="continuable")
    content: list[ContentBlock] = Field(default_factory=list)
    clientTimeZone: str | None = Field(default=None)


class SubagentInterruptParams(BaseModel):
    model_config = ConfigDict(extra="allow")
    parentSessionId: str = Field(default="")
    childSessionId: str = Field(default="")
    mode: str | Literal["continuable"] = Field(default="continuable")


# ---- skills ---- #


class SkillEntry(BaseModel):
    model_config = ConfigDict(extra="allow")
    name: str = Field(default="")
    description: str = Field(default="")
    whenToUse: str | None = Field(default=None)
    modelInvocable: bool = Field(default=False)


class SkillListParams(BaseModel):
    model_config = ConfigDict(extra="allow")
    sessionId: str = Field(default="")


class SkillListValue(BaseModel):
    model_config = ConfigDict(extra="allow")
    skills: list[SkillEntry] = Field(default_factory=list)


# ---- goals ---- #


class GoalRef(BaseModel):
    model_config = ConfigDict(extra="allow")
    id: str = Field(default="")
    revision: int = Field(default=0)


class GoalCreateParams(BaseModel):
    model_config = ConfigDict(extra="allow")
    sessionId: str = Field(default="")
    objective: str = Field(default="")
    maxGoalRounds: int | None = Field(default=None)


class GoalCreateValue(BaseModel):
    model_config = ConfigDict(extra="allow")
    ref: GoalRef = Field(default_factory=GoalRef)


class GoalEditParams(BaseModel):
    model_config = ConfigDict(extra="allow")
    sessionId: str = Field(default="")
    ref: GoalRef = Field(default_factory=GoalRef)
    objective: str | None = Field(default=None)
    maxGoalRounds: int | None = Field(default=None)


class GoalRefValue(BaseModel):
    """goal.pause/resume/complete 的共享 value."""

    model_config = ConfigDict(extra="allow")
    ref: GoalRef = Field(default_factory=GoalRef)


class GoalClearValue(BaseModel):
    model_config = ConfigDict(extra="allow")
    cleared: bool = Field(default=True)


# ---- agent-presets ---- #


class AgentPresetEntry(BaseModel):
    model_config = ConfigDict(extra="allow")
    id: str = Field(default="")
    trust: str | Literal["system", "user"] = Field(default="system")
    isDefault: bool = Field(default=False)
    name: str | None = Field(default=None)
    description: str | None = Field(default=None)
    broken: str | None = Field(default=None)


class AgentPresetListValue(BaseModel):
    model_config = ConfigDict(extra="allow")
    presets: list[AgentPresetEntry] = Field(default_factory=list)
    authorable: bool = Field(default=False)
    hasDocument: bool = Field(default=False)


class AgentPresetSelectParams(BaseModel):
    model_config = ConfigDict(extra="allow")
    sessionId: str = Field(default="")
    agentPreset: str = Field(default="")


class AgentPresetSelectValue(BaseModel):
    model_config = ConfigDict(extra="allow")
    agentPreset: str = Field(default="")


class AgentPresetReadParams(BaseModel):
    model_config = ConfigDict(extra="allow")
    agentPreset: str = Field(default="")


class AgentPresetReadValue(BaseModel):
    model_config = ConfigDict(extra="allow")
    agentPreset: str = Field(default="")
    trust: str | Literal["system", "user"] = Field(default="system")
    content: str = Field(default="")
    name: str | None = Field(default=None)
    description: str | None = Field(default=None)


class AgentPresetCopyParams(BaseModel):
    model_config = ConfigDict(extra="allow")
    from_: str = Field(default="", alias="from")
    agentPreset: str = Field(default="")
    name: str | None = Field(default=None)


class AgentPresetCopyValue(BaseModel):
    model_config = ConfigDict(extra="allow")
    agentPreset: str = Field(default="")


class AgentPresetOpenDocumentParams(BaseModel):
    model_config = ConfigDict(extra="allow")
    agentPreset: str = Field(default="")


class AgentPresetOpenDocumentValue(BaseModel):
    model_config = ConfigDict(extra="allow")
    opened: bool = Field(default=False)
    path: str = Field(default="")


class AgentPresetRemoveParams(BaseModel):
    model_config = ConfigDict(extra="allow")
    agentPreset: str = Field(default="")


# ---- settings ---- #


class SettingsSecretView(BaseModel):
    model_config = ConfigDict(extra="allow")
    path: list[str] = Field(default_factory=list)
    set: bool = Field(default=False)


class SettingsNamespaceView(BaseModel):
    model_config = ConfigDict(extra="allow")
    ns: str = Field(default="")
    schema_: Any | None = Field(default=None, alias="schema")
    value: Any | None = Field(default=None)
    base: Any | None = Field(default=None)
    user: Any | None = Field(default=None)
    applies: str | Literal["live", "restart"] = Field(default="live")
    secrets: list[SettingsSecretView] = Field(default_factory=list)
    revision: int = Field(default=0)


class SettingsPathOpView(BaseModel):
    model_config = ConfigDict(extra="allow")
    op: str | Literal["set", "unset"] = Field(default="set")
    path: list[str] = Field(default_factory=list)
    value: Any | None = Field(default=None)


class SettingsDescribeValue(BaseModel):
    model_config = ConfigDict(extra="allow")
    writable: bool = Field(default=False)
    hasDocument: bool = Field(default=False)
    namespaces: list[SettingsNamespaceView] = Field(default_factory=list)


class SettingsOpenDocumentValue(BaseModel):
    model_config = ConfigDict(extra="allow")
    opened: bool = Field(default=True)


class SettingsUpdateParams(BaseModel):
    model_config = ConfigDict(extra="allow")
    ns: str = Field(default="")
    patch: dict[str, Any] = Field(default_factory=dict)
    expectedRevision: int | None = Field(default=None)


class SettingsReplaceParams(BaseModel):
    model_config = ConfigDict(extra="allow")
    ns: str = Field(default="")
    section: dict[str, Any] = Field(default_factory=dict)
    expectedRevision: int | None = Field(default=None)


class SettingsMutateParams(BaseModel):
    model_config = ConfigDict(extra="allow")
    ns: str = Field(default="")
    ops: list[SettingsPathOpView] = Field(default_factory=list)
    expectedRevision: int | None = Field(default=None)


# ---- credentials ---- #


class CredentialView(BaseModel):
    model_config = ConfigDict(extra="allow")
    configured: bool = Field(default=False)
    source: str | None = Field(default=None)
    writable: bool = Field(default=False)


class CredentialDescribeParams(BaseModel):
    model_config = ConfigDict(extra="allow")
    refs: list[str] = Field(default_factory=list)


class CredentialDescribeValue(BaseModel):
    model_config = ConfigDict(extra="allow")
    credentials: dict[str, CredentialView] = Field(default_factory=dict)


class CredentialSetParams(BaseModel):
    model_config = ConfigDict(extra="allow")
    ref: str = Field(default="")
    value: str = Field(default="")


class CredentialUnsetParams(BaseModel):
    model_config = ConfigDict(extra="allow")
    ref: str = Field(default="")


# ---- llm ---- #


class ConfigurableProviderView(BaseModel):
    model_config = ConfigDict(extra="allow")
    provider: str = Field(default="")
    displayName: str = Field(default="")
    settingsNs: str = Field(default="")
    settingsPath: list[str] = Field(default_factory=list)
    active: bool = Field(default=False)
    declared: bool | None = Field(default=None)


class DiscoveredModelView(BaseModel):
    model_config = ConfigDict(extra="allow")
    id: str = Field(default="")
    name: str | None = Field(default=None)
    contextWindow: int | None = Field(default=None)
    maxTokens: int | None = Field(default=None)


class LlmProvidersValue(BaseModel):
    model_config = ConfigDict(extra="allow")
    providers: list[ConfigurableProviderView] = Field(default_factory=list)


class LlmModelsValue(BaseModel):
    model_config = ConfigDict(extra="allow")
    groups: list[ModelProviderGroup] = Field(default_factory=list)
    failures: list[ModelCatalogFailure] = Field(default_factory=list)


class LlmDiscoverModelsParams(BaseModel):
    model_config = ConfigDict(extra="allow")
    settingsNs: str = Field(default="")
    provider: str | None = Field(default=None)
    baseURL: str | None = Field(default=None)
    api: str | None = Field(default=None)
    apiKey: str | None = Field(default=None)


class LlmDiscoverModelsValue(BaseModel):
    model_config = ConfigDict(extra="allow")
    models: list[DiscoveredModelView] = Field(default_factory=list)
