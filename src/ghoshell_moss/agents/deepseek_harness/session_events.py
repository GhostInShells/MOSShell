"""
dsh session events — DeepSeek Harness 会话事件信封容器与强类型封装.

两层结构, 镜像 `core/concepts/topic.py` 的 Topic / TopicMeta / TopicModel:

- `SessionEvent`(对应 Topic): 信封容器, `meta` + `data`(裸 dict), 不会因未知事件炸.
- `SessionEventMeta`(对应 TopicMeta): 信封上可复用的元信息, 不逐事件复制.
- `SessionEventModel`(对应 TopicModel): 具体事件的强类型基类. 子类声明
  `event_type()` 判别符; `from_session_event(event)` 先按事件名判断, 名字不符直接
  返回 None 不碰 data, 名字匹配才 `model_validate(data)` 解析成强类型字段.

所有 merge-extensible 判别(type/kind/role/status)均用 `str | Literal`, 支持 plugin
扩展: 遇到未知变体不会 ValidationError, 交给消费方按需处理.

事件来源于 dsh `packages/core/session/src/types.ts` 的 `SessionEventMap`(13 种),
载荷类型取自 `packages/llm/llm/src/message.ts` 与 `types.ts`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Literal, Union

from pydantic import BaseModel, ConfigDict, Field
from typing_extensions import Self

__all__ = [
    "SessionEventMeta",
    "SessionEvent",
    "SessionEventModel",
    # 判别联合与支撑类型
    "SurfaceReplace",
    "SurfaceOp",
    "ContentBlock",
    "ContextSnapshotSection",
    "TokenUsage",
    "LlmFailure",
    "FinishReason",
    "MessageSource",
    "StreamChunk",
    "Message",
    "TodoItem",
    "AgentCancelCause",
    "TurnEndReason",
    "LlmCallConfig",
    "LlmCallConfigAdapterDefaults",
    "ToolSchema",
    "EpochHeader",
    "RequestContext",
    "ToolResultError",
    # 13 个具体事件
    "TurnStart",
    "TurnEnd",
    "StepStart",
    "StepEnd",
    "UserMessageEvent",
    "AssistantChunk",
    "AssistantMessageEvent",
    "ToolCallEvent",
    "ToolResultEvent",
    "TodoWrite",
    "RequestHeader",
    "RequestContextEvent",
    "SessionEndSeed",
]


class SurfaceReplace(BaseModel):
    """surfaceOp 的 replace 形态: 用此节点替换 [start, end] 区间内的 surface 节点."""

    model_config = ConfigDict(extra="allow")

    op: Literal["replace"] = Field(default="replace")
    start: int = Field(default=0)
    end: int = Field(default=0)


# 'append' 或 replace 结构; 容忍未知字符串以便向前兼容.
SurfaceOp = Union[str, SurfaceReplace]


class SessionEventMeta(BaseModel):
    """信封上可复用的元信息(对应 TopicMeta). 具体事件持有引用, 不逐事件复制."""

    model_config = ConfigDict(extra="allow")

    type: str = Field(default="", description="事件类型, 即判别符.")
    seq: int = Field(default=0, description="会话内单调递增序号.")
    time: int = Field(default=0, description="Unix epoch 毫秒.")
    ignorable: bool | None = Field(
        default=None,
        description="为 true 时, 遇到未知 type 的 reader 可安全跳过该事件.",
    )
    surfaceOp: SurfaceOp | None = Field(
        default=None,
        description="仅 surface 事件(user/message, assistant/message, tool/result)携带.",
    )
    sourceEventSeqs: list[int] | None = Field(
        default=None,
        description="surface 事件引用的源事件 seq 列表.",
    )


class SessionEvent(BaseModel):
    """信封容器(对应 Topic). 承载完整事件, `data` 保持裸 dict 不解析."""

    model_config = ConfigDict(extra="allow")

    meta: SessionEventMeta = Field(default_factory=SessionEventMeta)
    data: dict = Field(default_factory=dict, description="类型化载荷, 原样保留.")

    @classmethod
    def from_dict(cls, d: dict) -> "SessionEvent":
        """从 dsh 扁平信封 dict 构造容器."""
        meta = SessionEventMeta(
            type=d.get("type", ""),
            seq=d.get("seq", 0),
            time=d.get("time", 0),
            ignorable=d.get("ignorable"),
            surfaceOp=d.get("surfaceOp"),
            sourceEventSeqs=d.get("sourceEventSeqs"),
        )
        return cls(meta=meta, data=d.get("data", {}) or {})

    def to_dict(self) -> dict:
        """等价返回原始扁平信封 dict."""
        d: dict = {
            "type": self.meta.type,
            "seq": self.meta.seq,
            "time": self.meta.time,
            "data": self.data,
        }
        if self.meta.ignorable is not None:
            d["ignorable"] = self.meta.ignorable
        if self.meta.surfaceOp is not None:
            d["surfaceOp"] = self.meta.surfaceOp
        if self.meta.sourceEventSeqs is not None:
            d["sourceEventSeqs"] = self.meta.sourceEventSeqs
        return d


class SessionEventModel(BaseModel, ABC):
    """具体事件的强类型基类(对应 TopicModel). 子类声明 `event_type()` 判别符."""

    model_config = ConfigDict(extra="allow")

    meta: SessionEventMeta = Field(default_factory=SessionEventMeta)

    @classmethod
    @abstractmethod
    def event_type(cls) -> str:
        """定义事件的判别符. 对于从 `SessionEvent` 还原具体事件的场景, 依赖它做分派."""
        pass

    @classmethod
    def from_session_event(cls, event: SessionEvent) -> Self | None:
        """先按事件名判断, 名字不符返回 None 不解析 data; 匹配才 `model_validate(data)`."""
        if event.meta.type != cls.event_type():
            return None
        obj = cls.model_validate(event.data)
        obj.meta = event.meta
        return obj

    def to_session_event(self) -> SessionEvent:
        """把强类型载荷装回信封容器(meta + data)."""
        data = self.model_dump(exclude={"meta"}, exclude_none=True)
        return SessionEvent(meta=self.meta, data=data)

    def to_dict(self) -> dict:
        """等价返回原始扁平信封 dict."""
        return self.to_session_event().to_dict()

    @property
    def seq(self) -> int:
        return self.meta.seq

    @property
    def time(self) -> int:
        return self.meta.time

    @property
    def type(self) -> str:
        return self.meta.type


# ---- 判别联合与支撑类型 ---- #


class ContentBlock(BaseModel):
    """merge-extensible 内容块. `type` 用 `str | Literal` 支持 plugin 扩展."""

    model_config = ConfigDict(extra="allow")

    type: str | Literal["text", "reasoning", "image", "tool-call", "tool-result"] = Field(
        default="text",
        description="内容块类型判别符.",
    )
    text: str | None = None
    # image
    attachment: dict[str, Any] | None = None
    # tool-call
    id: str | None = None
    name: str | None = None
    arguments: str | None = None
    # tool-result
    toolCallId: str | None = None
    content: list["ContentBlock"] | None = None
    isError: bool | None = None


class ContextSnapshotSection(BaseModel):
    """snapshot-form 上下文里命名分节的一项."""

    model_config = ConfigDict(extra="allow")

    name: str = Field(default="")
    text: str = Field(default="")


class TokenUsage(BaseModel):
    """一次模型调用的 token 记账. 计数 DISJOINT: input 为未缓存输入."""

    model_config = ConfigDict(extra="allow")

    inputTokens: int = 0
    outputTokens: int = 0
    cacheReadTokens: int | None = None
    cacheWriteTokens: int | None = None
    reasoningTokens: int | None = None


class LlmFailure(BaseModel):
    """可序列化的 provider/transport 失败事实."""

    model_config = ConfigDict(extra="allow")

    message: str = Field(default="")
    code: str = Field(default="")
    status: int | None = None
    providerRetryAfterMs: int | None = None
    requestId: str | None = None


class FinishReason(BaseModel):
    """模型响应为何停止. merge-extensible."""

    model_config = ConfigDict(extra="allow")

    kind: str | Literal["stop", "tool-calls", "max-tokens", "aborted", "error"] = Field(
        default="stop",
    )
    failure: LlmFailure | None = None


class MessageSource(BaseModel):
    """消息(或被注入内容)从哪来. merge-extensible kind."""

    model_config = ConfigDict(extra="allow")

    kind: str | Literal["user", "plugin", "model", "tool"] = Field(default="user")
    # plugin kind + ContextFormed
    plugin: str | None = None
    form: str | None = None
    sections: list[ContextSnapshotSection] | None = None
    summary: str | None = None
    # model kind
    provider: str | None = None
    model: str | None = None
    replayState: Any | None = None
    # tool kind
    callId: str | None = None


class StreamChunk(BaseModel):
    """adapter 发出的原始流块. merge-extensible type."""

    model_config = ConfigDict(extra="allow")

    type: str | Literal[
        "block-start",
        "text-delta",
        "reasoning-delta",
        "tool-call-delta",
        "block-end",
        "usage",
        "finish",
    ] = Field(default="text-delta")
    index: int | None = None
    blockType: str | None = None
    text: str | None = None
    id: str | None = None
    name: str | None = None
    argumentsDelta: str | None = None
    block: ContentBlock | None = None
    usage: TokenUsage | None = None
    reason: FinishReason | None = None
    replayState: Any | None = None


class Message(BaseModel):
    """统一消息表示, 用于 user/assistant/tool-result 三种载荷."""

    model_config = ConfigDict(extra="allow")

    id: str = Field(default="")
    role: str | Literal["system", "user", "assistant"] = Field(default="user")
    content: list[ContentBlock] = Field(default_factory=list)
    source: MessageSource = Field(default_factory=MessageSource)


class TodoItem(BaseModel):
    """agent todo 列表的一项, 整表 last-write-wins."""

    model_config = ConfigDict(extra="allow")

    content: str = Field(default="")
    status: str | Literal["pending", "in_progress", "completed"] = Field(default="pending")


class AgentCancelCause(BaseModel):
    """为何取消一个活跃 agent driver."""

    model_config = ConfigDict(extra="allow")

    kind: str | Literal["user", "parent", "hook", "disposed", "legacy"] = Field(default="user")
    reason: str | None = None


class TurnEndReason(BaseModel):
    """一个 turn 为何结束. merge-extensible kind."""

    model_config = ConfigDict(extra="allow")

    kind: str | Literal["completed", "aborted", "blocked", "error", "max-tokens", "interrupted"] = Field(
        default="completed",
    )
    reason: AgentCancelCause | None = None
    error: LlmFailure | None = None


class LlmCallConfig(BaseModel):
    """一次请求的调用配置."""

    model_config = ConfigDict(extra="allow")

    provider: str = Field(default="")
    model: str = Field(default="")
    reasoningEffort: str | None = None
    temperature: float | None = None
    maxTokens: int | None = None
    stop: list[str] | None = None


class LlmCallConfigAdapterDefaults(BaseModel):
    """exact-model adapter 解析供应的有效配置字段."""

    model_config = ConfigDict(extra="allow")

    reasoningEffort: bool | None = None
    maxTokens: bool | None = None


class ToolSchema(BaseModel):
    """发给模型的工具 JSON-schema 描述."""

    model_config = ConfigDict(extra="allow")

    name: str = Field(default="")
    description: str = Field(default="")
    parameters: dict[str, Any] = Field(default_factory=dict)


class EpochHeader(BaseModel):
    """request/header 的快照: 调用配置, system prompt, tools."""

    model_config = ConfigDict(extra="allow")

    config: LlmCallConfig = Field(default_factory=LlmCallConfig)
    adapterDefaults: LlmCallConfigAdapterDefaults | None = None
    system: str | None = None
    tools: list[ToolSchema] | None = None


class RequestContext(BaseModel):
    """一条已解析模型路由的注册绑定元信息."""

    model_config = ConfigDict(extra="allow")

    provider: str = Field(default="")
    model: str = Field(default="")
    contextWindow: int | None = None


class ToolResultError(BaseModel):
    """tool/result 载荷里的可选内部失败标识."""

    model_config = ConfigDict(extra="allow")

    name: str = Field(default="")
    code: str = Field(default="")


# ---- 13 个具体事件 ---- #


class TurnStart(SessionEventModel):
    turn: int = Field(default=0)

    @classmethod
    def event_type(cls) -> str:
        return "turn/start"


class TurnEnd(SessionEventModel):
    turn: int = Field(default=0)
    reason: TurnEndReason = Field(default_factory=TurnEndReason)

    @classmethod
    def event_type(cls) -> str:
        return "turn/end"


class StepStart(SessionEventModel):
    turn: int = Field(default=0)
    step: int = Field(default=0)

    @classmethod
    def event_type(cls) -> str:
        return "step/start"


class StepEnd(SessionEventModel):
    turn: int = Field(default=0)
    step: int = Field(default=0)

    @classmethod
    def event_type(cls) -> str:
        return "step/end"


class UserMessageEvent(SessionEventModel):
    """user/message — data 即一个 UserMessage."""

    id: str = Field(default="")
    role: str | Literal["user"] = Field(default="user")
    content: list[ContentBlock] = Field(default_factory=list)
    source: MessageSource = Field(default_factory=MessageSource)

    @classmethod
    def event_type(cls) -> str:
        return "user/message"

    def text(self) -> str:
        """拼接全部 text 块的可见文本."""
        return "".join(b.text or "" for b in self.content if b.type == "text")


class AssistantChunk(SessionEventModel):
    """assistant/chunk — 原始流块."""

    turn: int = Field(default=0)
    step: int = Field(default=0)
    chunk: StreamChunk = Field(default_factory=StreamChunk)

    @classmethod
    def event_type(cls) -> str:
        return "assistant/chunk"


class AssistantMessageEvent(SessionEventModel):
    """assistant/message — 组装完成的 assistant 消息 + 可选 usage."""

    turn: int = Field(default=0)
    step: int = Field(default=0)
    message: Message = Field(default_factory=Message)
    usage: TokenUsage | None = None

    @classmethod
    def event_type(cls) -> str:
        return "assistant/message"


class ToolCallEvent(SessionEventModel):
    """tool/call — 模型请求一次工具调用."""

    turn: int = Field(default=0)
    step: int = Field(default=0)
    callId: str = Field(default="")
    name: str = Field(default="")
    arguments: str = Field(default="")

    @classmethod
    def event_type(cls) -> str:
        return "tool/call"


class ToolResultEvent(SessionEventModel):
    """tool/result — 一次工具调用的完成结果.

    dsh 载荷的 `data.meta`(工具私有表现载荷)与信封 `meta` 同名, 故改名为
    `tool_meta` 并在此 override 搬运, 不污染基类的信封 meta.
    """

    turn: int = Field(default=0)
    step: int = Field(default=0)
    message: Message = Field(default_factory=Message)
    error: ToolResultError | None = None
    tool_meta: Any | None = Field(
        default=None,
        description="工具私有表现载荷(data.meta), 对 core 不透明.",
    )

    @classmethod
    def event_type(cls) -> str:
        return "tool/result"

    @classmethod
    def from_session_event(cls, event: SessionEvent) -> Self | None:
        if event.meta.type != cls.event_type():
            return None
        data = dict(event.data)
        tool_meta = data.pop("meta", None)
        obj = cls.model_validate(data)
        obj.tool_meta = tool_meta
        obj.meta = event.meta
        return obj

    def to_session_event(self) -> SessionEvent:
        data = self.model_dump(exclude={"meta"}, exclude_none=True)
        data.pop("tool_meta", None)
        if self.tool_meta is not None:
            data["meta"] = self.tool_meta
        return SessionEvent(meta=self.meta, data=data)


class TodoWrite(SessionEventModel):
    """todo/write — 整表快照, log-only."""

    todos: list[TodoItem] = Field(default_factory=list)

    @classmethod
    def event_type(cls) -> str:
        return "todo/write"


class RequestHeader(SessionEventModel):
    """request/header — 下一次请求的完整 header, log-only."""

    header: EpochHeader = Field(default_factory=EpochHeader)
    reason: str | Literal["initial", "resume", "change"] = Field(default="initial")

    @classmethod
    def event_type(cls) -> str:
        return "request/header"


class RequestContextEvent(SessionEventModel):
    """request/context — 路由元信息, 路由或容量变化时记录, log-only."""

    provider: str = Field(default="")
    model: str = Field(default="")
    contextWindow: int | None = None

    @classmethod
    def event_type(cls) -> str:
        return "request/context"


class SessionEndSeed(SessionEventModel):
    """session/end-seed — 标记 constructor seed 结束, 载荷为空."""

    @classmethod
    def event_type(cls) -> str:
        return "session/end-seed"
