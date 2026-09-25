"""dsh 驱动表面 — 注入 sandbox 的两个窄投影 (connection 级管理 + session 级驱动).

方案二 (code reflection): 方法返回 codified 的 pydantic value 模型 (不 dict-ify).
这些 BaseModel 随本模块一起被 Reflector / ``moss codex get-interface`` 反射进
prompt (``<attr>`` 块), 模型无需 import 即可读到字段结构 — 这正是「协议代码化」
的目的: schema 由反射自动给出, 不手搓 dict 协议 (那是方案一).

窄投影的边界是「暴露哪些方法」: 隐藏 raw client / http / token / config 等
plumbing, 而非 dict 化返回值. 两个表面分别绑定一个现场对象:
- ``DshConnectionSurface`` → 注入 ``async def run(connection)`` (管理面)
- ``DshSessionSurface``     → 注入 ``async def run(session)``     (驱动面)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ghoshell_moss.deepseek_harness.session import DshRunResult
from ghoshell_moss.deepseek_harness.types.nouns import WorkspaceView
from ghoshell_moss.deepseek_harness.types.session_events import (
    Message,
    SessionEvent,
    TokenUsage,
)
from ghoshell_moss.deepseek_harness.types.sessions import (
    ModelCatalog,
    ModelSelection,
    PromptContentPart,
    SessionCancelValue,
    SessionCreateParams,
    SessionCreateValue,
    SessionForkValue,
    SessionListParams,
    SessionListValue,
    SessionPromptValue,
    SessionRenameValue,
    SessionSelectModelValue,
    SessionSummary,
)

if TYPE_CHECKING:
    from ghoshell_moss.deepseek_harness.launcher import DshConnection
    from ghoshell_moss.deepseek_harness.session import DshSession

__all__ = ["DshConnectionSurface", "DshSessionSurface"]


def _content(content: str | list[PromptContentPart]) -> list[PromptContentPart]:
    """str → ``[{type:text, text}]``; list 原样透传 (镜像 DshSession._normalize_content)."""
    if isinstance(content, str):
        return [PromptContentPart(type="text", text=content)]
    return list(content)


class DshConnectionSurface:
    """connection 级管理面 — 列 workspace/session、建 session、读 model catalog.

    只暴露管理动词, 不透传 raw client / http / token / config.
    """

    def __init__(self, connection: DshConnection) -> None:
        self._connection = connection

    async def workspaces(self) -> list[WorkspaceView]:
        """List workspaces known to this dsh connection."""
        return await self._connection.workspaces()

    async def sessions(self) -> list[SessionSummary]:
        """List sessions (session/list)."""
        value = await self._connection.client.call(
            "session/list",
            SessionListParams(),
            SessionListValue,
            args_key="_request",
        )
        return value.items

    async def create_session(
        self,
        *,
        cwd: str | None = None,
        agent_preset: str | None = None,
    ) -> SessionCreateValue:
        """Create a new session; returns SessionCreateValue(sessionId, agentPreset)."""
        return await self._connection.client.session_create(
            SessionCreateParams(cwd=cwd, agentPreset=agent_preset)
        )

    async def model_catalog(self) -> ModelCatalog:
        """Session model catalog: default + routableProviders + groups + failures."""
        return await self._connection.client.call(
            "session/modelCatalog", None, ModelCatalog, args_key=None
        )


class DshSessionSurface:
    """session 级驱动面 — 单轮 run/cancel、读历史、fork/rename/选模型.

    绑定单个 session; 方法收 plain args (sessionId 由底层 facade 自动填充),
    返回 codified value 模型. run() 是 loop 一个 session 的基本原语.
    """

    def __init__(self, session: DshSession) -> None:
        self._session = session

    @property
    def session_id(self) -> str:
        """本 surface 绑定的 dsh session id."""
        return self._session.session_id

    async def run(
        self,
        content: str | list[PromptContentPart],
        *,
        mode: str = "queue",
        timeout: float | None = None,
    ) -> DshRunResult:
        """Single-turn loop primitive: prompt, then wait for the turn to end.

        Returns DshRunResult(session_id, final_response, finish_reason, events).
        ``cancel()`` interrupts an in-flight run. :param timeout: seconds; None = wait forever.
        """
        return await self._session.run(content, mode=mode, timeout=timeout)

    async def prompt(
        self,
        content: str | list[PromptContentPart],
        *,
        mode: str = "queue",
        client_timezone: str | None = None,
    ) -> SessionPromptValue:
        """Fire-and-return prompt (does not wait for the turn to end); returns accepted."""
        return await self._session.prompt(
            content=_content(content), mode=mode, client_timezone=client_timezone
        )

    async def cancel(self) -> SessionCancelValue:
        """Interrupt an in-flight turn, keeping queued work; makes a pending run() settle."""
        return await self._session.cancel()

    async def history(
        self,
        *,
        before_seq: int | None = None,
        max_messages: int | None = None,
    ) -> list[SessionEvent]:
        """Read one page of session events backwards from the follow cursor (session/page)."""
        return await self._session.history(before_seq=before_seq, max_messages=max_messages)

    async def surface_messages(self) -> list[Message]:
        """Model-visible surface projection (user/assistant/tool-result, compact replace applied)."""
        return await self._session.surface_messages()

    async def instruction(self) -> str | None:
        """Current assembled system prompt / instruction."""
        return await self._session.instruction()

    async def fork(self, *, at_seq: int | None = None) -> SessionForkValue:
        """Fork this session at a seq; returns the new sessionId."""
        return await self._session.fork(at_seq=at_seq)

    async def rename(self, *, title: str) -> SessionRenameValue:
        """Rename this session."""
        return await self._session.rename(title=title)

    async def select_model(
        self,
        *,
        provider: str,
        model: str,
        reasoning_effort: str | None = None,
    ) -> SessionSelectModelValue:
        """Select provider/model (and optional reasoning effort) for this session."""
        return await self._session.select_model(
            provider=provider, model=model, reasoning_effort=reasoning_effort
        )

    async def model_selection(self, *, force: bool = False) -> ModelSelection:
        """Current selected model (provider/model/reasoningEffort)."""
        return await self._session.model_selection(force=force)

    async def cwd(self, *, force: bool = False) -> str | None:
        """Session working directory (constant once created)."""
        return await self._session.cwd(force=force)

    async def agent_preset(self, *, force: bool = False) -> str | None:
        """Session agentPreset (running mode, set at creation)."""
        return await self._session.agent_preset(force=force)

    @property
    def running(self) -> bool:
        """Whether the dsh agent is currently running a turn (live status mirror)."""
        return self._session.running

    @property
    def token_usage(self) -> TokenUsage:
        """Cumulative token usage for this session."""
        return self._session.token_usage

    async def when_idle(self) -> None:
        """Wait until the agent is idle (already idle returns immediately)."""
        await self._session.when_idle()

    async def when_running(self) -> None:
        """Wait until the agent is running (already running returns immediately)."""
        await self._session.when_running()
