"""dsh 连接/session 的元 channel — 父→connection→session 三层, 代码驱动有状态运行时 | 元能力 | alpha

把「驱动 dsh」复刻成一颗可寻址、生命周期被 shell runtime 托管的 channel 树:

    dsh (父)
    └── connect(host, port, ...) -> alias          # always_observe, 返别名
        └── dsh.<alias> (connection 子 channel, 持有 DshConnection)
            ├── exec(code)                          # 注入 connection, 管理面: 建/列 session
            ├── open(...) -> session_alias          # always_observe, 返 session 别名
            ├── close_session(alias)
            └── dsh.<alias>.<session_alias> (session 子 channel, 持有 DshSession)
                ├── exec(code)                      # 注入 session, loop 原语 run/cancel
                └── (随 close_session 拆除)

channel builder 的闭包 + ``@build.startup``/``@build.close`` 生命周期, 天然承载
「一个被持有的、可用代码驱动的有状态对象」: connection 与 session 同构, 各是一层。
虚拟子 channel 由 ChannelTree 在 refresh 时 materialize / 拆除 —— startup 进入连接,
close 退出连接, 生命周期不归模型手写。

与 runtime_debug 同套「编译 + 注入 + 调 main」骨架, 但注入的是现场 connection / session
而非 IoC 容器; 连接只 attach 不 spawn (子进程生命周期是 ghost 的职责)。

Example:
    from ghoshell_moss.channels.dsh_channel import new_dsh_channel
    main.import_channels(new_dsh_channel())

    # CTML:
    #   <dsh:connect host="127.0.0.1" port="3080" alias="main"/>
    #   <dsh.main:open alias="s1"/>
    #   <dsh.main.s1:exec><![CDATA[
    #   async def main(session):
    #       r = await session.run("reply ok")
    #       return r.final_response
    #   ]]></dsh.main.s1:exec>
"""

from __future__ import annotations

import asyncio
import contextlib
import inspect
import os
import traceback as _traceback
from contextlib import redirect_stdout
from io import StringIO
from typing import Any, Callable

from ghoshell_moss.core.blueprint.channel_builder import (
    ChannelFactory,
    MutableChannel,
    new_channel,
)
from ghoshell_moss.core.blueprint.states_channel import PrimeChannel
from ghoshell_moss.core.codex.compiler import Compiler
from ghoshell_moss.deepseek_harness.launcher import DshConnection, DshConnectionConfig
from ghoshell_moss.deepseek_harness.session import DshSession
from ghoshell_moss.deepseek_harness.types import sessions

__all__ = ["new_dsh_channel", "build_dsh_channel"]

_MAIN = "main"
# 代码上文: 让编译源码里天然有 asyncio, 模型在函数体里可用 asyncio.to_thread.
_PROLOGUE = "import asyncio\n"

# 模型读模块的入口 — 指令里不内联 API, 只指路.
_SESSION_MODULE = "ghoshell_moss.deepseek_harness.session:DshSession"
_CONNECTION_MODULE = "ghoshell_moss.deepseek_harness.launcher:DshConnection"
_SESSION_EVENTS_MODULE = "ghoshell_moss.deepseek_harness.types.session_events"


# ---- 编译 + 注入 + 渲染 (镜像 runtime_debug) ---- #


def _compile(text: str):
    """把上文 + 模型补的正文编译成临时 module. 失败抛异常, 由调用方格式化."""
    return Compiler(
        source=_PROLOGUE + text,
        filename="<moss_dsh_exec>",
    ).compiled


def _render_result(out: str, ret: Any) -> str:
    """把 stdout + 返回值渲染成 command 返回值."""
    parts: list[str] = []
    if out.strip():
        parts.append("--- stdout ---\n" + out.rstrip())
    if ret is not None:
        parts.append("--- result ---\n" + repr(ret))
    if not parts:
        return "(main returned None, no stdout)"
    return "\n".join(parts)


def _render_error(title: str, exc: Exception) -> str:
    """渲染运行/编译错误. 过滤掉本 channel 与 compiler 的内部 frame, 只留模型代码的栈."""
    tb = _traceback.extract_tb(exc.__traceback__)
    excluded = set()
    for mod in (Compiler,):
        try:
            excluded.add(inspect.getfile(mod))
        except TypeError:
            pass
    try:
        excluded.add(__file__)
    except NameError:
        pass
    frames = [f for f in tb if f.filename not in excluded]
    body = ''.join(_traceback.format_list(frames))
    last = _traceback.format_exception_only(type(exc), exc)[-1]
    head = f"{title}: {type(exc).__name__}: {exc}"
    return head + "\n" + body + last


def _register_exec(
        chan: PrimeChannel,
        *,
        inject: Callable[[], Any],
        injected_name: str,
) -> None:
    """挂一个 exec 命令: 编译模型代码, 调其 ``main(<injected>)``, 注入现场对象."""

    @chan.build.command(
        name="exec",
        blocking=True,
        always_observe=True,
        doc=(
            f"Compile `text__` and run `async def main({injected_name})` with the live "
            f"{injected_name} injected (asyncio pre-imported). Captures stdout + return value; "
            f"reports errors."
        ),
    )
    async def _exec(text__: str) -> str:
        try:
            module = _compile(text__)
        except Exception as e:
            return _render_error("COMPILE ERROR", e)

        main_fn = module.__dict__.get(_MAIN)
        if main_fn is None:
            return (
                f"COMPILE ERROR: no `{_MAIN}` defined in the source. "
                f"You must define async def {_MAIN}({injected_name})."
            )
        if not inspect.iscoroutinefunction(main_fn):
            return (
                f"COMPILE ERROR: `{_MAIN}` must be `async def`. "
                f"A plain def would block the event loop — mark it `async def` and "
                f"wrap blocking calls with `asyncio.to_thread(...)`."
            )

        buffer = StringIO()
        with redirect_stdout(buffer):
            try:
                ret = await main_fn(inject())
            except Exception as e:
                return _render_error("RUN ERROR", e)

        return _render_result(buffer.getvalue(), ret)


# ---- session 子 channel ---- #


def _new_session_child(session: DshSession, *, name: str) -> PrimeChannel:
    """一个 dsh session 的 channel: 持有 DshSession, exec 注入它; 生命周期随 add/remove 托管."""
    chan: PrimeChannel = new_channel(
        name=name,
        description="A live dsh session — code-driven turn loop (run/cancel/history/fork).",
    )
    stack = contextlib.AsyncExitStack()

    @chan.build.startup
    async def _startup() -> None:
        await stack.enter_async_context(session)

    @chan.build.close
    async def _close() -> None:
        await stack.aclose()

    _register_exec(chan, inject=lambda: session, injected_name="session")

    @chan.build.instruction
    def _session_instruction() -> str:
        return (
            "A dsh session, held for you. exec() runs your Python with the live session "
            "injected as `main(session)`: `session.run(prompt)` is the blocking single-turn "
            "loop, `session.cancel()` interrupts it. State persists across exec. Read the API:\n"
            f"  moss codex get-interface {_SESSION_MODULE}\n"
            f"  moss codex get-interface {_SESSION_EVENTS_MODULE}"
        )

    @chan.build.context_messages
    def _context() -> list[str]:
        usage = session.token_usage
        return [
            f"[dsh session] running={session.running} "
            f"tokens_in={usage.inputTokens} tokens_out={usage.outputTokens}"
        ]

    return chan


# ---- connection 子 channel ---- #


def _new_connection_child(
        *,
        name: str,
        connection: DshConnection,
        cwd: str,
        agent_preset: str,
) -> PrimeChannel:
    """一个 dsh connection 的 channel: 持有 DshConnection, exec 注入它; open/close_session 管 session.

    ``connection`` 由调用方注入 (父 channel 的 connect 构建真连接, 测试注入 fake), 不在本函数内
    构建 —— 避免把网络副作用藏进 channel 构造。
    """
    default_cwd = cwd
    chan: PrimeChannel = new_channel(
        name=name,
        description="A live dsh connection — code-driven management (open sessions, list, fork).",
    )
    stack = contextlib.AsyncExitStack()
    session_counter = 0

    @chan.build.startup
    async def _startup() -> None:
        await stack.enter_async_context(connection)

    @chan.build.close
    async def _close() -> None:
        await stack.aclose()

    _register_exec(chan, inject=lambda: connection, injected_name="connection")

    @chan.build.instruction
    def _connection_instruction() -> str:
        return (
            "A dsh connection, held for you. exec() runs your Python with the live connection "
            "injected as `main(connection)` — manage sessions (create/list/fork) here; open() "
            "mounts a session child for running turns. Read the API:\n"
            f"  moss codex get-interface {_CONNECTION_MODULE}\n"
            f"  moss codex get-interface {_SESSION_MODULE}"
        )

    @chan.build.command(name="open", blocking=True, always_observe=True)
    async def _open(alias: str = "", cwd: str | None = None) -> str:
        """Create a dsh session and mount it as a session child channel; return its address."""
        nonlocal session_counter
        created = await connection.client.session_create(
            sessions.SessionCreateParams(cwd=cwd or default_cwd, agentPreset=agent_preset)
        )
        session = connection.create_session(created.sessionId)
        session_alias = alias or f"s{session_counter}"
        session_counter += 1
        child = _new_session_child(session, name=session_alias)
        chan.add_virtual_channel(child, session_alias)
        return f"opened session — address as `{name}.{session_alias}`"

    @chan.build.command(name="close_session", blocking=False, always_observe=False)
    async def _close_session(alias: str) -> str:
        """Remove a session child channel (its DshSession is closed with it)."""
        chan.remove_virtual_channel(alias)
        return f"closed session `{alias}`"

    @chan.build.command(name="sessions", blocking=False, always_observe=True)
    async def _sessions() -> str:
        """List mounted session child channels."""
        children = chan.virtual_children()
        if not children:
            return "[dsh] no open sessions"
        return "\n".join(f"  {alias}" for alias in children)

    return chan


# ---- 父 channel ---- #


def new_dsh_channel(
        *,
        name: str = "dsh",
        description: str | None = None,
) -> MutableChannel:
    """父 channel: connect 出 connection 子 channel, disconnect 拆除.

    :param name: CTML 标签名, 默认 ``dsh``.
    :param description: 覆盖默认描述.
    """
    chan: PrimeChannel = new_channel(
        name=name,
        description=description or (
            "Meta dsh driver — connect to dsh connections, each a child channel; "
            "drive them with code (stateful, lifecycle-managed)."
        ),
    )
    conn_counter = 0

    @chan.build.command(name="connect", blocking=True, always_observe=True)
    async def _connect(
            host: str = "127.0.0.1",
            port: int = 3083,
            alias: str = "",
            cwd: str | None = None,
            agent_preset: str = "standard",
    ) -> str:
        """Attach a dsh connection as a child channel; return its address (dsh.<alias>)."""
        nonlocal conn_counter
        conn_alias = alias or f"c{conn_counter}"
        conn_counter += 1
        connection = DshConnection(DshConnectionConfig(host=host, port=port))
        child = _new_connection_child(
            name=conn_alias,
            connection=connection,
            cwd=cwd or os.getcwd(),
            agent_preset=agent_preset,
        )
        chan.add_virtual_channel(child, conn_alias)
        return f"connected dsh at {host}:{port} — address as `{name}.{conn_alias}`"

    @chan.build.command(name="disconnect", blocking=False, always_observe=False)
    async def _disconnect(alias: str) -> str:
        """Remove a connection child channel (its DshConnection is closed with it)."""
        chan.remove_virtual_channel(alias)
        return f"disconnected `{alias}`"

    @chan.build.command(name="connections", blocking=False, always_observe=True)
    async def _connections() -> str:
        """List attached connection child channels."""
        children = chan.virtual_children()
        if not children:
            return "[dsh] no connections"
        return "\n".join(f"  {alias}" for alias in children)

    @chan.build.instruction
    def _instruction() -> str:
        return (
            "Three levels: connect() attaches a connection child (dsh.<alias>); open() on it "
            "mounts a session child (dsh.<alias>.<s>). connect()/open() are @observe — read the "
            "returned address, then address the child next turn. exec() on a child runs your "
            "Python against its held object (`async def main(connection)` / `async def "
            "main(session)`, asyncio pre-imported). Read the object APIs:\n"
            f"  moss codex get-interface {_CONNECTION_MODULE}\n"
            f"  moss codex get-interface {_SESSION_MODULE}"
        )

    return chan


def build_dsh_channel(
        *,
        name: str = "dsh",
        description: str | None = None,
) -> ChannelFactory:
    """IoC 集成工厂: 返回 dsh 父 channel 的 ChannelFactory (连接层无 IoC 依赖)."""
    def factory(_container) -> MutableChannel:
        return new_dsh_channel(name=name, description=description)
    return factory
