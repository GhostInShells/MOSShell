"""dsh runtime channel — 父 (connection) → 子 (session) 两层, 代码驱动有状态运行时 | 元能力 | alpha

把 :class:`DshRuntime` 挂成一颗可寻址的 channel 树:

    dsh (父, 注入 connection surface)
    ├── run(code)            # 阻塞: async def run(connection) — 管理面 (列/建 session)
    ├── run_bg(code, next)   # 后台: 完成发 notify signal
    ├── open(session_id, alias) -> 地址   # 物化 session 子 channel
    ├── close_session(alias)
    ├── sessions             # 列已开 session 子 channel
    └── dsh.<alias> (session 子 channel, 注入 session surface)
        ├── run(code)        # 阻塞: async def run(session) — 驱动面 (run/cancel/history)
        └── run_bg(code, next)

生命周期: channel 的 startup/close 绑到 runtime 的 ``__aenter__``/``close`` — runtime 是
有状态父对象 (持 connection + sessions + 后台 task), close 时 cancel 后台 task. 两个 surface
(connection/session) 的 API 经 ``moss codex get-interface ghoshell_moss.deepseek_harness.surfaces``
反射 (父 channel instruction 指路, 不在子 channel 重复).

Example:
    from ghoshell_moss.deepseek_harness.channel import new_dsh_runtime_channel
    from ghoshell_moss.deepseek_harness.runtime import DshRuntime
    from ghoshell_moss.deepseek_harness.launcher import DshConnection, DshConnectionConfig

    connection = DshConnection(DshConnectionConfig(host="127.0.0.1", port=3080, token="..."))
    runtime = DshRuntime(connection)
    main.import_channels(new_dsh_runtime_channel(runtime))

    # CTML:
    #   <dsh:run><![CDATA[
    #   async def run(connection):
    #       return [s.sessionId for s in await connection.sessions()]
    #   ]]></dsh:run>
    #   <dsh:open session_id="s1"/>
    #   <dsh.s1:run><![CDATA[
    #   async def run(session):
    #       r = await session.run("reply ok")
    #       return r.final_response
    #   ]]></dsh.s1:run>
"""

from __future__ import annotations

import re

from ghoshell_moss.core.blueprint.channel_builder import (
    MutableChannel,
    new_channel,
)
from ghoshell_moss.core.blueprint.states_channel import PrimeChannel
from ghoshell_moss.deepseek_harness.runtime import DshRuntime
from ghoshell_moss.deepseek_harness.surfaces import DshSessionSurface

__all__ = ["new_dsh_runtime_channel"]

_SURFACES_MODULE = "ghoshell_moss.deepseek_harness.surfaces"
_CHANNEL_NAME_RE = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")


def _register_run(
    chan: PrimeChannel,
    runtime: DshRuntime,
    get_surface,
    *,
    injected_name: str,
) -> None:
    """挂 run (阻塞) / run_bg (后台) 两个命令, 注入现场 surface."""

    @chan.build.command(
        name="run",
        blocking=True,
        always_observe=True,
        doc=(
            f"Compile `text__` and run `async def run({injected_name})` with the live "
            f"{injected_name} injected (asyncio pre-imported). Blocks until it returns; "
            f"captures stdout + return value."
        ),
    )
    async def _run(text__: str) -> str:
        return await runtime.run_code(get_surface(), text__)

    @chan.build.command(
        name="run_bg",
        blocking=False,
        always_observe=False,
        doc=(
            f"Run `async def run({injected_name})` in the background; returns immediately. "
            f"The result is delivered later as a notify signal (`next` true = queue-jump)."
        ),
    )
    async def _run_bg(text__: str, next: bool = False) -> str:
        return runtime.run_code_bg(get_surface(), text__, next=next)


def _new_session_child(runtime: DshRuntime, surface: DshSessionSurface, *, name: str) -> PrimeChannel:
    """一个 dsh session 的 channel: 注入 session surface; 生命周期随 add/remove 托管."""
    chan: PrimeChannel = new_channel(
        name=name,
        description="A live dsh session — code-driven turn loop (run/run_bg).",
    )
    _register_run(chan, runtime, lambda: surface, injected_name="session")

    @chan.build.instruction
    def _instruction() -> str:
        return (
            "A dsh session, held for you. run() injects `run(session)`: `session.run(prompt)` "
            "is the blocking single-turn loop, `session.cancel()` interrupts, `session.history()` "
            "reads back. State persists across run(). Read the session API:\n"
            f"  moss codex get-interface {_SURFACES_MODULE}"
        )

    return chan


def new_dsh_runtime_channel(
    runtime: DshRuntime,
    *,
    name: str = "dsh",
    description: str | None = None,
) -> MutableChannel:
    """父 channel: 注入 connection surface, open/close_session 管 session 子 channel.

    :param runtime: 有状态父对象 (持 connection + sessions + 后台 task).
    :param name: CTML 标签名, 默认 ``dsh``.
    :param description: 覆盖默认描述.
    """
    chan: PrimeChannel = new_channel(
        name=name,
        description=description or (
            "dsh connection driver — code-driven management (run/run_bg) + session children."
        ),
    )
    alias_to_session: dict[str, str] = {}
    session_counter = 0

    @chan.build.startup
    async def _startup() -> None:
        await runtime.__aenter__()

    @chan.build.close
    async def _close() -> None:
        await runtime.close()

    _register_run(chan, runtime, runtime.connection_surface, injected_name="connection")

    @chan.build.command(name="open", blocking=True, always_observe=True)
    async def _open(session_id: str, alias: str = "") -> str:
        """Open an existing session by id and mount it as a session child; return its address.

        dsh session ids are UUID-like (contain `-`, may start with a digit) — not valid
        channel names. Default to a generated ``sN``; a caller-supplied alias must match
        the channel name pattern.
        """
        nonlocal session_counter
        if alias:
            if not _CHANNEL_NAME_RE.fullmatch(alias):
                return f"invalid alias `{alias}` — channel name must match [a-zA-Z_][a-zA-Z0-9_]*"
        else:
            alias = f"s{session_counter}"
            session_counter += 1
        surface = await runtime.open_session(session_id)
        chan.add_virtual_channel(_new_session_child(runtime, surface, name=alias), alias)
        alias_to_session[alias] = session_id
        return f"opened session — address as `{name}.{alias}`"

    @chan.build.command(name="close_session", blocking=False, always_observe=False)
    async def _close_session(alias: str) -> str:
        """Remove a session child channel (its DshSession is closed with it)."""
        session_id = alias_to_session.pop(alias, alias)
        chan.remove_virtual_channel(alias)
        await runtime.close_session(session_id)
        return f"closed session `{alias}`"

    @chan.build.command(name="sessions", blocking=False, always_observe=True)
    async def _sessions() -> str:
        """List mounted session child channels."""
        children = chan.virtual_children()
        if not children:
            return "[dsh] no open sessions"
        return "\n".join(f"  {alias}" for alias in children)

    @chan.build.named_notices
    def _named_notices() -> dict[str, str | None]:
        """每个 open session 一个 named notice 片段 (key=alias): session_id + running.

        模型下一轮据 alias → session_id 反查; close 时片段缺席 → 框架发墓碑.
        """
        fragments: dict[str, str | None] = {}
        for alias, session_id in alias_to_session.items():
            surface = runtime.session_surface(session_id)
            running = surface.running if surface is not None else False
            fragments[alias] = f"session_id={session_id} running={running}"
        return fragments

    @chan.build.instruction
    def _instruction() -> str:
        return (
            "A dsh connection, held for you. run() compiles your Python and injects the "
            "connection surface as `run(connection)` — list workspaces/sessions or create a "
            "session here. open(session_id) mounts a session child; its run() injects "
            "`run(session)` for driving turns. run_bg() is the background variant (result "
            "delivered as a notify signal). Read both surfaces:\n"
            f"  moss codex get-interface {_SURFACES_MODULE}"
        )

    return chan
