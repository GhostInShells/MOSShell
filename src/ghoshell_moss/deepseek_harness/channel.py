"""dsh agent 控制面 channel — flat 单 channel, 借 IM 会话列表的感知形态 | 元能力 | alpha

一个 ``dsh`` channel, 持 :class:`DshRuntime`(一个 dsh 连接上的若干 agent 会话).
会话按 ``name`` 寻址, 不物化成 channel 子节点 —— 控制面是平的, 认知成本是
「一个 channel + 一组吃 name 的动词」.

感知面 (最小污染注意力):
- ``notice`` 单行折叠: 连接 + 三个状态桶计数 (running/unread/idle), 计数不翻不重发.
- ``named_notices`` 只放 **watched** 的会话 —— 每行是那个会话的状态变更 (diff 才发);
  非 watched 的静默, 是 silent background.

控制面 (最常用的几个拎出来, 其余走代码):
- ``send(name, text)`` 发消息 · ``wait(name)`` 等结果 · ``interrupt(name)`` 中断 ·
  ``status(name)`` 状态 · ``read(name)`` 读上下文.
- ``watch(name, level)`` 选择性关注 (row/notify/next), ``unwatch`` 回静默.
- ``run(code, name)`` / ``run_bg(name, code)`` 代码逃生口 —— 注入 connection/session surface.

注意力铁律 (instruction 也写死):
- 默认 silent —— 后台结果只进 runs() 账本 + notice 计数, 零 signal.
- ``next`` 是显式升档, 绝不默认 —— 100 个并行 agent 时默认 notify 就是轰炸.
- 拓扑依赖 (fan-in/fan-out/时序) 一律写 code, 控制面不提供编排动词.

Example:
    # 开一个会话, 发一条消息, 停在房里等结果:
    <dsh:new name="p1"/>
    <dsh:send name="p1" text="看一下 runtime.py 的接口"/>
    <dsh:wait name="p1"/>

    # 后台跑, 有事通知我 (notify), 急事 next:
    <dsh:watch name="p1" level="notify"/>
    <dsh:send name="p1" text="继续"/>

    # 代码逃生口 (注入 session surface):
    <dsh:run name="p1"><![CDATA[
    async def run(session):
        return await session.history(max_messages=20)
    ]]></dsh:run>
"""

from __future__ import annotations

import re

from ghoshell_moss.core.blueprint.channel_builder import (
    MutableChannel,
    new_channel,
)
from ghoshell_moss.core.blueprint.states_channel import PrimeChannel
from ghoshell_moss.deepseek_harness.runtime import DshRuntime

__all__ = ["new_dsh_runtime_channel"]

_NAME_RE = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")


def new_dsh_runtime_channel(
    runtime: DshRuntime,
    *,
    name: str = "dsh",
    description: str | None = None,
) -> MutableChannel:
    """构建 dsh agent 控制面 channel (flat). 生命周期绑 runtime 的 enter/close."""
    chan: PrimeChannel = new_channel(
        name=name,
        description=description or (
            "dsh agent control plane — drive dsh sessions as a code-first surface: "
            "send/wait/interrupt/status/read, watch for notifications, run(code) escape."
        ),
    )
    session_counter = 0

    @chan.build.startup
    async def _startup() -> None:
        await runtime.__aenter__()

    @chan.build.close
    async def _close() -> None:
        await runtime.close()

    def _mint(alias: str) -> str:
        nonlocal session_counter
        if alias:
            return alias
        n = session_counter
        session_counter += 1
        return f"s{n}"

    # -- 会话接线 -- #

    @chan.build.command(name="sessions", blocking=False, always_observe=True)
    async def _sessions() -> str:
        """列已接线的会话 (一行一个)."""
        names = runtime.session_names()
        if not names:
            return f"[{name}] no sessions — new()/open() to attach one"
        lines = [runtime.notice_summary()]
        lines.extend(runtime.describe(n) for n in names)
        return "\n".join(lines)

    @chan.build.command(name="open", blocking=False, always_observe=True)
    async def _open(session_id: str, name: str = "") -> str:
        """把 dsh 上已有的一个会话接进来, 以 name 寻址. 不传 name 自动 mint sN."""
        alias = _mint(name)
        if not _NAME_RE.fullmatch(alias):
            return f"invalid name `{alias}` — must match [a-zA-Z_][a-zA-Z0-9_]*"
        try:
            await runtime.open_session(session_id, alias)
        except ValueError as exc:
            return str(exc)
        return f"opened `{alias}` — address it as send(name={alias!r})/wait(name={alias!r})"

    @chan.build.command(name="new", blocking=False, always_observe=True)
    async def _new(cwd: str | None = None, agent_preset: str | None = None, name: str = "") -> str:
        """新建一个 dsh 会话并接进来, 返回其 name."""
        alias = _mint(name)
        if not _NAME_RE.fullmatch(alias):
            return f"invalid name `{alias}` — must match [a-zA-Z_][a-zA-Z0-9_]*"
        try:
            await runtime.create_session(alias, cwd=cwd, agent_preset=agent_preset)
        except Exception as exc:
            return f"create failed: {exc}"
        return f"created `{alias}` — address it as send(name={alias!r})/wait(name={alias!r})"

    # -- 对话动词 -- #

    @chan.build.command(name="send", blocking=False, always_observe=False)
    async def _send(name: str, text: str) -> str:
        """发一条消息 (fire-and-forget). 结果走 wait()/read() 或 watch 通知."""
        return await runtime.send(name, text)

    @chan.build.command(name="wait", blocking=True, always_observe=True)
    async def _wait(name: str, timeout: float | None = None) -> str:
        """停在房里等下一轮结束, 返回 settled 尾句 + reason. 阻塞; 别默认用它."""
        return await runtime.wait(name, timeout=timeout)

    @chan.build.command(name="interrupt", blocking=False, always_observe=False)
    async def _interrupt(name: str) -> str:
        """中断当前 turn (保留队尾)."""
        return await runtime.interrupt(name)

    @chan.build.command(name="status", blocking=False, always_observe=True)
    async def _status(name: str) -> str:
        """会话状态: running / watch / unread / tokens / 最近尾句."""
        return await runtime.status(name)

    @chan.build.command(name="read", blocking=False, always_observe=True)
    async def _read(name: str, n: int = 10) -> str:
        """读会话上下文 (模型可见 surface 投影), 清 unread."""
        return await runtime.read(name, n=n)

    # -- 感知面 -- #

    @chan.build.command(name="watch", blocking=False, always_observe=False)
    async def _watch(name: str, level: str = "row") -> str:
        """选择性关注一个会话: level ∈ row(只更新行) / notify(留痕) / next(保证下一轮)."""
        return runtime.watch(name, level)

    @chan.build.command(name="unwatch", blocking=False, always_observe=False)
    async def _unwatch(name: str) -> str:
        """取消关注, 回到 silent background."""
        return runtime.unwatch(name)

    @chan.build.command(name="runs", blocking=False, always_observe=True)
    async def _runs() -> str:
        """后台任务账本 (run_bg 回执)."""
        return runtime.runs()

    # -- 代码逃生口 -- #

    @chan.build.command(name="run", blocking=True, always_observe=True)
    async def _run(text__: str, name: str = "") -> str:
        """编译并跑 ``async def run(<surface>)``. name 空 = 注入 connection surface;
        给 name = 注入该会话的 session surface. 拓扑依赖/一次性拉取都写这里."""
        if name:
            surface = runtime.session_surface(name)
            if surface is None:
                return f"no dsh session `{name}` — sessions() lists open sessions"
            return await runtime.run_code(surface, text__)
        return await runtime.run_code(runtime.connection_surface(), text__)

    @chan.build.command(name="run_bg", blocking=False, always_observe=False)
    async def _run_bg(name: str, text__: str) -> str:
        """后台跑 session 级代码, 返回回执 id; 完成按 watch level 发 signal."""
        return runtime.run_code_bg(name, text__)

    # -- 感知面 -- #

    @chan.build.notice
    def _notice() -> str:
        return runtime.notice_summary()

    @chan.build.named_notices
    def _named_notices() -> dict[str, str | None]:
        return runtime.notice_rows()

    @chan.build.instruction
    def _instruction() -> str:
        return (
            "dsh agent control plane — 一个连接上的若干 agent 会话, 借 IM 会话列表的感知形态.\n"
            "会话按 name 寻址 (不是 channel 子节点). 控制面是平的.\n\n"
            "常用: new/open 接线 → send(name, text) 发消息 → wait(name) 停在房里等结果 → "
            "interrupt(name) 中断 → status(name) 状态 → read(name) 读上下文.\n\n"
            "注意力铁律 (100 个并行 agent 也成立):\n"
            "- 默认 silent: 后台结果只进 runs() 账本 + notice 计数, 零 signal.\n"
            "- watch(name, level) 才推送: row=只更新行 / notify=留痕 / next=保证下一轮. "
            "next 是显式升档, 绝不默认.\n"
            "- 有拓扑依赖 (fan-in/fan-out/时序) 一律写 code, 用 run()/run_bg() 编排, "
            "控制面不提供编排动词.\n\n"
            "代码逃生口: run(code, name) 注入 connection(不给 name)/session(给 name) surface; "
            "run_bg(name, code) 后台跑. 其余所有一次性能力 (fork/rename/选模型/attachments/…) "
            "都走代码, 不为它们开命令."
        )

    return chan
