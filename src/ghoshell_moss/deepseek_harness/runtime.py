"""DshRuntime — 持一个 dsh 连接 + 若干 agent 会话 + 感知/收据/信号.

flat 控制面: 会话按 ``name`` 寻址, 不物化成 channel 子节点. 每个 open 的会话挂一个
**ledger 订阅** —— 消费它的 session 事件 (user/message · assistant/message · turn/end),
维护一个轻量账本 (last human / last preview / last reason / unread). 一切状态面
(status / read / wait / named_notices) 都读这个账本, 不各自拉事件。

watch(level) 把账本变更外发成 typed notify signal:
- ``row``    默认 —— 只更新 named_notice 行 (diff 变了才发), 零 signal.
- ``notify`` 留痕不打断 (Priority.NOTICE, next=False).
- ``next``   显式升档, 保证下一轮 (next=True). 绝不默认 —— 100 个并行 agent 时默认
  notify 就是注意力轰炸.
拓扑依赖 (fan-in/fan-out/时序) 一律写 code, 本模块不提供编排动词。

lifted 动词 send/wait/interrupt/status/read 直接包 DshSessionSurface, 是「最常用的几个」;
其余全部走 run()/run_bg() 代码逃生口 (注入 surface)。send_signal 注入传输 —— node 层
传入 ``matrix.send_signal_to_ghost``, 本模块不碰 Matrix。

信号内容格式: ``[dsh {name}] {kind}: {preview}`` + description/hint, 与 ``input``
(用户消息) 天然两个 name, 模型一眼认出「这是 watch 会话的回执, 不是用户说话」。
"""

from __future__ import annotations

import asyncio
import inspect
import traceback as _traceback
from dataclasses import dataclass, field
from io import StringIO
from typing import Any, Callable, Literal

from ghoshell_moss.contracts.logger import LoggerItf, get_moss_logger
from ghoshell_moss.core.blueprint.mindflow import Priority, Signal
from ghoshell_moss.core.codex.compiler import Compiler
from ghoshell_moss.core.mindflow.notify_nucleus import new_notify_signal
from ghoshell_moss.deepseek_harness.launcher import DshConnection
from ghoshell_moss.deepseek_harness.session import DshSession
from ghoshell_moss.deepseek_harness.surfaces import DshConnectionSurface, DshSessionSurface
from ghoshell_moss.deepseek_harness.types.session_events import (
    AssistantMessageEvent,
    TurnEnd,
    UserMessageEvent,
)

__all__ = ["DshRuntime", "WatchLevel"]

# 预览/结果/单条消息的截断上限 —— 防止把整段 transcript 塞进 signal 或 command 返回值.
_PREVIEW_CAP = 200
_RESULT_CAP = 20_000
_READ_MSG_CAP = 300

# 代码上文: 让编译源码里天然有 asyncio, 模型在函数体里可用 asyncio.to_thread.
_PROLOGUE = "import asyncio\n"
# 模型代码的入口函数名: async def run(<surface>).
_ENTRY = "run"

WatchLevel = Literal["row", "notify", "next"]


def _compile(code: str, print_fn: Callable[..., None]) -> Any:
    """把上文 + 模型补的正文编译成临时 module; 注入 print 写进本次 buffer (并发隔离)."""
    return Compiler(
        source=_PROLOGUE + code,
        filename="<moss_dsh_run>",
        local_injections={"print": print_fn},
    ).compiled


def _render_result(out: str, ret: Any) -> str:
    """把 stdout + 返回值渲染成 command 返回值 (截断防 context 炸弹)."""
    parts: list[str] = []
    if out.strip():
        parts.append("--- stdout ---\n" + out.rstrip())
    if ret is not None:
        parts.append("--- result ---\n" + repr(ret))
    if not parts:
        return "(run returned None, no stdout)"
    return _truncate("\n".join(parts), _RESULT_CAP)


def _render_error(title: str, exc: Exception) -> str:
    """渲染运行/编译错误. 过滤掉本模块与 compiler 的内部 frame, 只留模型代码的栈."""
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
    body = "".join(_traceback.format_list(frames))
    last = _traceback.format_exception_only(type(exc), exc)[-1]
    return f"{title}: {type(exc).__name__}: {exc}\n{body}{last}"


def _truncate(text: str, cap: int) -> str:
    if len(text) <= cap:
        return text
    return text[:cap] + f"\n... [truncated, {len(text) - cap} more chars]"


def _message_text(message: Any) -> str:
    """从 Message / UserMessageEvent 抽取可见文本 (text 块拼接)."""
    content = getattr(message, "content", None)
    if content is None:
        return ""
    return "".join(
        block.text or "" for block in content if getattr(block, "type", "") == "text"
    )


# 人类消息判别: dsh 用 MessageSource.kind 区分来源, "user" = 人类在 web UI 输入,
# 其它 ("plugin"/"model") = 模型/插件注入. 模型自己的 send() 不应当作人类交互.
# (需对活 dsh 验证: session/prompt 落下的 user/message 是否确实带 kind="plugin")
def _is_human_user_message(event: UserMessageEvent) -> bool:
    return getattr(event.source, "kind", "user") == "user"


@dataclass
class _Ledger:
    """一个会话的轻量账本 (ledger 订阅维护, 状态面只读它)."""

    human: str = ""
    preview: str = ""
    reason: str = ""
    unread: int = 0


@dataclass
class _Receipt:
    """run_bg 的后台回执, 供 runs() 拉账本."""

    id: str
    name: str
    status: str = "running"
    preview: str = ""


Surface = DshConnectionSurface | DshSessionSurface
SignalSender = Callable[[Signal], None]


class DshRuntime:
    """持一个 dsh connection 及其会话 (按 name 寻址), 拥有 ledger 订阅 / watch / 后台回执.

    生命周期: ``async with runtime`` 或手动 ``__aenter__``/``__aexit__``.
    close() 幂等, 先 cancel 后台 task 再逐层关 sessions/connection.
    """

    def __init__(
        self,
        connection: DshConnection,
        *,
        send_signal: SignalSender | None = None,
        logger: LoggerItf | None = None,
    ) -> None:
        self._connection = connection
        self._connection_surface = DshConnectionSurface(self._connection)
        self._sessions: dict[str, DshSession] = {}
        self._surfaces: dict[str, DshSessionSurface] = {}
        self._names: dict[str, str] = {}  # session_id -> name
        self._ledgers: dict[str, _Ledger] = {}
        self._watch: dict[str, WatchLevel] = {}
        self._disposers: dict[str, list[Callable[[], None]]] = {}
        self._tasks: set[asyncio.Task] = set()
        self._receipts: dict[str, _Receipt] = {}
        self._receipt_counter = 0
        self._open_lock = asyncio.Lock()
        self._send_signal = send_signal
        self._logger = logger or get_moss_logger()
        self._closed = False

    # -- 生命周期 -- #

    async def __aenter__(self) -> "DshRuntime":
        await self._connection.__aenter__()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.close()

    async def close(self) -> None:
        """幂等关闭: cancel 后台 task → 关 sessions → 关 connection."""
        if self._closed:
            return
        self._closed = True
        tasks = list(self._tasks)
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._tasks.clear()
        for session in list(self._sessions.values()):
            await session.close()
        self._sessions.clear()
        self._surfaces.clear()
        self._names.clear()
        self._ledgers.clear()
        self._watch.clear()
        self._disposers.clear()
        await self._connection.__aexit__(None, None, None)

    # -- 会话接线 (name 注册) -- #

    def connection_surface(self) -> DshConnectionSurface:
        """connection 级管理面 (注入 connection 级 run)."""
        return self._connection_surface

    async def open_session(self, session_id: str, name: str) -> str:
        """按 sessionId 接线一个会话, 挂 ledger 订阅, 以 name 寻址. 幂等.

        加锁 + 双重检查防并发两次 open. 会话已在 (同 sessionId 不同 name) 则复用同一
        DshSession, 只多记一个 name —— 但一个会话只应有一个 name, 传错即被拒.
        """
        existing_name = self._names.get(session_id)
        if existing_name is not None:
            if existing_name != name:
                raise ValueError(
                    f"session {session_id} already open as `{existing_name}` (asked `{name}`)"
                )
            return name
        async with self._open_lock:
            existing_name = self._names.get(session_id)
            if existing_name is not None:
                return existing_name
            session = self._connection.create_session(session_id)
            await session.__aenter__()
            self._sessions[name] = session
            self._surfaces[name] = DshSessionSurface(session)
            self._names[session_id] = name
            self._ledgers[name] = _Ledger()
            self._disposers[name] = self._subscribe(name, session)
            return name

    async def create_session(self, name: str, *, cwd: str | None = None, agent_preset: str | None = None) -> str:
        """新建一个 dsh session 并接线, 返回其 name."""
        value = await self._connection_surface.create_session(cwd=cwd, agent_preset=agent_preset)
        return await self.open_session(value.sessionId, name)

    async def close_session(self, name: str) -> None:
        """关闭一个会话: 解绑 ledger 订阅 → 关 session (停消费 task + 断流)."""
        session = self._sessions.pop(name, None)
        self._surfaces.pop(name, None)
        self._ledgers.pop(name, None)
        self._watch.pop(name, None)
        for dispose in self._disposers.pop(name, []):
            dispose()
        if session is not None:
            self._names.pop(session.session_id, None)
            await session.close()

    def session_names(self) -> list[str]:
        """当前已接线的会话 name 列表."""
        return list(self._sessions)

    def session_surface(self, name: str) -> DshSessionSurface | None:
        """已接线会话的驱动面; 未接线返回 None."""
        return self._surfaces.get(name)

    def _surface_for(self, name: str) -> DshSessionSurface:
        surface = self._surfaces.get(name)
        if surface is None:
            raise ValueError(f"no dsh session `{name}` — sessions() lists open sessions")
        return surface

    # -- ledger 订阅 -- #

    def _subscribe(self, name: str, session: DshSession) -> list[Callable[[], None]]:
        """挂 ledger 订阅: user/message (人类交互) + assistant/message (尾句) + turn/end (结果)."""

        async def on_user(event: UserMessageEvent) -> None:
            if not _is_human_user_message(event):
                return
            text = _truncate(event.text(), _PREVIEW_CAP)
            ledger = self._ledgers.get(name)
            if ledger is None:
                return
            ledger.human = text
            ledger.unread += 1
            await self._maybe_signal(name, "human", text)

        async def on_assistant(event: AssistantMessageEvent) -> None:
            text = _truncate(_message_text(event.message), _PREVIEW_CAP)
            ledger = self._ledgers.get(name)
            if ledger is None:
                return
            ledger.preview = text
            ledger.unread += 1

        async def on_turn_end(event: TurnEnd) -> None:
            ledger = self._ledgers.get(name)
            if ledger is None:
                return
            ledger.reason = event.reason.kind if event.reason is not None else ""
            await self._maybe_signal(name, "reply", ledger.preview)

        return [
            session.on_session_event_model(UserMessageEvent, on_user),
            session.on_session_event_model(AssistantMessageEvent, on_assistant),
            session.on_session_event_model(TurnEnd, on_turn_end),
        ]

    async def _maybe_signal(self, name: str, kind: str, preview: str) -> None:
        """按 watch level 外发 signal. 未 watch 或 ``row`` = 零 signal (silent)."""
        if self._send_signal is None:
            return
        level = self._watch.get(name)
        if level is None or level == "row":
            return
        content = f"[dsh {name}] {kind}: {preview or '(no text)'}"
        if level == "next":
            signal = new_notify_signal(content, next=True, description=f"dsh session {name}", hint=kind)
        else:  # notify
            signal = new_notify_signal(
                content, priority=Priority.NOTICE, next=False,
                description=f"dsh session {name}", hint=kind,
            )
        self._send_signal(signal)

    # -- lifted 动词 -- #

    async def send(self, name: str, text: str) -> str:
        """发消息 (fire-and-forget). 不回结果, 结果走 wait/read 或 watch 通知."""
        surface = self._surface_for(name)
        await surface.prompt(content=text)
        return f"[dsh {name}] sent"

    async def wait(self, name: str, *, timeout: float | None = None) -> str:
        """等下一轮 turn 结束, 返回 settled 尾句 + reason. 阻塞原语 (少见, 别默认)."""
        session = self._sessions.get(name)
        if session is None:
            raise ValueError(f"no dsh session `{name}`")
        done = asyncio.Event()
        holder: dict[str, str] = {}

        async def on_turn_end(event: TurnEnd) -> None:
            holder["reason"] = event.reason.kind if event.reason is not None else ""
            done.set()

        dispose = session.on_session_event_model(TurnEnd, on_turn_end)
        try:
            if timeout is None:
                await done.wait()
            else:
                await asyncio.wait_for(done.wait(), timeout)
        finally:
            dispose()
        ledger = self._ledgers.get(name, _Ledger())
        ledger.unread = 0
        reason = holder.get("reason", "")
        preview = ledger.preview or "(no reply)"
        return f"[dsh {name}] {preview}\n  reason={reason or '…'}"

    async def interrupt(self, name: str) -> str:
        """中断当前 turn (保留队尾)."""
        surface = self._surface_for(name)
        await surface.cancel()
        return f"[dsh {name}] interrupted"

    async def status(self, name: str) -> str:
        """会话状态: running / watch / unread / tokens / 最近尾句."""
        surface = self._surface_for(name)
        ledger = self._ledgers.get(name, _Ledger())
        usage = surface.token_usage
        watch = self._watch.get(name, "—")
        return (
            f"[dsh {name}] running={surface.running} watch={watch} unread={ledger.unread} "
            f"tokens_in={usage.inputTokens} tokens_out={usage.outputTokens}\n"
            f"  last: {ledger.preview or '-'} ({ledger.reason or '…'})"
        )

    async def read(self, name: str, *, n: int = 10) -> str:
        """读会话上下文 (模型可见 surface 投影, 尊重 compact), 清 unread."""
        surface = self._surface_for(name)
        ledger = self._ledgers.get(name)
        if ledger is not None:
            ledger.unread = 0
        messages = await surface.surface_messages()
        if not messages:
            return f"[dsh {name}] (empty surface)"
        tail = messages[-n:]
        lines = [f"[dsh {name}] last {len(tail)} of {len(messages)}:"]
        for msg in tail:
            role = getattr(msg, "role", "?")
            text = _truncate(_message_text(msg), _READ_MSG_CAP)
            lines.append(f"  {role}: {text or '(empty)'}")
        return "\n".join(lines)

    # -- watch (感知面) -- #

    def watch(self, name: str, level: WatchLevel = "row") -> str:
        """选择性关注一个会话. row=只更新行; notify=留痕; next=保证下一轮."""
        if name not in self._sessions:
            raise ValueError(f"no dsh session `{name}`")
        if level not in ("row", "notify", "next"):
            raise ValueError(f"invalid watch level `{level}` — row/notify/next")
        self._watch[name] = level
        return f"[dsh {name}] watch={level}"

    def unwatch(self, name: str) -> str:
        """取消关注, 回到 silent background."""
        if name not in self._sessions:
            raise ValueError(f"no dsh session `{name}`")
        self._watch.pop(name, None)
        return f"[dsh {name}] unwatched"

    def watched(self) -> dict[str, WatchLevel]:
        return dict(self._watch)

    # -- 状态面 (channel 的 notice / named_notices / sessions 只读这些) -- #

    def describe(self, name: str) -> str:
        """一个会话的紧凑一行: state · 尾句. 供 sessions() / named_notices 用."""
        surface = self._surfaces.get(name)
        ledger = self._ledgers.get(name)
        if surface is None or ledger is None:
            return f"{name} (gone)"
        state = "running" if surface.running else ("unread" if ledger.unread else "idle")
        tail = ledger.preview or ledger.human or "-"
        return f"{name} · {state} · {tail}"

    def notice_summary(self) -> str:
        """单行折叠: 连接 + 三个状态桶计数 (逐字稳定, 计数不翻不重发)."""
        running = unread = idle = 0
        for name in self._sessions:
            surface = self._surfaces.get(name)
            ledger = self._ledgers.get(name)
            if surface is not None and surface.running:
                running += 1
            elif ledger is not None and ledger.unread > 0:
                unread += 1
            else:
                idle += 1
        host = self._connection.config.host
        port = self._connection.config.port
        up = "up" if self._connection.is_running() else "down"
        return (
            f"dsh {host}:{port} · {up} · {len(self._sessions)} sessions "
            f"({running} running, {unread} unread, {idle} idle)"
        )

    def notice_rows(self) -> dict[str, str | None]:
        """watched 会话的 named_notice 行 (name -> 一行). 非 watched 的静默缺席."""
        rows: dict[str, str | None] = {}
        for name, level in self._watch.items():
            surface = self._surfaces.get(name)
            ledger = self._ledgers.get(name)
            if surface is None or ledger is None:
                continue
            state = "running" if surface.running else ("unread" if ledger.unread else "idle")
            tail = ledger.preview or ledger.human or "-"
            rows[name] = f"{state} · {tail} · watch={level}"
        return rows

    # -- 代码逃生口 -- #

    async def run_code(self, surface: Surface, code: str) -> str:
        """阻塞运行: 编译 + 提 ``async def run(surface)`` + 注入 surface + await.

        捕获 stdout + 返回值渲染成字符串; 编译/运行错误渲染干净栈 (只留模型代码帧).
        注入写进本次 buffer 的 ``print`` (而非 redirect_stdout) — 并发 run_bg 各有
        自己的 buffer, 不串线也不劫持 sys.stdout.
        """
        buffer = StringIO()

        def _print(*args, sep: str = " ", end: str = "\n", file=None, flush: bool = False) -> None:
            text = sep.join(map(str, args)) + end
            if file is not None:
                file.write(text)
                if flush:
                    file.flush()
            else:
                buffer.write(text)

        try:
            module = _compile(code, _print)
        except Exception as exc:
            return _render_error("COMPILE ERROR", exc)

        run_fn = module.__dict__.get(_ENTRY)
        if run_fn is None:
            return (
                f"COMPILE ERROR: no `{_ENTRY}` defined in the source. "
                f"You must define async def {_ENTRY}({_injected_name(surface)})."
            )
        if not inspect.iscoroutinefunction(run_fn):
            return (
                f"COMPILE ERROR: `{_ENTRY}` must be `async def`. "
                "A plain def would block the event loop — mark it `async def` and "
                "wrap blocking calls with `asyncio.to_thread(...)`."
            )

        try:
            ret = await run_fn(surface)
        except Exception as exc:
            return _render_error("RUN ERROR", exc)
        return _render_result(buffer.getvalue(), ret)

    def run_code_bg(self, name: str, code: str) -> str:
        """后台运行 session 级代码, 返回回执 id; 完成时按 watch level 发 signal.

        silent 默认: 未 watch 时完成只进 runs() 账本, 不 push.
        """
        surface = self._surface_for(name)
        self._receipt_counter += 1
        rid = str(self._receipt_counter)
        self._receipts[rid] = _Receipt(id=rid, name=name)
        task = asyncio.create_task(self._background(name, surface, code, rid))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return f"[dsh {name}] queued #{rid}"

    async def _background(self, name: str, surface: DshSessionSurface, code: str, rid: str) -> None:
        result = await self.run_code(surface, code)
        receipt = self._receipts.get(rid)
        if receipt is not None:
            receipt.status = "done"
            receipt.preview = _truncate(result, _PREVIEW_CAP)
        await self._maybe_signal(name, f"#{rid} done", _truncate(result, _PREVIEW_CAP))

    def runs(self) -> str:
        """后台任务账本 (run_bg 回执), 拉真相用."""
        if not self._receipts:
            return "[dsh] no background runs"
        lines = ["[dsh] background runs:"]
        for rid, receipt in self._receipts.items():
            lines.append(f"  #{receipt.id} {receipt.name} [{receipt.status}] {receipt.preview}")
        return "\n".join(lines)


def _injected_name(surface: Surface) -> str:
    """surface → 注入参数名 (connection / session), 用于编译错误提示."""
    return "connection" if isinstance(surface, DshConnectionSurface) else "session"
