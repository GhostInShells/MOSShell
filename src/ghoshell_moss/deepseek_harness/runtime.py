"""DshRuntime — 有状态父对象: 持一个 DshConnection + 其 sessions + 后台 task.

MatrixLifecycleObject 形状 (``__aenter__``/``__aexit__``). ``close()`` 按序拆除:
cancel 后台 task → 关 sessions → 关 connection. 后台 task 的 cancel 控制是它存在的
主因 — channel runtime 只给 startup/close, 不给后台 task cancel.

代码驱动: ``run_code``/``run_code_bg`` 用 ``Compiler`` 编译模型代码 (检查编译报错),
提取 ``async def run(surface)``, 注入现场 surface, await 之. 每次执行完丢弃 compiler.
后台完成经 ``on_background_done(result, next)`` 回调外发 (由 node 层接线到 notify
signal — 本模块不依赖 mindflow).
"""

from __future__ import annotations

import asyncio
import inspect
import traceback as _traceback
from io import StringIO
from typing import Any, Callable

from ghoshell_moss.contracts.logger import LoggerItf, get_moss_logger
from ghoshell_moss.core.codex.compiler import Compiler
from ghoshell_moss.deepseek_harness.launcher import DshConnection
from ghoshell_moss.deepseek_harness.session import DshSession
from ghoshell_moss.deepseek_harness.surfaces import DshConnectionSurface, DshSessionSurface

__all__ = ["DshRuntime"]

# 代码上文: 让编译源码里天然有 asyncio, 模型在函数体里可用 asyncio.to_thread.
_PROLOGUE = "import asyncio\n"
# 模型代码的入口函数名: async def run(<surface>).
_ENTRY = "run"


def _compile(code: str, print_fn: Callable[..., None]) -> Any:
    """把上文 + 模型补的正文编译成临时 module; 注入 print 写进本次 buffer (并发隔离)."""
    return Compiler(
        source=_PROLOGUE + code,
        filename="<moss_dsh_run>",
        local_injections={"print": print_fn},
    ).compiled


def _render_result(out: str, ret: Any) -> str:
    """把 stdout + 返回值渲染成 command 返回值."""
    parts: list[str] = []
    if out.strip():
        parts.append("--- stdout ---\n" + out.rstrip())
    if ret is not None:
        parts.append("--- result ---\n" + repr(ret))
    if not parts:
        return "(run returned None, no stdout)"
    return "\n".join(parts)


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


Surface = DshConnectionSurface | DshSessionSurface
BackgroundDoneHandler = Callable[[str, bool], None]


class DshRuntime:
    """持一个 dsh connection 及其 sessions, 拥有后台 task 与代码驱动入口.

    生命周期: ``async with runtime`` 或手动 ``__aenter__``/``__aexit__``.
    close() 幂等, 先 cancel 后台 task 再逐层关 sessions/connection.
    """

    def __init__(
        self,
        connection: DshConnection,
        *,
        on_background_done: BackgroundDoneHandler | None = None,
        logger: LoggerItf | None = None,
    ) -> None:
        self._connection = connection
        self._connection_surface = DshConnectionSurface(self._connection)
        self._sessions: dict[str, DshSession] = {}
        self._session_surfaces: dict[str, DshSessionSurface] = {}
        self._tasks: set[asyncio.Task] = set()
        self._session_tasks: dict[str, set[asyncio.Task]] = {}
        self._open_lock = asyncio.Lock()
        self._on_background_done = on_background_done
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
        self._session_surfaces.clear()
        await self._connection.__aexit__(None, None, None)

    # -- surfaces / sessions -- #

    def connection_surface(self) -> DshConnectionSurface:
        """connection 级管理面 (注入 connection 级 run)."""
        return self._connection_surface

    async def open_session(self, session_id: str) -> DshSessionSurface:
        """按 sessionId 接线一个 session (开流 + 起消费 task), 返回其驱动面.

        加锁 + 双重检查, 防并发两次 open 建两个 DshSession 泄漏消费 task.
        """
        surface = self._session_surfaces.get(session_id)
        if surface is not None:
            return surface
        async with self._open_lock:
            surface = self._session_surfaces.get(session_id)
            if surface is not None:
                return surface
            session = self._connection.create_session(session_id)
            await session.__aenter__()
            self._sessions[session_id] = session
            surface = DshSessionSurface(session)
            self._session_surfaces[session_id] = surface
            return surface

    def session_surface(self, session_id: str) -> DshSessionSurface | None:
        """已打开 session 的驱动面; 未打开返回 None."""
        return self._session_surfaces.get(session_id)

    async def close_session(self, session_id: str) -> None:
        """关闭一个 session: 先 cancel 其后台 task, 再关 session (停消费 task + 断流)."""
        session = self._sessions.pop(session_id, None)
        self._session_surfaces.pop(session_id, None)
        tasks = self._session_tasks.pop(session_id, set())
        for task in tasks:
            task.cancel()
        if session is not None:
            await session.close()

    def session_ids(self) -> list[str]:
        """当前已打开的 session ids."""
        return list(self._sessions)

    # -- 代码驱动 -- #

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

    def run_code_bg(self, surface: Surface, code: str, *, next: bool = False) -> str:
        """后台运行: 挂一个 task 跑 :meth:`run_code`, 完成时经 on_background_done 外发.

        立即返回. 结果不阻塞当前 turn; 若配了 on_background_done, 完成时回调
        (node 层接线成 notify signal, ``next`` 控制 queue-jump 与否).
        """
        task = asyncio.create_task(self._background(surface, code, next))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        if isinstance(surface, DshSessionSurface):
            self._session_tasks.setdefault(surface.session_id, set()).add(task)
        return "queued background run"

    async def _background(self, surface: Surface, code: str, next: bool) -> None:
        result = await self.run_code(surface, code)
        if self._on_background_done is not None:
            self._on_background_done(result, next)


def _injected_name(surface: Surface) -> str:
    """surface → 注入参数名 (connection / session), 用于编译错误提示."""
    return "connection" if isinstance(surface, DshConnectionSurface) else "session"
