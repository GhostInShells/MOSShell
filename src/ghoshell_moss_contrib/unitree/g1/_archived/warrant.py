"""
Warrant — 危险命令的统一封装事务.

语义:
  warrant = "授权令". 包裹一段操作, 三件事 race:
    1. coro 正常完成 → commit, fallback **不执行**, run() 返回 coro 的结果.
    2. scope 对应 abort 信号触发 → coro cancel + fallback → raise WarrantInterrupted.
    3. state token 失效(state DAG 切换) → coro cancel + fallback → raise WarrantInterrupted.

中断语义:
  被中断时, coro 在 await 点收到 CancelledError. 它 send 出去的物理 RPC 已经在飞行中,
  但下一行代码不会跑. 对 move 而言, 如果中断在 SetVelocity 调用之前, 等于这条 move
  从未发生; 如果中断在 SetVelocity 调用之后但 SetVelocity 本身已立即返回(它就是非阻塞),
  那 RPC 已发出, 然后由 fallback 接管(StopMove).

用法:
    async with bootstrap.warrant("locomotion") as w:
        await w.run(
            client.SetVelocity(vx, vy, vyaw, duration),
            fallback=lambda: client.SetVelocity(0, 0, 0),
        )

abort 触发:
    bootstrap.abort_scope("locomotion")  # 由按键 callback 调
    bootstrap.invalidate_state_token()    # state DAG 切换时调

scope 跟 channel 解耦. scope 是物理授权通道, 一个 scope 覆盖多个 channel 的命令.

实现注意:
  - 用 asyncio.Event 作为 abort 信号 + state_token 信号
  - 用 asyncio.wait(FIRST_COMPLETED) race coro task + abort_waiter + state_waiter
  - 中断时 task.cancel() + await task 等清理 + 跑 fallback (如有)
  - fallback 内部异常: log, 不二次 raise
  - coro 自己 raise (非 CancelledError): 直接传播, 不跑 fallback (那是 coro 自己的语义)

跟 channel runtime 关系:
  - warrant 不依赖 ChannelCtx / CommandUtil. 它是 bootstrap 级机制.
  - channel command 内部用 `async with bootstrap.warrant(scope) as w: await w.run(...)`.

线程安全:
  - warrant 实例是 per-call (async with 出来一个新的), 单 coroutine 内用. 不需要锁.
  - abort/invalidate 跨线程: 但 asyncio.Event 本身是非线程安全. 我们用 loop.call_soon_threadsafe
    封装 Event.set, 见 abort_scope / invalidate_state_token.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Awaitable, Callable

logger = logging.getLogger(__name__)


class WarrantInterrupted(Exception):
    """warrant 被中断 — 来自 scope abort 或 state token 失效."""

    def __init__(self, scope: str, reason: str):
        super().__init__(f"warrant '{scope}' interrupted: {reason}")
        self.scope = scope
        self.reason = reason


# ═══════════════════════════════════════════════════════════════════════════════
# 模块级状态 — scope 注册表, state token
# ═══════════════════════════════════════════════════════════════════════════════

# scope 名 -> asyncio.Event (set 表示该 scope 被 abort).
# 注意 asyncio.Event 绑定到创建时的 loop. 我们假设 bootstrap 跑在单一 loop 里.
_scope_events: dict[str, asyncio.Event] = {}

# state token. 每次 invalidate 时换一个新 Event, 旧 Event 被 set.
# 这样设计是为了避免"event set 后没清理就重用"的状态污染.
_state_token_event: asyncio.Event | None = None

# loop 引用. abort_scope 跨线程调时用. bootstrap 完成后由 channel runtime 设置.
_loop: asyncio.AbstractEventLoop | None = None


def set_event_loop(loop: asyncio.AbstractEventLoop) -> None:
    """由 channel runtime / main.py 调一次. 给跨线程 abort 用.

    如果不调, abort_scope / invalidate_state_token 只能在 loop 内调.
    """
    global _loop
    _loop = loop


def _get_or_create_scope_event(scope: str) -> asyncio.Event:
    """获取/创建一个 scope 的 abort Event. 必须在 loop 线程调."""
    ev = _scope_events.get(scope)
    if ev is None:
        ev = asyncio.Event()
        _scope_events[scope] = ev
    return ev


def _get_or_create_state_event() -> asyncio.Event:
    global _state_token_event
    if _state_token_event is None:
        _state_token_event = asyncio.Event()
    return _state_token_event


# ═══════════════════════════════════════════════════════════════════════════════
# Warrant 类 — async context manager
# ═══════════════════════════════════════════════════════════════════════════════


class Warrant:
    """单个授权事务作用域. 通过 warrant(scope) 创建.

    生命周期:
      __aenter__ — 记录当前 scope 的 abort Event + 当前 state Event 引用(快照).
                   关键: 即使 abort Event 在进入后被 set, 也会被 race 检测到.
      run(coro, fallback) — 启动 coro + 三回调 race.
      __aexit__ — 清理.
    """

    def __init__(self, scope: str):
        self._scope = scope
        self._abort_event: asyncio.Event | None = None
        self._state_event: asyncio.Event | None = None

    async def __aenter__(self) -> "Warrant":
        # 快照当前 events. 如果稍后 invalidate_state_token, 旧的 _state_event 仍是 set 的.
        self._abort_event = _get_or_create_scope_event(self._scope)
        self._state_event = _get_or_create_state_event()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        # 清理是 run 内做的(task cancel). 这里不需要额外动作.
        # 不吞异常 — WarrantInterrupted 应传播.
        return None

    async def run(
        self,
        coro: Awaitable[Any],
        *,
        fallback: Callable[[], Awaitable[None]] | Callable[[], None] | None = None,
    ) -> Any:
        """在事务保护下执行 coro.

        Args:
            coro: 要执行的协程对象.
            fallback: 中断时的安全恢复. 可以是 async 或 sync 函数, 无参数.

        Returns:
            coro 的返回值(如果正常完成).

        Raises:
            WarrantInterrupted: 被 abort/invalidate 中断.
            其他: coro 自己 raise 的异常, 直接传播(不跑 fallback).
        """
        assert self._abort_event is not None, "run() called outside async with"
        assert self._state_event is not None

        # 快速失败: 如果 abort 已经 set, 直接 cancel
        if self._abort_event.is_set():
            await self._run_fallback(fallback, reason=f"scope '{self._scope}' already aborted")
            raise WarrantInterrupted(self._scope, "already aborted at entry")
        if self._state_event.is_set():
            await self._run_fallback(fallback, reason="state token already invalidated")
            raise WarrantInterrupted(self._scope, "state token already invalidated at entry")

        coro_task = asyncio.ensure_future(coro)
        abort_task = asyncio.ensure_future(self._abort_event.wait())
        state_task = asyncio.ensure_future(self._state_event.wait())

        try:
            done, pending = await asyncio.wait(
                {coro_task, abort_task, state_task},
                return_when=asyncio.FIRST_COMPLETED,
            )
        except asyncio.CancelledError:
            # 外部 cancel 我们这个 await — 把 child 都收掉, 再传播
            coro_task.cancel()
            abort_task.cancel()
            state_task.cancel()
            for t in (coro_task, abort_task, state_task):
                try:
                    await t
                except Exception:
                    pass
            raise

        # 清理 pending
        for t in pending:
            t.cancel()
        # await pending 直到 CancelledError 跑完(避免警告)
        for t in pending:
            try:
                await t
            except (asyncio.CancelledError, Exception):
                pass

        # 判断完成原因
        if coro_task in done:
            # coro 正常完成(可能含 exception, 让它传播)
            return coro_task.result()

        # 中断
        if abort_task in done:
            reason = f"scope '{self._scope}' aborted"
        else:
            reason = "state token invalidated"

        await self._run_fallback(fallback, reason=reason)
        raise WarrantInterrupted(self._scope, reason)

    async def _run_fallback(
        self,
        fallback: Callable[[], Awaitable[None]] | Callable[[], None] | None,
        reason: str,
    ) -> None:
        if fallback is None:
            return
        try:
            result = fallback()
            if asyncio.iscoroutine(result):
                await result
        except Exception:
            logger.exception("warrant '%s' fallback raised (reason=%s)", self._scope, reason)


# ═══════════════════════════════════════════════════════════════════════════════
# 公共 API — 跟 bootstrap 协作
# ═══════════════════════════════════════════════════════════════════════════════


def warrant(scope: str) -> Warrant:
    """创建一个 warrant 作用域. 用作 async with 上下文管理器."""
    return Warrant(scope)


def abort_scope(scope: str) -> None:
    """触发某 scope 的中断. 该 scope 下所有正在跑的 warrant.run 都会被 cancel + fallback.

    可从任意线程调. 内部用 loop.call_soon_threadsafe.

    abort 之后 scope 保持"已 abort"状态, 后续进 warrant 会立刻失败. 用 clear_scope_abort 复位.
    """
    def _do_set():
        ev = _get_or_create_scope_event(scope)
        ev.set()
        logger.debug("warrant: abort_scope(%s)", scope)

    _schedule_in_loop(_do_set)


def clear_scope_abort(scope: str) -> None:
    """复位 scope abort 标志. 通常在人类授权或冷却期结束后调."""
    def _do_clear():
        ev = _scope_events.get(scope)
        if ev is not None:
            ev.clear()
            logger.debug("warrant: clear_scope_abort(%s)", scope)

    _schedule_in_loop(_do_clear)


def invalidate_state_token() -> None:
    """state DAG 切换时调. 所有正在跑的 warrant.run 中断.

    切换完成后, 调用方应当用 reset_state_token 装入新 token, 后续 warrant 才能正常进入.
    """
    def _do_invalidate():
        global _state_token_event
        ev = _get_or_create_state_event()
        ev.set()
        # 直接换新 Event, 让后续 warrant 拿到一个未 set 的
        _state_token_event = asyncio.Event()
        logger.debug("warrant: state token invalidated")

    _schedule_in_loop(_do_invalidate)


def reset_state_token() -> None:
    """显式重置 state token. 在 state DAG 切换完成后调."""
    def _do_reset():
        global _state_token_event
        _state_token_event = asyncio.Event()
        logger.debug("warrant: state token reset")

    _schedule_in_loop(_do_reset)


def _schedule_in_loop(fn: Callable[[], None]) -> None:
    """在 loop 线程里跑 fn. 如果 loop 已设置, 用 call_soon_threadsafe; 否则直接调."""
    global _loop
    if _loop is not None and _loop.is_running():
        _loop.call_soon_threadsafe(fn)
    else:
        # 没设置 loop 或 loop 没跑 — 假设调用方就在 loop 线程
        fn()


def _reset_all_for_testing() -> None:
    """测试 hook: 清空所有 scope events + state event + loop."""
    global _state_token_event, _loop
    _scope_events.clear()
    _state_token_event = None
    _loop = None
