"""DoloresRun — Dolores thinking 交易的 run 对象 (async with 边界 + events() 消费).

本文件实现 thinking 交易的 run 对象, 取代早期 articulate run 的设计 (seq watermark /
done 权威迁移 — 已被 turn/end 毒丸 + enter task finally 取代). 生命周期显式
(async with 边界, 无隐式逻辑, 非 async generator):

    async with ego.run_thinking(thinking) as run:
        async for event in run.events():
            ...

``async with`` 是 transaction 边界 (aenter 开 / aexit 关), ``async for`` 消费事件流.

── 拓扑: 两路生产, 一路消费 ──────────────────────────────────────
    [MOSS] run
      生产① events:  session catch-all 监听 → put event ──┐
      生产② done:    enter task → thinking/enter RPC → finally 塞毒丸 ──┼→ queue → events()
      强关闭:        __aexit__ → cancel enter task + 解绑 + 补发 exit   ┘

── 生命周期契约 (review 约束, 2026-08-28) ─────────────────────────
  aenter: 先绑监听队列 (避免丢 enter 广播事件) → 再建 enter task (async).
  aexit : cancel enter task (未完成) → 解绑监听 → 补发 thinking/exit (即使 enter 未通过,
          阻塞到确认, 带超时 fail-safe) → 异常时 thinking.abort(reason).
  毒丸 = 正常/异常 done (enter task finally 保证); enter 异常经毒丸传输 (consumer raise).

── 数据面 (迭代结束后可读) ─────────────────────────────────────────
  usage / messages / head 待后续 — 早期 seq watermark 设计被本实现取代, 见 git log.
"""

from __future__ import annotations

import asyncio
import contextlib
from typing import TYPE_CHECKING, Any, AsyncIterator, Callable

from typing_extensions import Self

if TYPE_CHECKING:
    from ghoshell_moss.core.blueprint.mindflow import Thinking
    from ghoshell_moss.deepseek_harness.session import DshSession
    from ghoshell_moss.deepseek_harness.types.session_events import SessionEvent

__all__ = ["DoloresRun"]

# 毒丸 sentinel: enter task 退出时入队, events() 读到即终止.
# enter 异常经毒丸携带 (consumer raise). 与 shutdown 分工: 毒丸 = 正常/异常 done.
_POISON = object()


class DoloresRun:
    """Dolores thinking 交易 run 对象 — async with 边界 + events() 消费.

    生命周期显式 (无隐式逻辑, 非 async generator):
      __aenter__: 先绑监听队列 (避免丢 enter 广播事件) → 再建 enter task (async).
      __aexit__ : cancel enter task (未完成) → 解绑监听 → 补发 exit (阻塞到确认, 带超时
                  fail-safe) → 异常时 thinking.abort(reason).
      events()  : 队列消费; 毒丸终止; enter 异常经毒丸传输 (consumer raise).

    构造: ego (Duck-typed — session / _rpc_thinking_enter / _rpc_thinking_exit /
          _articulating / _logger) + thinking. 窄依赖, 测试可用轻量 mock.
    """

    def __init__(self, ego: Any, thinking: "Thinking") -> None:
        self._ego = ego
        self._thinking = thinking
        self._queue: "asyncio.Queue[Any]" = asyncio.Queue()
        self._dispose_listener: "Callable[[], None] | None" = None
        self._enter_task: "asyncio.Task[None] | None" = None
        self._enter_error: Exception | None = None

    # ── transaction 边界 ──────────────────────────────────────────

    async def __aenter__(self) -> Self:
        """开 transaction. 先绑监听 (避免丢 enter 广播), 再建 enter task."""
        self._ego._articulating = True
        self._dispose_listener = self._ego.session.on_session_event("*", self._on_event)
        self._enter_task = asyncio.create_task(self._drive_enter())
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """关 transaction. cancel enter task → 解绑 → 补发 exit → abort (异常时)."""
        task = self._enter_task
        if task is not None and not task.done():
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
        if self._dispose_listener is not None:
            self._dispose_listener()
        # 补发 exit — enter 未通过也要发 (清理 plugin 侧状态), 阻塞到确认 (带超时 fail-safe).
        await self._ego._rpc_thinking_exit()
        self._ego._articulating = False
        reason = exc_val if exc_val is not None else self._enter_error
        if reason is not None:
            self._thinking.abort(reason)

    # ── 事件流 ─────────────────────────────────────────────────────

    async def events(self) -> "AsyncIterator[SessionEvent]":
        """从队列读原始 session event; 毒丸终止; enter 异常经毒丸传输 (raise)."""
        while True:
            item = await self._queue.get()
            if item is _POISON:
                if self._enter_error is not None:
                    raise self._enter_error
                return
            yield item

    # ── 内部 ──────────────────────────────────────────────────────

    async def _on_event(self, event: "SessionEvent") -> None:
        """catch-all 回调: 入队 (async 包装 — on_session_event 消费方 await)."""
        self._queue.put_nowait(event)

    async def _drive_enter(self) -> None:
        """enter task: thinking/enter RPC. finally 必塞毒丸 (成功/异常/取消都塞)."""
        try:
            await self._ego._rpc_thinking_enter(self._thinking)
        except asyncio.CancelledError:
            raise
        except Exception as error:
            self._enter_error = error
        finally:
            self._queue.put_nowait(_POISON)
