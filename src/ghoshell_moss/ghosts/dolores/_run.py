"""DoloresRun — Dolores thinking 交易的 run 对象 (async with 交易边界 + events() 事件流).

async with 是交易边界 (aenter 开 / aexit 关), events() 是被动拉的原始 session event 流:

    async with ego.run_thinking(thinking) as run:
        async for event in run.events():
            ...  # 消费方在 turn/end / wait_next_moment (yield) 处 break 收线

生命周期契约:
  aenter: 先绑 session catch-all 监听 (避免丢 enter 广播) → 再建 enter task (async).
  aexit : cancel enter task → 解绑监听 → 补发 thinking/exit (enter 未通过也发, 带超时
          fail-safe) → 异常时 thinking.abort(reason).
  events(): 队列消费; 毒丸只承载 enter 异常 (正常路径由消费方 turn/end break 收线).

yield tool (wait_next_moment): 消费方认出 tool/call = wait_next_moment → break 收线并置
run.yielded → exit 时 plugin 不 cancel (tool 留 pending, 下一轮 enter 用 moment 解锁).
moment 的生产归 mindflow 正常 loop, 不归 run.
"""

from __future__ import annotations

import asyncio
import contextlib
from typing import TYPE_CHECKING, Any, AsyncIterator, Callable
from typing_extensions import Self

if TYPE_CHECKING:
    from ._ego import DoloresEgo
    from ghoshell_moss.core.blueprint.mindflow import Thinking
    from ghoshell_moss.deepseek_harness.types.session_events import SessionEvent

__all__ = ["DoloresRun"]

# 毒丸 sentinel: enter task 异常时入队, events() 读到即抛 _enter_error 终止.
# 正常路径不塞 — 消费方在 turn/end 处 break 收线, 毒丸只管 enter 异常.
_POISON = object()


class DoloresRun:
    """Dolores thinking 交易 run 对象 — async with 交易边界 + events() 事件流.

    生命周期契约见模块 docstring. 构造: ego (DoloresEgo, 供 session/enter/exit 窄桥)
    + thinking + thinking_event (ego 持有的"交易进行中" Event — run aenter/aexit set/clear,
    供 ego self-wake gate 读取). 依赖经公有接口.
    """

    def __init__(
            self,
            ego: "DoloresEgo",
            thinking: "Thinking",
            thinking_event: asyncio.Event,
    ) -> None:
        self._ego = ego
        self._thinking = thinking
        self._queue: "asyncio.Queue[Any]" = asyncio.Queue()
        self._dispose_listener: "Callable[[], None] | None" = None
        self._enter_task: "asyncio.Task[None] | None" = None
        self._enter_error: Exception | None = None
        self._thinking_event: asyncio.Event = thinking_event
        # yield 收线标记: 消费方认出 tool/call == wait_next_moment 并 break 时置 True,
        # __aexit__ 经 exit_thinking(yielded=...) 传给 plugin — yield 时绝不 cancel.
        self.yielded = False

    # ── transaction 边界 ──────────────────────────────────────────

    async def __aenter__(self) -> Self:
        """开 transaction. 先绑监听 (避免丢 enter 广播), 再建 enter task."""
        self._thinking_event.set()
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
        # yielded 标记: 本次 break 是否 yield 收线, plugin 据此决定是否 cancel.
        await self._ego.exit_thinking(yielded=self.yielded)
        self._thinking_event.clear()
        reason = exc_val if exc_val is not None else self._enter_error
        if reason is not None:
            self._thinking.abort(reason)

    # ── 事件流 ─────────────────────────────────────────────────────

    async def events(self) -> "AsyncIterator[SessionEvent]":
        """从队列读原始 session event, 消费方在 turn/end 处 break 收线.

        毒丸只在 enter 异常时钉下, 读到即抛 _enter_error 终止; 正常路径不终止.
        """
        while self._thinking_event.is_set():
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
        """enter task: thinking/enter RPC. 只在异常时塞毒丸.

        正常路径不塞毒丸 — turn 由 dsh 自行 run, live moment 的事件经 catch-all 监听
        流式入队, 消费方在 turn/end 处 break 收线. 毒丸只在 enter 异常时钉下收尾标志,
        让 events() 能借 _enter_error 向消费方抛出. 若在正常路径也塞毒丸, 会抢在
        模型产出 logos 之前终止事件流 — enter RPC 返回的时刻模型尚未生成任何帧.
        """
        try:
            await self._ego.enter_thinking(self._thinking)
        except asyncio.CancelledError:
            raise
        except Exception as error:
            self._enter_error = error
            self._queue.put_nowait(_POISON)
