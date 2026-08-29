"""DoloresRun — Dolores thinking 交易的 run 对象 (async with 边界 + events() 消费).

本文件实现 thinking 交易的 run 对象, 取代早期 articulate run 的设计 (seq watermark /
done 权威迁移 — 已被 turn/end 收线 + enter 毒丸取代). 生命周期显式
(async with 边界, 无隐式逻辑, 非 async generator):

    async with ego.run_thinking(thinking) as run:
        async for event in run.events():
            ...

``async with`` 是 transaction 边界 (aenter 开 / aexit 关), ``async for`` 消费事件流.

── 拓扑: 两路生产, 一路消费 ──────────────────────────────────────
    [MOSS] run
      生产① events:  session catch-all 监听 → put event ──┐
      生产② 收尾:    enter task → thinking/enter RPC; 异常时塞毒丸 ──┼→ queue → events()
      强关闭:        __aexit__ → cancel enter task + 解绑 + 补发 exit   ┘
      正常收线:      消费方在 turn/end 处 break — 毒丸只管 enter 异常, 不管正常 done.

── 生命周期契约 (review 约束, 2026-08-28) ─────────────────────────
  aenter: 先绑监听队列 (避免丢 enter 广播事件) → 再建 enter task (async).
  aexit : cancel enter task (未完成) → 解绑监听 → 补发 thinking/exit (即使 enter 未通过,
          阻塞到确认, 带超时 fail-safe) → 异常时 thinking.abort(reason).
  毒丸 = enter 异常 done (enter task 出错时钉下); 正常路径由消费方在 turn/end break 收线,
  enter 异常经毒丸传输 (consumer raise).

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

# 毒丸 sentinel: enter task 异常时入队, events() 读到即抛 _enter_error 终止.
# 正常路径不塞 — 消费方在 turn/end 处 break 收线, 毒丸只管 enter 异常.
_POISON = object()


class DoloresRun:
    """Dolores thinking 交易 run 对象 — async with 边界 + events() 消费.

    生命周期显式 (无隐式逻辑, 非 async generator):
      __aenter__: 先绑监听队列 (避免丢 enter 广播事件) → 再建 enter task (async).
      __aexit__ : cancel enter task (未完成) → 解绑监听 → 补发 exit (阻塞到确认, 带超时
                  fail-safe) → 异常时 thinking.abort(reason).
      events()  : 队列消费; 毒丸终止; enter 异常经毒丸传输 (consumer raise).

    构造: ego (Duck-typed — session / articulating / enter_thinking / exit_thinking)
    + thinking. 依赖经公有接口, 不访问 ego 私有成员; 测试可用轻量 mock.
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
        self._ego.articulating = True
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
        await self._ego.exit_thinking()
        self._ego.articulating = False
        reason = exc_val if exc_val is not None else self._enter_error
        if reason is not None:
            self._thinking.abort(reason)

    # ── 事件流 ─────────────────────────────────────────────────────

    async def events(self) -> "AsyncIterator[SessionEvent]":
        """从队列读原始 session event, 消费方在 turn/end 处 break 收线.

        毒丸只在 enter 异常时钉下, 读到即抛 _enter_error 终止; 正常路径不终止.
        """
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
