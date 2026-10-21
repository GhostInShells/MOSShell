"""
Thinking 实现.

重构自历史文件 ./base_attention.py
"""

import asyncio
import contextlib
import logging
from typing import Callable, Awaitable
from typing_extensions import Self

import janus

from ghoshell_moss.core.blueprint.mindflow import (
    Action, ActionExitedException, Articulator, Attention, AttentionExitedException,
    Thinking, ThinkExitedException, ThinkingEffort
)
from ghoshell_moss.core.blueprint.moment import Moment, Observer
from ghoshell_moss.core.helpers import ThreadSafeEvent
from ghoshell_moss.contracts import get_moss_logger
from ._action import BaseAction, BaseArticulator, ApproveCallback

__all__ = ['BaseThinking']


class BaseThinking(Thinking):
    """Thinking 的基类实现.

    职责: 持有 attention / moment / observer, 把 thinking 与 action 通过
    logos_queue + events 装线在一起. gate 惰性创建, per-thinking 一个;
    articulator() 依据 gate 是否已实例化来区分 gated / non-gated 接入策略.

    从 attention 能拿到的都走 attention: effort 由 attention.draw_from() 的
    发起 impulse 提供; 生命周期/中断由 attention.wait_abort() 级联下来.
    """

    def __init__(
            self,
            *,
            attention: Attention,
            observer: Observer,
            put_action: Callable[[Action], None],
            mindflow_stop_event: ThreadSafeEvent,
            moment: Moment | None = None,
            logger: logging.Logger | None = None,
    ):
        self._attention = attention
        self._observer = observer
        self._put_action = put_action
        self._mindflow_stop_event = mindflow_stop_event
        # 初始帧来自 mindflow 预观测+折帧; 为 None 时首次访问 moment 才懒观测一帧.
        self._moment = moment
        self._logger = logger or get_moss_logger()
        self._log_prefix = f"<Thinking attention={attention.id}>"

        self._warrant: ApproveCallback | None = None
        self._stop_event = ThreadSafeEvent()
        self._started = False
        self._stopped = False
        # stop() 的"清理只做一次"标记 (停止位与清理分开, 见 stop()).
        self._cleaned = False
        # attention 的 abort 回调撤销函数.
        self._dispose_attention_abort: Callable[[], None] | None = None
        # 本帧签发的 action / articulator 的终止回调 —— 契约层次是 attention → thinking →
        # action/articulator, 所以这一层要能自己向下传导.
        # **必须先于 on_abort 初始化**: 若 attention 构造时已 abort, on_abort 会当场补发
        # _notify_stop, 它读 _stop_callbacks — 顺序颠倒会 AttributeError.
        self._stop_callbacks: list[Callable[[], None]] = []
        # 从属关系在**构造时**就成立, 不等 __aenter__: 构造与 enter 之间有真实窗口, 窗口里
        # 没人挂牌的从属者会安静地掉在退出级联之外 (与 Attention.on_abort 同一形状).
        self._dispose_attention_abort = self._attention.on_abort(self._notify_stop)
        self._lifecycle_task: asyncio.Task | None = None
        self._waiting_futures: list[asyncio.Future] = []
        # 每签发一个 action 就保留它的 stop event, 供清空(全部) / 对齐(最后一个)治理.
        self._action_stop_events: list[ThreadSafeEvent] = []

    def __repr__(self) -> str:
        return self._log_prefix

    # ── Thinking ABC ──────────────────────────────

    @property
    def attention(self) -> Attention:
        return self._attention

    @property
    def observer(self) -> Observer:
        return self._observer

    @property
    def moment(self) -> Moment:
        if self._moment is None:
            self._moment = self._observer.observe()
        return self._moment

    def observe(self) -> Moment:
        self._moment = self._observer.observe()
        return self._moment

    def effort(self) -> ThinkingEffort:
        return self._attention.impulse().thinking_effort

    def register_gate(self, warrant: Callable[[str], Awaitable[tuple[bool, str]]]):
        self._warrant = warrant

    def articulator(self, replan: bool = False, wait_action_done: bool = False) -> Articulator:
        """
        创建一个可以发布 logos 的 articulator, 与一个新的 BaseAction 成对.

        注册了 warrant (= gated 模式) 时, articulator 在 commit 时创建被持有的审批
        task 并 await 它裁决完整 logos, 通过才投递 action; 否则立即 put_action,
        logos 直接进 queue.
        """
        logos_queue: janus.Queue[str | None] = janus.Queue()
        compiled_event = ThreadSafeEvent()
        action_stop_event = ThreadSafeEvent()
        observed_event = ThreadSafeEvent()

        action = BaseAction(
            attention=self._attention,
            moments=self._observer,
            replaned=replan,
            logos_queue=logos_queue,
            compiled_event=compiled_event,
            observed_event=observed_event,
            action_stop_event=action_stop_event,
            mindflow_stop_event=self._mindflow_stop_event,
            thinking_stop_event=self._stop_event,
            notify_thinking_stop=self._notify_stop,
            logger=self._logger,
        )
        # 保留 action 的 stop event, 供 thinking 清空/对齐治理.
        self._action_stop_events.append(action_stop_event)
        # 生命周期从属本帧 (契约层次: attention → thinking → action/articulator): 构造时就挂牌,
        # 不等 enter —— 这一帧一停, 它签发的 action/articulator 立刻收线, 不必等谁被调度.
        action.bind_to_thinking_stop(self.on_stop)

        gated = self._warrant is not None
        if not gated:
            self._put_action(action)

        return BaseArticulator(
            moment=self.moment,
            logos_queue=logos_queue,
            compiled_event=compiled_event,
            observed_event=observed_event,
            action_stop_event=action_stop_event,
            warrant=self._warrant if gated else None,
            action=action,
            put_action=self._put_action if gated else None,
            logger=self._logger,
        )

    async def wait_until_done(self, *futures: asyncio.Future) -> None:
        ensured = []
        for future in futures:
            fut = asyncio.ensure_future(future)
            ensured.append(fut)
            if self.is_aborted():
                fut.cancel()
        self._waiting_futures.extend(ensured)
        await asyncio.gather(*ensured, return_exceptions=True)

    async def wait_actions_done(self) -> None:
        waits = []
        for action_stop_event in self._action_stop_events:
            waits.append(action_stop_event.wait())
        if len(waits) > 0:
            await asyncio.gather(*waits, return_exceptions=True)

    # ── MindflowStatement 生命周期 ────────────────

    def is_running(self) -> bool:
        return (
                self._started and not self._stopped
                and not self._mindflow_stop_event.is_set()
                and not self._attention.is_aborted()
        )

    def is_aborted(self) -> bool:
        return (
                self._stop_event.is_set()
                or self._attention.is_aborted()
                or self._mindflow_stop_event.is_set()
        )

    def abort(self, reason: str | Exception | None) -> None:
        self._attention.abort(reason)
        self._notify_stop()

    def abort_reason(self) -> str | None:
        return self._attention.abort_reason()

    async def stop(self) -> None:
        # 停止位与清理分开: 停止位可能已被 abort 回调**同步**置位 (见 _notify_stop),
        # 但等待中的 future 仍然要有人取消一次, 所以这里不能因为 event 已置位就早退.
        self._notify_stop()
        if self._cleaned:
            return
        self._cleaned = True
        if self._waiting_futures:
            for t in self._waiting_futures:
                t.cancel()
            _ = await asyncio.gather(*self._waiting_futures, return_exceptions=True)

    async def _wait_last_action_done(self) -> None:
        """等最后一个 action 停止 — 非公开, 供 __aexit__ 正常退出时重新对齐边界.

        只等最后一个 (不 wait 全部): 中间段可以 interleaved 超速, 但最后一帧的
        action 必须跑完 (observe 落盘), 否则 attention 会误判自然结束而失序.
        """
        if not self._action_stop_events:
            return
        last = self._action_stop_events[-1]
        if not last.is_set():
            await last.wait()

    async def wait_abort(self) -> None:
        await self._stop_event.wait()

    async def _lifecycle_aborted_monitor(self) -> None:
        """等"该收线"的两路来源, 到了就把这一帧收干净.

        不再等 attention: attention 的 abort 由 ``on_abort`` 回调**同步**置位 ``_stop_event``
        (见 ``_notify_stop``), 这里只是在同一事件上补做异步清理.
        """
        stop_waiter = asyncio.create_task(self._stop_event.wait(), name="thinking-stop-event")
        mindflow_waiter = asyncio.create_task(self._mindflow_stop_event.wait(), name="mindflow-stop-event")
        try:
            await asyncio.wait(
                {stop_waiter, mindflow_waiter},
                return_when=asyncio.FIRST_COMPLETED,
            )
        finally:
            # asyncio.wait 不管被取消时的遗留: 取消 monitor 不能把这两个 waiter 漏在循环里.
            for waiter in (stop_waiter, mindflow_waiter):
                if not waiter.done():
                    waiter.cancel()
        await self.stop()

    def on_stop(self, callback: Callable[[], None]) -> Callable[[], None]:
        """Register a stop callback for a child this frame issues. Synchronous, never lost.

        与 ``Attention.on_abort`` 同一形状: 先 append 再判读, 晚注册当场补发. 于是"已构造、
        还没被 enter"的 action / articulator 也稳稳挂在本帧的退出级联上 —— 这正是它们唯一的
        终点 (它们可能一次都没被调度过, 没有自己的收尾任务).
        """
        self._stop_callbacks.append(callback)
        if self._stop_event.is_set():
            try:
                self._stop_callbacks.remove(callback)
            except ValueError:
                return lambda: None
            callback()
            return lambda: None

        def _dispose() -> None:
            try:
                self._stop_callbacks.remove(callback)
            except ValueError:
                # 已经触发过了.
                pass

        return _dispose

    def _notify_stop(self) -> None:
        """本帧停止位落地, 并向下传导到本帧签发的 action / articulator.

        必须是同步的: 可能跑在 attention 的 abort 调用栈上, 而一个还没被调度就被取消的 waiter
        会让这次退出永远传不到. 对子对象同样同步收起 —— 只置停止位不够: 它们还各自等着自己的
        compiled/observed 落地, 而一个从没被调度过的对象只有这里能放行 (2026-10-13 的卡死).
        """
        self._stop_event.set()
        callbacks, self._stop_callbacks = self._stop_callbacks, []
        for callback in callbacks:
            try:
                callback()
            except Exception:
                self._logger.exception("%s stop callback %r failed", self._log_prefix, callback)

    async def __aenter__(self) -> Self:
        if self._started:
            return self
        self._started = True
        # 牌已在 __init__ 挂好 (构造即从属); 这里只起异步收尾的监视任务.
        self._lifecycle_task = asyncio.create_task(self._lifecycle_aborted_monitor())
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        if self._stopped:
            return None
        self._stopped = True
        try:
            # 正常退出 (无异常): 先等最后一个 action 跑完, 保证最后一帧被观测后
            # 才收线, 避免 thinking 先于 last action 退出导致 observe 丢失.
            if exc_val is None:
                await self._wait_last_action_done()
            await self.stop()
            if self._lifecycle_task and not self._lifecycle_task.done():
                self._lifecycle_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await self._lifecycle_task
                self._lifecycle_task = None
            if self._waiting_futures:
                for t in self._waiting_futures:
                    if not t.done():
                        t.cancel()
                _ = await asyncio.gather(*self._waiting_futures, return_exceptions=True)

            if exc_val:
                if isinstance(exc_val, (ActionExitedException, AttentionExitedException, ThinkExitedException)):
                    return True
                self._attention.abort(exc_val)
            return None
        finally:
            self._notify_stop()
            # 撤牌: 这一帧已收线, 之后的 abort 不该再回调到它.
            if self._dispose_attention_abort is not None:
                self._dispose_attention_abort()
                self._dispose_attention_abort = None
